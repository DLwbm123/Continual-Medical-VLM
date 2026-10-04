"""Mechanism tests use fabricated tensors, never clinical concept annotations."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

from core import (ClinicalModel, choose_replay, classification_metrics, diagnosis_kd,
                  frozen_teacher, masked_concept_loss, multi_positive_alignment,
                  restore_checkpoint, save_checkpoint, supported_concept_kd,
                  validate_group_separation)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.2))
        self.text = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
        self.logit_scale = nn.Parameter(torch.tensor(1.))

    def encode_image(self, x, normalize=False):
        return self.visual(x)


class CoreTests(unittest.TestCase):
    def test_missing_na_unassessable_are_not_negative(self):
        for _reason in ["not_annotated", "not_applicable", "not_assessable"]:
            logits = torch.tensor([[1.], [2.]], requires_grad=True)
            truth = {"echo": {"value": torch.full_like(logits, float("nan")), "valid": torch.zeros_like(logits, dtype=torch.bool)}}
            loss = masked_concept_loss({"echo": logits}, truth, [{"id": "echo", "type": "binary"}], logits)
            self.assertTrue(torch.isfinite(loss)); self.assertEqual(loss.item(), 0.)
            loss.backward(); self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_ce_and_bce_masks_and_group_normalization(self):
        c = torch.zeros(2, 3, requires_grad=True)
        b = torch.zeros(2, 2, requires_grad=True)
        truth = {"shape": {"value": torch.tensor([1, -1]), "valid": torch.tensor([True, False])},
                 "echo": {"value": torch.tensor([[1., 0.], [float("nan"), float("nan")]]),
                          "valid": torch.tensor([[True, True], [False, False]])}}
        loss = masked_concept_loss({"shape": c, "echo": b}, truth,
            [{"id": "shape", "type": "categorical"}, {"id": "echo", "type": "multilabel"}], c)
        self.assertAlmostEqual(loss.item(), (np.log(3) + np.log(2)) / 2, places=6)
        loss.backward(); self.assertEqual(c.grad[1].abs().sum(), 0); self.assertEqual(b.grad[1].abs().sum(), 0)

    def test_duplicate_descriptions_are_multi_positive(self):
        image = torch.ones(2, 4, requires_grad=True)
        loss = multi_positive_alignment(image, torch.ones(2, 4), torch.ones(2, 2, dtype=torch.bool), 2.)
        self.assertAlmostEqual(loss.item(), 0., places=6)

    def test_teacher_is_frozen_and_identical_eval_kd_is_zero(self):
        model = ClinicalModel(TinyBackbone(), 4, {"a": 2}).eval()
        teacher = frozen_teacher(model)
        x = torch.randn(5, 4)
        y = model(x, "a")[0]; t = teacher(x, "a")[0]
        self.assertLess(abs(diagnosis_kd(y, t).item()), 1e-6)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))

    def test_teacher_wrong_or_unknown_concepts_are_not_distilled(self):
        s = {"shape": torch.randn(2, 2, requires_grad=True)}
        t = {"shape": torch.tensor([[5., -5.], [5., -5.]])}
        targets = {"shape": {"value": torch.tensor([1, 0]), "valid": torch.tensor([True, False])}}
        loss, coverage = supported_concept_kd(s, t, targets, [{"id": "shape", "type": "categorical"}], s["shape"])
        self.assertEqual(coverage["shape"], 0); self.assertEqual(loss.item(), 0)
        with self.assertRaises(ValueError):
            supported_concept_kd(s, {"other": t["shape"]}, targets, [{"id": "shape", "type": "categorical"}], s["shape"])

    def test_expanding_tasks_does_not_reset_old_weights(self):
        model = ClinicalModel(TinyBackbone(), 4, {"a": 2})
        old = copy.deepcopy(model.state_dict())
        model.add_task("b", 3); model.acquire("b")
        for key, value in old.items():self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertFalse(any(p.requires_grad for p in model.diagnosis["a"].parameters()))
        with self.assertRaises(ValueError):model.add_task("a", 2)

    def test_replay_ids_shared_and_train_only(self):
        rows = [{"sample_id": str(i), "split": "train", "diagnosis_label": i % 3} for i in range(30)]
        expected = choose_replay(rows, 2, 0)
        for _method in ["P2", "P3", "P4"]:self.assertEqual(expected, choose_replay(rows[::-1], 2, 0))
        with self.assertRaises(ValueError):choose_replay([dict(rows[0], split="test")], 1, 0)

    def test_patient_split_overlap_detected_unknowns_not_claimed_verified(self):
        with self.assertRaises(ValueError):
            validate_group_separation([{"patient_id": "fixture", "split": "train"}, {"patient_id": "fixture", "split": "test"}])
        self.assertFalse(validate_group_separation([{"patient_id": None, "split": "train"}])["verified"])

    def test_single_class_auc_is_na_with_reason(self):
        result = classification_metrics([0, 0], [[.9, .1], [.8, .2]])
        self.assertIsNone(result["macro_ovr_auc"]); self.assertIsNotNone(result["auc_undefined_reason"])

    def test_inference_accepts_images_and_task_not_truth_or_filename(self):
        model = ClinicalModel(TinyBackbone(), 4, {"a": 2}).eval()
        x = torch.randn(2, 4)
        for forbidden in ["labels", "concepts", "filenames", "reports"]:
            with self.assertRaises(TypeError):model(x, "a", **{forbidden: ["malignant.png"]})
        model.train(); self.assertFalse(model.backbone.text.training)

    def test_resume_matches_uninterrupted_stochastic_training(self):
        torch.manual_seed(0)
        model = ClinicalModel(TinyBackbone(), 4, {"a": 2})
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        generator = torch.Generator().manual_seed(17)
        data = torch.arange(80).reshape(20, 4).float() / 80
        def step():
            model.train(); ix = torch.randint(20, (4,), generator=generator)
            optimizer.zero_grad(); loss = model(data[ix], "a")[0].square().mean(); loss.backward(); optimizer.step()
        step(); step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pt"
            save_checkpoint(path, model, optimizer, 2, generator, {})
            step(); step(); expected = copy.deepcopy(model.state_dict())
            self.assertEqual(restore_checkpoint(path, model, optimizer, generator), 2)
            step(); step()
            for key, value in expected.items():self.assertTrue(torch.equal(value, model.state_dict()[key]), key)


if __name__ == "__main__":unittest.main()
