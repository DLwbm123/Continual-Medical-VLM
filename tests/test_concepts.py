import json
import unittest
from pathlib import Path
import torch
from concepts import targets_for, vocabulary_logits, concept_metrics
from core import masked_concept_loss, validate_group_separation
from tools.audit_breast import encode_annotation, assign_splits


class ConceptTests(unittest.TestCase):
    def setUp(self):
        self.specs = json.loads((Path(__file__).parents[1] / "schemas/breast_v1.json").read_text())["groups"]

    def test_source_multiselect_and_missing_semantics(self):
        margin = next(g for g in self.specs if g["id"] == "margin")
        a = encode_annotation("not circumscribed - angular&indistinct", margin)
        self.assertEqual(a["value"], [0, 1, 1, 0, 0])
        for raw in ["not applicable", "not available", "indefinable", ""]:
            self.assertFalse(encode_annotation(raw, margin)["valid"])
        halo = next(g for g in self.specs if g["id"] == "halo")
        self.assertEqual(encode_annotation("no", halo)["value"], [0])

    def test_all_missing_batch_is_differentiable_zero(self):
        rows = [{"concept_annotations": {}}] * 2
        targets = targets_for(rows, self.specs)
        logits = {g["id"]: torch.zeros(2, g["outputs"], requires_grad=True) for g in self.specs}
        loss = masked_concept_loss(logits, targets, self.specs, logits["shape"])
        self.assertEqual(loss.item(), 0); loss.backward()

    def test_fixed_vocabulary_uses_no_sample_truth_or_names(self):
        g = next(g for g in self.specs if g["id"] == "halo")
        features = torch.eye(2)
        out = vocabulary_logits(features, {"halo": torch.eye(2)}, [g], 1.)
        self.assertTrue(torch.equal(out["halo"], torch.tensor([[1.], [-1.]])))
        # Both same-label images get the same positive vocabulary target, never each other's negative.
        y = {"halo": {"value": torch.ones(2, 1), "valid": torch.ones(2, 1, dtype=torch.bool)}}
        duplicated = {"halo": torch.ones(2, 1)}
        self.assertAlmostEqual(masked_concept_loss(duplicated, y, [g], features).item(), 0.3132616875)

    def test_split_is_deterministic_disjoint_and_keeps_case_units(self):
        def rows(): return [{"sample_id": str(i), "patient_id": str(i), "diagnosis_label": i % 2} for i in range(40)]
        a, b = assign_splits(rows()), assign_splits(rows()[::-1])
        self.assertEqual({r["sample_id"]: r["split"] for r in a}, {r["sample_id"]: r["split"] for r in b})
        self.assertTrue(validate_group_separation(a)["verified"])

    def test_rare_category_is_not_reported_as_defined_full_auc(self):
        g = self.specs[0]
        rows = [{"concept_annotations": {"shape": encode_annotation("oval", g)}}] * 2
        m = concept_metrics(rows, {"shape": [[.1, .8, .1]] * 2}, [g])["shape"]
        self.assertIsNone(m["macro_ovr_auc"]); self.assertIsNone(m["macro_f1"])


if __name__ == "__main__": unittest.main()
