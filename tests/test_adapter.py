"""Small artificial image files test data boundaries, not medical validity."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.audit_classification import audit


class AdapterTests(unittest.TestCase):
    def make_fixture(self, root, duplicate=False, mask=False):
        data = root / "data"; task = data / "fixture"; task.mkdir(parents=True)
        split, inventory = {}, []
        for phase in ["train", "val", "test"]:
            name = "image_mask.png" if mask else ("train.png" if duplicate else phase + ".png")
            p = task / name
            Image.new("RGB", (8, 8), color=(64, 64, 64)).save(p)
            split[phase] = [[name, 0, "artificial class"]]
            inventory.append({"relative_path": "fixture/" + name, "role_hint": "ordinary_image_candidate", "bytes": p.stat().st_size})
        (task / "split_fixture.json").write_text(json.dumps(split))
        listing = root / "inventory.jsonl"
        listing.write_text("".join(json.dumps(r) + "\n" for r in inventory))
        return data, listing

    def test_manifest_preserves_missingness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, listing = self.make_fixture(root)
            result = audit(data, root / "audit", listing)
            self.assertEqual(result["status"], "passed_classification_structure")
            row = json.loads((root / "audit/manifest.jsonl").read_text().splitlines()[0])
            self.assertIsNone(row["patient_id"]); self.assertEqual(row["concept_annotations"], [])

    def test_same_source_path_leakage_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, listing = self.make_fixture(root, duplicate=True)
            self.assertEqual(audit(data, root / "audit", listing)["status"], "needs_review")

    def test_mask_excluded_from_classification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, listing = self.make_fixture(root, mask=True)
            with self.assertRaises(ValueError):audit(data, root / "audit", listing)


if __name__ == "__main__":unittest.main()
