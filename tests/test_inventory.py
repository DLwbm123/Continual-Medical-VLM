"""Tests use fabricated filesystem fixtures, not real medical data."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'inventory_data.py'
spec = importlib.util.spec_from_file_location('inventory_data', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.gpu_inventory = lambda: {'available': False, 'test_fixture': True}

class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'data'
        self.root.mkdir()
        self.out = self.base / 'report'
    def tearDown(self):
        self.tmp.cleanup()
    def test_no_label_inference_and_read_only(self):
        p = self.root / 'train' / 'malignant'
        p.mkdir(parents=True)
        f = p / 'x.png'; f.write_bytes(b'fixture not an actual image')
        before = hashlib.sha256(f.read_bytes()).hexdigest()
        s = module.run([self.root], self.out)
        row = json.loads((self.out / 'file_inventory.jsonl').read_text())
        self.assertIsNone(row['label']); self.assertIsNone(row['split'])
        self.assertEqual(before, hashlib.sha256(f.read_bytes()).hexdigest())
        self.assertFalse(s['images_decoded'])
    def test_output_under_root_rejected(self):
        with self.assertRaises(ValueError):
            module.run([self.root], self.root / 'output')
    def test_overlap_rejected(self):
        child = self.root / 'child'; child.mkdir()
        with self.assertRaises(ValueError):
            module.run([self.root, child], self.out)
    def test_no_overwrite(self):
        self.out.mkdir(); (self.out / 'keep.txt').write_text('preserve')
        with self.assertRaises(ValueError):
            module.run([self.root], self.out)
        self.assertEqual((self.out / 'keep.txt').read_text(), 'preserve')
    def test_truncation(self):
        for n in range(3): (self.root / f'{n}.jpg').write_bytes(b'x')
        s = module.run([self.root], self.out, max_files=2)
        self.assertEqual(s['roots'][0]['coverage'], 'partial_file_limit')
        self.assertEqual(s['roots'][0]['physical_files_scanned'], 2)
    def test_duplicates_and_headers_only(self):
        (self.root / 'a.png').write_bytes(b'same')
        (self.root / 'b_mask.png').write_bytes(b'same')
        (self.root / 'labels.csv').write_text('id,diagnosis\nprivate_identifier,private_value\n')
        module.run([self.root], self.out, hash_images=True)
        d = json.loads((self.out / 'exact_duplicate_candidates.json').read_text())
        self.assertEqual(len(d), 1)
        headers = (self.out / 'metadata_field_candidates.json').read_text()
        self.assertNotIn('private_identifier', headers)
        self.assertIn('diagnosis', headers)
    def test_symlink_skipped(self):
        target = self.base / 'external.jpg'; target.write_bytes(b'outside')
        (self.root / 'linked.jpg').symlink_to(target)
        s = module.run([self.root], self.out)
        self.assertEqual(s['roots'][0]['symlinks_skipped'], 1)
        self.assertEqual(s['roots'][0]['physical_files_scanned'], 0)
    def test_json_keys_only(self):
        (self.root / 'concepts.jsonl').write_text('{"patient_id":"secret-value","shape":"irregular"}\n')
        module.run([self.root], self.out)
        h = (self.out / 'metadata_field_candidates.json').read_text()
        self.assertIn('shape', h)
        self.assertNotIn('secret-value', h)

if __name__ == '__main__': unittest.main()
