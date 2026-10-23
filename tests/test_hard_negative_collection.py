import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'raspberry'))
from hard_negative_collection import HardNegativeCollector


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.frame = np.full((120, 160, 3), 80, dtype=np.uint8)
        self.boxes = [[10, 10, 80, 80, .7, 0]]

    def collector(self, **kwargs):
        return HardNegativeCollector(self.root, {'model': 'test'},
                                     confirmed_no_fire_smoke=True, **kwargs)

    def test_confirmation_required(self):
        with self.assertRaises(ValueError):
            HardNegativeCollector(self.root, {})
        self.assertFalse(self.root.exists())

    def test_original_frame_and_pending_metadata_without_training_labels(self):
        c = self.collector()
        self.assertTrue(c.observe(self.frame, self.boxes, 12, {}, now=1))
        record = json.loads((self.root / 'candidates.jsonl').read_text().splitlines()[0])
        image = cv2.imread(str(self.root / record['image']))
        self.assertEqual(image.shape, self.frame.shape)
        self.assertLess(int(image.max()) - int(image.min()), 3)
        self.assertEqual(record['review_status'], 'pending')
        self.assertEqual(record['predictions'][0]['class_name'], 'fire')
        self.assertFalse((self.root / 'labels').exists())
        c.close()
        self.assertEqual(json.loads((self.root / 'session.json').read_text())['status'],
                         'finished_pending_review')

    def test_interval_deduplication_and_valid_json_lines(self):
        c = self.collector(interval=2, min_change=3)
        self.assertFalse(c.observe(self.frame, [], 1, {}, now=0))
        self.assertTrue(c.observe(self.frame, self.boxes, 2, {}, now=1))
        changed = np.full_like(self.frame, 120)
        self.assertFalse(c.observe(changed, self.boxes, 3, {}, now=2))
        self.assertFalse(c.observe(self.frame, self.boxes, 4, {}, now=4))
        self.assertTrue(c.observe(changed, self.boxes, 5, {}, now=5))
        lines = (self.root / 'candidates.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(s)['sequence'] for s in lines], [2, 5])
        self.assertEqual(c.skipped_interval, 1)
        self.assertEqual(c.skipped_similar, 1)

    def test_storage_limit_and_no_session_overwrite(self):
        c = self.collector(max_images=1)
        self.assertTrue(c.observe(self.frame, self.boxes, 1, {}, now=1))
        self.assertFalse(c.observe(np.full_like(self.frame, 200), self.boxes, 2, {}, now=9))
        self.assertEqual(len(list((self.root / 'images').glob('*.jpg'))), 1)
        self.assertEqual(c.skipped_limit, 1)
        with self.assertRaises(ValueError):
            self.collector()

    def test_cli_rejects_collection_without_confirmation_before_camera_access(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/raspberry/webcam_usb.py'),
                                 '--collect-dir', str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--confirm-no-fire-smoke', result.stderr)
        self.assertFalse(self.root.exists())


if __name__ == '__main__':
    unittest.main()
