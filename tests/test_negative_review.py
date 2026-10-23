import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'raspberry'))
sys.path.insert(0, str(ROOT / 'scripts' / 'revisao'))
from hard_negative_collection import HardNegativeCollector
from preparar_revisao import prepare
from exportar_negativos import export_negatives


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'coletas'
        collector = HardNegativeCollector(self.root/'model'/'session',
            {'arguments': {'model': 'modelos/test.tflite'}},
            interval=1, confirmed_no_fire_smoke=True)
        for i, value in enumerate([80, 140]):
            collector.observe(np.full((120, 160, 3), value, np.uint8),
                              [[10, 10, 80, 80, .8, 0]], i, {}, now=i*2)
        collector.close()
        self.gallery = self.root/'revisao'/'teste'
        self.manifest = prepare(self.root, self.gallery)
        self.review = dict(schema_version=1, review_id=self.manifest['review_id'],
                           human_review_confirmed=True,
                           decisions=[dict(id=x['id'],sha256=x['sha256'],status='pending')
                                      for x in self.manifest['images']])
        self.decision_file = self.base/'decisions.json'
        self.output = self.base/'negativos'

    def export(self):
        self.decision_file.write_text(json.dumps(self.review))
        return export_negatives(self.root, self.gallery/'manifest.json',
                                self.decision_file, self.output)

    def test_gallery_keeps_all_pending_and_preserves_originals(self):
        self.assertEqual(len(self.manifest['images']), 2)
        self.assertTrue(all(x['review_status']=='pending' for x in self.manifest['images']))
        html = (self.gallery/'index.html').read_text()
        self.assertNotIn('__REVIEW_DATA__', html)
        self.assertIn('human_review_confirmed', html)
        for row in self.manifest['images']:
            path = self.root/row['source_image']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['sha256'])
        self.assertFalse(self.output.exists())

    def test_only_approved_exported_and_label_empty(self):
        self.review['decisions'][0]['status'] = 'approved_negative'
        result = self.export()
        self.assertEqual(result['negatives'], 1)
        self.assertEqual(len(list((self.output/'images/train').glob('*.jpg'))), 1)
        labels = list((self.output/'labels/train').glob('*.txt'))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].read_bytes(), b'')
        with self.assertRaises(ValueError):
            self.export()

    def test_no_confirmation_or_no_approval_rejected_before_writing(self):
        self.review['human_review_confirmed'] = False
        with self.assertRaises(ValueError):
            self.export()
        self.review['human_review_confirmed'] = True
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_wrong_review_or_duplicate_ids_rejected(self):
        self.review['decisions'][0]['status'] = 'approved_negative'
        self.review['review_id'] = 'outro-lote'
        with self.assertRaises(ValueError):
            self.export()
        self.review['review_id'] = self.manifest['review_id']
        self.review['decisions'].append(dict(self.review['decisions'][0]))
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_changed_image_rejected_before_export(self):
        self.review['decisions'][0]['status'] = 'approved_negative'
        path = self.root/self.manifest['images'][0]['source_image']
        path.write_bytes(b'alterado')
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_path_outside_collection_rejected(self):
        self.review['decisions'][0]['status'] = 'approved_negative'
        manifest = json.loads((self.gallery/'manifest.json').read_text())
        manifest['images'][0]['source_image'] = '../fora.jpg'
        (self.base/'fora.jpg').write_bytes(b'fora')
        (self.gallery/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
