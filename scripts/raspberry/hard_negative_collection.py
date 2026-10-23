"""Collect unannotated hard-negative candidates; never generate training labels."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


class HardNegativeCollector:
    def __init__(self, directory, session, interval=2.0, max_images=300, min_change=3.0, confirmed_no_fire_smoke=False):
        if not confirmed_no_fire_smoke:
            raise ValueError('Confirme que a cena nao tem fogo nem fumaca')
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError('A pasta de coleta deve estar vazia; use uma pasta por sessao')
        if interval <= 0 or max_images < 1 or min_change < 0:
            raise ValueError('Parametros de coleta invalidos')
        self.images = self.root / 'images'
        self.images.mkdir()
        self.interval = interval
        self.max_images = max_images
        self.min_change = min_change
        self.saved = 0
        self.detected_frames = 0
        self.skipped_interval = 0
        self.skipped_similar = 0
        self.skipped_limit = 0
        self.last_saved_at = None
        self.last_signature = None
        self.session = dict(session)
        self.session.update(
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            human_confirmed_scene_without_fire_or_smoke=True,
            review_required=True,
            training_labels_generated=False,
            image_content='Original camera frame without boxes or HUD',
            interval_seconds=interval,
            max_images=max_images,
            min_mean_pixel_change=min_change,
            status='collecting',
        )
        self._write_session()

    def _write_session(self):
        data = dict(self.session, counts=self.summary())
        temp = self.root / 'session.json.tmp'
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(self.root / 'session.json')

    def observe(self, frame, detections, sequence, timing, now=None):
        if not detections:
            return False
        self.detected_frames += 1
        if self.saved >= self.max_images:
            self.skipped_limit += 1
            return False
        now = time.monotonic() if now is None else now
        if self.last_saved_at is not None and now - self.last_saved_at < self.interval:
            self.skipped_interval += 1
            return False
        signature = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
        change = None
        if self.last_signature is not None:
            change = float(np.mean(np.abs(signature - self.last_signature)))
            if change < self.min_change:
                self.skipped_similar += 1
                return False
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        name = f'{self.saved + 1:05d}_{stamp}.jpg'
        path = self.images / name
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f'Falha ao salvar candidato: {path}')
        predictions = []
        for x1, y1, x2, y2, score, category in detections:
            predictions.append(dict(
                class_id=int(category), class_name=['fire', 'smoke'][int(category)],
                score=float(score), xyxy=[float(v) for v in [x1, y1, x2, y2]],
            ))
        record = dict(
            image='images/' + name, captured_at_utc=datetime.now(timezone.utc).isoformat(),
            sequence=int(sequence), shape=list(frame.shape), review_status='pending',
            predictions=predictions, mean_pixel_change=change,
            processing_ms={k:float(v) for k,v in timing.items()},
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        with (self.root / 'candidates.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.saved += 1
        self.last_saved_at = now
        self.last_signature = signature
        self._write_session()
        return True

    def summary(self):
        return dict(saved_images=self.saved, detected_frames=self.detected_frames,
                    skipped_interval=self.skipped_interval, skipped_similar=self.skipped_similar,
                    skipped_limit=self.skipped_limit)

    def close(self):
        self.session['status'] = 'finished_pending_review'
        self.session['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        self._write_session()
