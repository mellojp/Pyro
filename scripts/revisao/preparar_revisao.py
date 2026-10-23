"""Build an offline review gallery from collected camera candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_candidates(root):
    root = Path(root).resolve()
    records = []
    sessions = []
    for path in sorted(root.glob('*/*/session.json')):
        session = json.loads(path.read_text(encoding='utf-8-sig'))
        if session.get('status') != 'finished_pending_review':
            raise ValueError(f'Encerre a coleta antes de revisar: {path.parent}')
        rows = [json.loads(line) for line in (path.parent / 'candidates.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != session['counts']['saved_images']:
            raise ValueError(f'Contagem divergente: {path.parent}')
        session_id = path.parent.relative_to(root).as_posix()
        sessions.append(dict(id=session_id, model=session['arguments']['model'], images=len(rows)))
        for row in rows:
            image = (path.parent / row['image']).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError('Imagem fora da coleta ou inexistente')
            if digest(image) != row['sha256']:
                raise ValueError(f'Hash divergente: {image}')
            relative = image.relative_to(root).as_posix()
            identity = hashlib.sha256((relative + row['sha256']).encode()).hexdigest()[:20]
            records.append(dict(id=identity, source_image=relative, sha256=row['sha256'],
                                session=session_id, model=Path(session['arguments']['model']).stem,
                                shape=row['shape'], predictions=row['predictions'],
                                captured_at_utc=row['captured_at_utc']))
    if not records:
        raise ValueError('Nenhum candidato encontrado')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Identificadores duplicados')
    return records, sessions


def prepare(root, output):
    import cv2
    import numpy as np
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use uma pasta nova ou vazia para preservar revisoes anteriores')
    records, sessions = read_candidates(root)
    output.mkdir(parents=True, exist_ok=True)
    signatures = []
    images = []
    for number, row in enumerate(records, 1):
        image = cv2.imread(str(root / row['source_image']))
        if image is None or list(image.shape) != row['shape']:
            raise ValueError(f'Imagem invalida: {row["source_image"]}')
        small = cv2.cvtColor(cv2.resize(image, (9, 8)), cv2.COLOR_BGR2GRAY)
        bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
        dhash = sum(int(v) << i for i, v in enumerate(bits))
        mean = image.mean(axis=(0, 1))
        similar = next((old['id'] for old in signatures
                        if (dhash ^ old['hash']).bit_count() <= 4
                        and float(np.mean(np.abs(mean - old['mean']))) <= 12), None)
        signatures.append(dict(id=row['id'], hash=dhash, mean=mean))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        row.update(number=number, probable_duplicate_of=similar,
                   sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                   dark_fraction=float(np.mean(gray <= 10)),
                   bright_fraction=float(np.mean(gray >= 250)),
                   review_status='pending',
                   image_url=quote(os.path.relpath(root / row['source_image'], output).replace(os.sep, '/'), safe='/'))
        images.append(image)
    review_id = hashlib.sha256(''.join(r['id'] + r['sha256'] for r in records).encode()).hexdigest()
    data = dict(schema_version=1, review_id=review_id, sessions=sessions, images=records,
                review_required=True, training_labels_generated=False,
                possible_similar_images=sum(r['probable_duplicate_of'] is not None for r in records))
    (output / 'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    template = Path(__file__).with_name('galeria.html').read_text(encoding='utf-8-sig')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('__REVIEW_DATA__', payload), encoding='utf-8')
    sheets = output / 'folhas'
    sheets.mkdir()
    for start in range(0, len(records), 16):
        sheet = np.full((4 * 270, 4 * 320, 3), 245, dtype=np.uint8)
        for pos, row in enumerate(records[start:start + 16]):
            image = images[start + pos]
            preview = cv2.resize(image, (320, 240))
            for prediction in row['predictions']:
                x1, y1, x2, y2 = prediction['xyxy']
                sx, sy = 320 / image.shape[1], 240 / image.shape[0]
                color = (0, 0, 255) if prediction['class_id'] == 0 else (0, 200, 255)
                cv2.rectangle(preview, (round(x1*sx), round(y1*sy)),
                              (round(x2*sx), round(y2*sy)), color, 1)
            x, y = (pos % 4) * 320, (pos // 4) * 270
            sheet[y:y+240, x:x+320] = preview
            label = f'#{row["number"]:03d} {row["model"]}'
            if row['probable_duplicate_of']:
                label += ' | similar'
            cv2.putText(sheet, label, (x+4, y+259), cv2.FONT_HERSHEY_SIMPLEX, .45, (20,20,20), 1)
        path = sheets / f'pagina-{start//16+1:02d}.jpg'
        if not cv2.imwrite(str(path), sheet):
            raise OSError(path)
    print(json.dumps(dict(images=len(records), possible_similar=data['possible_similar_images'],
                          index=str(output/'index.html'), review_id=review_id)))
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Galeria local para revisar candidatos negativos')
    p.add_argument('--root', default='dados/coletas')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    prepare(a.root, a.output)
