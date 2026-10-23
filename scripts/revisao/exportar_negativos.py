"""Export only explicitly reviewed negatives, with empty YOLO label files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_negatives(collection_root, manifest_path, decisions_path, output, split='train'):
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Split invalido')
    root = Path(collection_root).resolve()
    manifest_path = Path(manifest_path)
    decisions_path = Path(decisions_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    review = json.loads(decisions_path.read_text(encoding='utf-8-sig'))
    if review.get('human_review_confirmed') is not True:
        raise ValueError('A revisao humana precisa ser confirmada')
    if review.get('review_id') != manifest.get('review_id'):
        raise ValueError('Decisoes pertencem a outra galeria')
    images = {r['id']:r for r in manifest['images']}
    if len(images) != len(manifest['images']):
        raise ValueError('Manifesto com IDs repetidos')
    decisions = review.get('decisions', [])
    ids = [r['id'] for r in decisions]
    if len(set(ids)) != len(ids) or set(ids) != set(images):
        raise ValueError('IDs desconhecidos, repetidos ou ausentes nas decisoes')
    validated = []
    for decision in decisions:
        if decision.get('status') not in {'pending', 'approved_negative', 'rejected'}:
            raise ValueError('Estado de revisao invalido')
        image = images[decision['id']]
        if decision.get('sha256') != image['sha256']:
            raise ValueError('Hash das decisoes nao confere com o manifesto')
        if decision['status'] != 'approved_negative':
            continue
        source = (root / image['source_image']).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError('Imagem fora da raiz da coleta ou inexistente')
        if sha(source) != image['sha256']:
            raise ValueError(f'Imagem modificada: {image["source_image"]}')
        validated.append((image, decision, source))
    if not validated:
        raise ValueError('Nenhum negativo aprovado; nada sera exportado')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Destino deve ser novo ou vazio')
    image_dir = output / 'images' / split
    label_dir = output / 'labels' / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    exported = []
    duplicate_ids = []
    seen = set()
    for image, decision, source in validated:
        if image['sha256'] in seen:
            duplicate_ids.append(image['id'])
            continue
        seen.add(image['sha256'])
        name = image['id']
        target = image_dir / (name + '.jpg')
        shutil.copy2(source, target)
        if sha(target) != image['sha256']:
            raise OSError('Falha de integridade na copia')
        (label_dir / (name + '.txt')).write_text('', encoding='utf-8')
        exported.append(dict(id=name, source_image=image['source_image'],
                             source_session=image['session'], sha256=image['sha256'],
                             image=target.relative_to(output).as_posix(),
                             label=(label_dir/(name+'.txt')).relative_to(output).as_posix(),
                             review_note=decision.get('note', '')))
    provenance = dict(schema_version=1, review_id=manifest['review_id'],
                      human_review_confirmed=True, reviewed_at_utc=review.get('reviewed_at_utc'),
                      source_manifest_sha256=sha(manifest_path), decisions_sha256=sha(decisions_path),
                      split=split, negatives=len(exported), exact_duplicate_ids_skipped=duplicate_ids,
                      images=exported)
    (output/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'README.md').write_text(
        '# Negativos revisados\n\nPacote complementar, nao um dataset completo de treinamento. '
        'Os rotulos YOLO estao vazios porque o operador confirmou ausencia de fogo e fumaca. '
        'Misturar com exemplos positivos e dados originais. Manter sessoes e ambientes '
        'independentes para validacao/teste, sem sortear quadros vizinhos entre splits.\n',
        encoding='utf-8')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Exporta somente negativos aprovados na galeria')
    p.add_argument('--root', default='dados/coletas')
    p.add_argument('--manifest', required=True)
    p.add_argument('--decisions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    a = p.parse_args()
    result = export_negatives(a.root, a.manifest, a.decisions, a.output, a.split)
    print(json.dumps(dict(negatives=result['negatives'], output=str(Path(a.output).resolve()))))
