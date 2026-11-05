"""Quick sanity checks on the manifests: counts, balance, image sizes, missing files."""
import json, os, sys, collections
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name='dataset.jsonl'):
    path = os.path.join(ROOT, 'data', name)
    return [json.loads(line) for line in open(path, encoding='utf-8')]


def main():
    recs = load()
    print(f'records        : {len(recs):,}')

    labels = collections.Counter(r['label'] for r in recs)
    for k, v in labels.items():
        print(f'  {k:<7}      : {v:,}')

    faces = [r for r in recs if r['label'] == 'face']
    gender = collections.Counter(r['gender'] for r in faces)
    print('face gender    :', dict(gender))

    style = collections.Counter(r.get('sketch_style') for r in recs if r['label'] == 'sketch')
    print('sketch style   :', dict(style))

    lic = collections.Counter(r['license'] for r in recs)
    print('top licences   :', lic.most_common(5))

    missing, bad = [], []
    for r in recs:
        p = os.path.join(ROOT, r['image'])
        if not os.path.exists(p):
            missing.append(r['id'])
            continue
        with Image.open(p) as im:
            if im.size != (512, 512):
                bad.append((r['id'], im.size))

    print(f'missing images : {len(missing)}')
    print(f'wrong size     : {len(bad)}')
    if missing:
        print('  e.g.', missing[:3], '— run tools/unpack.sh first if you cloned without unpacking')
    if bad:
        print('  e.g.', bad[:3])

    problems = len(missing) + len(bad)
    print('OK' if not problems else 'PROBLEMS FOUND')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
