#!/usr/bin/env python3
"""Pack ./images into transport-friendly tar shards and verify the manifests.

Usage:
    python3 tools/pack_images.py            # images/ -> images/{class}-NN.tar
    python3 tools/pack_images.py --verify   # check every record against disk

Shards are 2,000 images each (~80 MB of JPEG). Use --size to change that.
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def records():
    path = os.path.join(ROOT, 'data', 'dataset.jsonl')
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def pack(size):
    recs = records()
    groups = defaultdict(list)
    for r in recs:
        groups[(r['label'], r['image'].split('/')[1])].append(r)
    os.makedirs(os.path.join(ROOT, 'images'), exist_ok=True)
    made = []
    for (label, shard), rs in sorted(groups.items()):
        tar_path = os.path.join(ROOT, 'images', f'{shard}.tar')
        with tarfile.open(tar_path, 'w') as tf:
            for r in rs:
                tf.add(os.path.join(ROOT, r['image']), arcname=r['image'])
        made.append((tar_path, len(rs), os.path.getsize(tar_path)))
    for p, n, b in made:
        print(f'{os.path.relpath(p, ROOT):32s} {n:5d} images  {b / 1e6:6.1f} MB')
    print(f'{len(made)} shards, {sum(b for _, _, b in made) / 1e6:.1f} MB total')


def verify():
    recs = records()
    labels = Counter(r['label'] for r in recs)
    print('records:', dict(labels), 'total', len(recs))

    tar_index = {}
    img_dir = os.path.join(ROOT, 'images')
    for f in sorted(os.listdir(img_dir)):
        if f.endswith('.tar'):
            with tarfile.open(os.path.join(img_dir, f)) as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        tar_index[m.name] = f

    problems = []
    for r in recs:
        d = os.path.join(ROOT, r['image'])
        if os.path.exists(d):
            with open(d, 'rb') as fh:
                if hashlib.sha256(fh.read()).hexdigest() != r['sha256']:
                    problems.append((r['id'], 'checksum'))
        elif r['image'] not in tar_index:
            problems.append((r['id'], 'missing'))

    print(f'images in shards: {len(tar_index)}')
    print(f'problems: {len(problems)}')
    for p in problems[:10]:
        print('  ', p)
    return 1 if problems else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--size', type=int, default=2000, help='images per shard')
    a = ap.parse_args()
    sys.exit(verify() if a.verify else (pack(a.size) or 0))
