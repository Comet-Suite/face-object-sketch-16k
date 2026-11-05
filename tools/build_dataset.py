#!/usr/bin/env python3
"""
Reproducible builder for the Face / Object / Sketch 512x512 dataset.

    python3 tools/build_dataset.py manifest   # crawl source metadata -> work/*.jsonl
    python3 tools/build_dataset.py build      # download, normalise to 512x512, write data/*.jsonl

Sources
  * Wikimedia Commons (MediaWiki API) - faces, objects, traditional drawings
  * Google Quick, Draw! simplified stroke data (CC BY 4.0) - rendered line drawings

Requires: requests, Pillow  (pip install requests pillow)
"""
import argparse
import concurrent.futures as cf
import hashlib
import html
import io
import json
import os
import random
import re
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')
API = 'https://commons.wikimedia.org/w/api.php'
UA = {'User-Agent': 'FaceObjectSketchDataset/1.0 (https://github.com/Amol-coder5643; research dataset)'}
S = requests.Session()
S.headers.update(UA)
SIZE = 512
SHARD = 2000

# --------------------------------------------------------------------------- filters
TAG = re.compile(r'<[^>]+>')
BAD_TITLE = ('nude', 'nudity', 'naked', 'erotic', 'porn', 'genital', 'sexual', 'topless',
             'lingerie', 'fetish', 'bdsm', 'strip club')
NONOBJ = re.compile(r'(map of|world map|chart|graph of|diagram|infographic|scheme of|plan of|'
                    r'flow ?chart|pie chart|bar chart|statistic|table of|timeline|logo|flag of|'
                    r'coat of arms|emblem|seal of|banknote|postage stamp|text page|newspaper|'
                    r'cover of|poster of|title page|manuscript page|certificate|document)', re.I)
FACEBAD = re.compile(r'(diagram|anatomy|anatomical|statue|sculpture|bust|painting by|oil on canvas|'
                     r'engraving|lithograph|etching|woodcut|caricature|cartoon|illustration|'
                     r'drawing|sketch|manga|anime|doll|mannequin|mask|mummy|skull|skeleton|'
                     r'x-ray|wax|3d render|ai generated)', re.I)
SKETCH_HINT = re.compile(r'(sketch|skizze|esquisse|croquis|drawing|dessin|zeichnung|dibujo|'
                         r'disegno|charcoal|chalk|graphite|pencil|ink|wash|studies of|studie|'
                         r'sketchbook|notebook|contour|line art|hand-?drawn)', re.I)
PHOTO_HINT = re.compile(r'(photograph|photo of|photographed|nikon|canon eos|selfie|screenshot|'
                        r'colored pencils|coloured pencils|pencil case|crayon box|set of pencils|'
                        r'paint tubes|art supplies|museum room|gallery wall)', re.I)

FACE_CATS = [('Category:Portrait photographs of women', 'female'),
             ('Category:Portrait photographs of men', 'male'),
             ('Category:Photographs of women', 'female'),
             ('Category:Photographs of men', 'male'),
             ('Category:Quality images of people', None)]
OBJECT_CATS = ['Category:Quality images of objects', 'Category:Still life', 'Category:Chairs',
               'Category:Cameras', 'Category:Tools', 'Category:Musical instruments',
               'Category:Cutlery', 'Category:Hats', 'Category:Shoes', 'Category:Bicycles',
               'Category:Umbrellas', 'Category:Watches', 'Category:Kitchenware',
               'Category:Bottles', 'Category:Toys', 'Category:Vases', 'Category:Books']
SKETCH_CATS = ['Category:Sketches', 'Category:Pencil drawings', 'Category:Ink drawings',
               'Category:Charcoal drawings', 'Category:Drawings', 'Category:Line drawings',
               'Category:Figure drawings', 'Category:Life drawings', 'Category:Sketchbooks',
               'Category:Drawings by artist', 'Category:Studies by artist',
               'Category:Pen and ink drawings', 'Category:Chalk drawings',
               'Category:Drawings of people', 'Category:Drawings of men',
               'Category:Drawings of women', 'Category:Notebooks']
QD_WORDS = ('cat dog house tree car fish flower apple guitar bicycle chair bird book clock cup '
            'shoe hat umbrella star mountain sun airplane hammer scissors elephant face key '
            'pencil ladder table pizza spider candle door knife spoon fork snowman toothbrush '
            'violin whale windmill zebra lion monkey owl penguin rabbit strawberry watermelon '
            'donut cake bus train truck helicopter bridge bed').split()


def clean(txt, limit=200):
    if not txt:
        return ''
    txt = html.unescape(TAG.sub(' ', str(txt)))
    return re.sub(r'\s+', ' ', txt).strip()[:limit]


def api(params, tries=4):
    for i in range(tries):
        try:
            r = S.get(API, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return {}


def norm_title(t):
    t = re.sub(r'\.(jpe?g|png)$', '', t.replace('File:', ''), flags=re.I).lower()
    return re.sub(r'[^a-z0-9]+', ' ', t).strip()


def keep(title, info, cls):
    if info.get('mime') not in ('image/jpeg', 'image/png'):
        return False
    w, h = info.get('width', 0), info.get('height', 0)
    if w < 700 or h < 700 or not (0.62 <= w / h <= 1.62):
        return False
    low = norm_title(title)
    if any(b in low for b in BAD_TITLE):
        return False
    if cls == 'face' and FACEBAD.search(low):
        return False
    if cls in ('object', 'sketch') and NONOBJ.search(low):
        return False
    return True


def crawl(cat, cls, gender, cap, depth, seen, sub_seen, out):
    queue = [(cat, 0)]
    while queue and len(out) < cap:
        title, d = queue.pop(0)
        cont = {}
        while len(out) < cap:
            p = dict(action='query', generator='categorymembers', gcmtitle=title,
                     gcmtype='file|subcat', gcmlimit=500, prop='imageinfo',
                     iiprop='url|extmetadata|size|mime|sha1', iiurlwidth=512, format='json')
            p.update(cont)
            data = api(p)
            for pg in data.get('query', {}).get('pages', {}).values():
                if pg.get('ns') == 14:
                    if pg['title'] not in sub_seen and d < depth:
                        sub_seen.add(pg['title'])
                        queue.append((pg['title'], d + 1))
                    continue
                if 'imageinfo' not in pg:
                    continue
                info = pg['imageinfo'][0]
                if not keep(pg['title'], info, cls):
                    continue
                sha = info.get('sha1') or norm_title(pg['title'])
                if sha in seen:
                    continue
                seen.add(sha)
                em = info.get('extmetadata', {})
                g = lambda k, n=200: clean(em.get(k, {}).get('value', ''), n)
                out.append(dict(
                    cls=cls, gender=gender if cls == 'face' else None,
                    title=pg['title'], name=clean(pg['title'].replace('File:', ''), 140),
                    caption=g('ImageDescription'),
                    page_url=info.get('descriptionurl', ''), url_512=info.get('thumburl', ''),
                    orig_url=info.get('url', ''), orig_w=info.get('width', 0),
                    orig_h=info.get('height', 0), sha1=sha,
                    license=g('LicenseShortName', 60) or 'see file page',
                    artist=g('Artist', 160) or 'Wikimedia Commons contributor'))
            if 'continue' not in data:
                break
            cont = data['continue']
    return out


def cmd_manifest(args):
    os.makedirs(WORK, exist_ok=True)
    jobs = []
    jobs += [(c, 'face', g, args.faces // len(FACE_CATS) + 200) for c, g in FACE_CATS]
    jobs += [(c, 'object', None, args.objects // len(OBJECT_CATS) + 150) for c in OBJECT_CATS]
    jobs += [(c, 'sketch', None, args.sketches // len(SKETCH_CATS) + 200) for c in SKETCH_CATS]
    seen, sub_seen = set(), set()

    def run(j):
        return crawl(j[0], j[1], j[2], j[3], 2, seen, sub_seen, [])

    t0 = time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        results = list(ex.map(run, jobs))
    rows = [r for rs in results for r in rs]
    rows = [r for r in rows
            if not (r['cls'] == 'sketch' and (PHOTO_HINT.search(r['title']) or
                                              not SKETCH_HINT.search(r['title'] + ' ' + r['caption'])))]
    with open(f'{WORK}/manifest.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    counts = {}
    for r in rows:
        counts[r['cls']] = counts.get(r['cls'], 0) + 1
    print(f'manifest: {counts} in {round(time.time() - t0)}s -> work/manifest.jsonl')


# --------------------------------------------------------------------------- quickdraw
def qd_sample(word, n, max_bytes=6_000_000, seed=0):
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{word}.ndjson'
    req = requests.Request('GET', url, headers=dict(UA, Range=f'bytes=0-{max_bytes}')).prepare()
    raw = S.send(req, timeout=180).content
    lines = raw.split(b'\n')[:-1]
    random.Random(seed).shuffle(lines)
    out = []
    for ln in lines:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        strokes = d.get('drawing') or []
        if d.get('recognized') is False or len(strokes) < 2:
            continue
        out.append(d)
        if len(out) >= n:
            break
    return out


def qd_render(d, seed=0):
    rnd = random.Random(seed)
    strokes = d['drawing']
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    sc = (SIZE * 0.80) / max(w, h)
    ox, oy = (SIZE - w * sc) / 2 - x0 * sc, (SIZE - h * sc) / 2 - y0 * sc
    img = Image.new('RGB', (SIZE, SIZE), 'white')
    dr = ImageDraw.Draw(img)
    width = rnd.choice([4, 4, 5, 5, 6])
    tint = rnd.choice([(20, 20, 24), (30, 30, 34), (45, 40, 35), (18, 22, 30)])
    for sxs, sys_ in zip([s[0] for s in strokes], [s[1] for s in strokes]):
        pts = [(x * sc + ox, y * sc + oy) for x, y in zip(sxs, sys_)]
        if len(pts) > 1:
            dr.line(pts, fill=tint, width=width, joint='curve')
        for x, y in (pts[0], pts[-1]):
            dr.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=tint)
    return img


# --------------------------------------------------------------------------- build
def sketchy(img):
    """Pixel gate: line art on paper => plenty of white, low saturation."""
    px = list(img.resize((128, 128)).getdata())
    white = sum(1 for r, g, b in px if r > 226 and g > 226 and b > 226) / len(px)
    sat = sum(max(r, g, b) - min(r, g, b) for r, g, b in px) / len(px)
    return white >= 0.10 and sat <= 62


def fetch(url, tries=3):
    for a in range(tries):
        try:
            r = S.get(url, timeout=45)
            if r.status_code == 200 and len(r.content) > 2000:
                return r.content
        except Exception:
            pass
        time.sleep(1 + a)
    raise RuntimeError(f'download failed: {url}')


def square_512(content):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert('RGB')
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = int((h - side) * 0.42)   # bias upward: keep heads and object tops in frame
    img = img.crop((left, top, left + side, top + side))
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=88, optimize=True, progressive=True)
    return buf.getvalue()


def caption_of(r):
    c = clean(r.get('caption') or r.get('name') or '')
    if len(c) < 4:
        c = clean(re.sub(r'[_-]+', ' ', r.get('name', '')))
    if len(c) > 150:
        c = c[:147].rsplit(' ', 1)[0] + '...'
    return c


def cmd_build(args):
    rows = [json.loads(l) for l in open(f'{WORK}/manifest.jsonl')]
    rnd = random.Random(42)
    faces = [r for r in rows if r['cls'] == 'face']
    fem = [r for r in faces if r['gender'] == 'female']
    mal = [r for r in faces if r['gender'] == 'male']
    rnd.shuffle(fem)
    rnd.shuffle(mal)
    half = args.faces // 2
    plan = [('face', 'female', r) for r in fem[:half]] + [('face', 'male', r) for r in mal[:half]]
    objs = [r for r in rows if r['cls'] == 'object']
    rnd.shuffle(objs)
    plan += [('object', None, r) for r in objs[:args.objects]]
    # sketches need the pixel gate, so they are downloaded and filtered one by one
    sk = [r for r in rows if r['cls'] == 'sketch']
    rnd.shuffle(sk)

    recs = {'face': [], 'object': [], 'sketch': []}
    queue = list(plan)
    random.Random(7).shuffle(queue)
    t0 = time.time()

    def work(job):
        cls, gender, r = job
        data = square_512(fetch(r['url_512'] or r['orig_url']))
        if cls == 'sketch':
            img = Image.open(io.BytesIO(data))
            if not sketchy(img):
                return job, None
        return job, data

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for job, data in ex.map(work, queue):
            if data is None:
                continue
            cls, gender, r = job
            recs[cls].append((gender, r, data))
            done = sum(len(v) for v in recs.values())
            if done % 500 == 0:
                print(f'  {done} images  {done / (time.time() - t0):.1f}/s', flush=True)

    # quickdraw fills whatever the sketch budget still needs
    need_qd = max(0, args.sketches - len(recs['sketch']))
    per = need_qd // len(QD_WORDS) + 1
    for w in QD_WORDS:
        if need_qd <= 0:
            break
        try:
            for i, d in enumerate(qd_sample(w, per)):
                if need_qd <= 0:
                    break
                img = qd_render(d, seed=i)
                buf = io.BytesIO()
                img.save(buf, 'JPEG', quality=90, optimize=True)
                recs['sketch'].append(('qd', dict(word=w, key_id=d.get('key_id', '')), buf.getvalue()))
                need_qd -= 1
        except Exception as e:
            print('  quickdraw skip', w, e)

    os.makedirs(f'{ROOT}/data', exist_ok=True)
    all_recs = []
    for cls, items in recs.items():
        items = items[:{'face': args.faces, 'object': args.objects, 'sketch': args.sketches}[cls]]
        out = []
        for i, (gender, r, data) in enumerate(items):
            rel = f'images/{cls}-{i // SHARD:02d}/{cls}_{i:05d}.jpg'
            path = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as fh:
                fh.write(data)
            cap = caption_of(r) if r.get('name') else r['word']
            rec = dict(id=f'{cls}_{i:05d}', image=rel, label=cls,
                       gender=gender if cls == 'face' else None, caption=cap,
                       width=SIZE, height=SIZE, bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest())
            if cls == 'sketch' and 'word' in r:
                rec.update(
                    prompt=f'a simple black line drawing of a {cap}, white background',
                    target=f'a simple line drawing of a {cap}',
                    sketch_style='quickdraw line drawing',
                    source='Google Quick, Draw! (simplified dataset)',
                    source_page='https://github.com/googlecreativelab/quickdraw-dataset',
                    original_url=f'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{cap}.ndjson',
                    license='CC BY 4.0',
                    attribution=f'Google Quick, Draw! contributor ({r["key_id"]})',
                    original_size=[256, 256])
            else:
                kind = ('a photograph of the face of a ' + (gender or 'person')
                        if cls == 'face' else 'an everyday object')
                rec.update(
                    prompt=(f'a photograph of the face of a {gender}, head and shoulders, natural lighting'
                            if cls == 'face' else f'a photograph of a single everyday object: {cap}'
                            if cls == 'object' else f'a hand-drawn pencil sketch of {cap}'),
                    target=(f'a photograph of the face of a {gender}' if cls == 'face'
                            else f'a photograph of an everyday object: {cap}' if cls == 'object'
                            else f'a hand-drawn sketch: {cap}'),
                    source='Wikimedia Commons', source_page=r['page_url'],
                    original_url=r['orig_url'], original_size=[r['orig_w'], r['orig_h']],
                    license=r['license'], attribution=r['artist'],
                    **({'sketch_style': 'traditional drawing'} if cls == 'sketch' else {}))
            out.append(rec)
        with open(f'{ROOT}/data/{cls}s.jsonl', 'w') as fh:
            for rec in out:
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        all_recs += out
        print(f'{cls}: {len(out)}', flush=True)
    with open(f'{ROOT}/data/dataset.jsonl', 'w') as fh:
        for rec in all_recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'total {len(all_recs)} in {round(time.time() - t0)}s')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['manifest', 'build'])
    ap.add_argument('--faces', type=int, default=4000)
    ap.add_argument('--objects', type=int, default=4000)
    ap.add_argument('--sketches', type=int, default=8000)
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    {'manifest': cmd_manifest, 'build': cmd_build}[args.command](args)


if __name__ == '__main__':
    main()
