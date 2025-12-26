# prune_dataset.py   —— 置于项目根（和 datasets 同级）运行
import shutil, pathlib, sys

ROOT = pathlib.Path('datasets/UA-DETRAC')

# ---------- 1. 你要保留的 10 条序列 ----------
KEEP = {
    'MVI_20011',
    'MVI_20012',
    'MVI_20032',
    'MVI_20033',
    'MVI_20034',
    'MVI_20035',
    'MVI_39031',   # ← 做 val
    'MVI_39051',
    'MVI_39211',
    'MVI_39271',
}
VAL_SEQ = 'MVI_39031'          # 想放到 val 的那条
# ---------------------------------------------

# 2. 确保目标子目录存在
for split in ['train', 'val']:
    for kind in ['images', 'labels', 'xml']:
        (ROOT/kind/split).mkdir(parents=True, exist_ok=True)

img_src   = ROOT/'images'/'train'
label_src = ROOT/'labels'/'train'
xml_src   = ROOT/'xml'/'train'

# 3. 删除不属于 KEEP 的所有文件 ---------------
for jpg in img_src.glob('*.jpg'):
    seq = jpg.stem.split('_')[0]           # MVI_xxxxx
    if seq not in KEEP:
        jpg.unlink(missing_ok=True)
        (label_src/f'{jpg.stem}.txt').unlink(missing_ok=True)

for xml in xml_src.glob('*.xml'):
    seq = xml.stem.split('_')[0]
    if seq not in KEEP:
        xml.unlink(missing_ok=True)

# 4. 把 VAL_SEQ 对应的文件搬到 val ----------
def move_set(stem, src_dir, dst_dir, old_ext, new_ext=None):
    src = src_dir/f'{stem}{old_ext}'
    if src.exists():
        shutil.move(src, dst_dir/f'{stem}{new_ext or old_ext}')

for jpg in img_src.glob(f'{VAL_SEQ}_*.jpg'):
    stem = jpg.stem

    # move jpg & label
    move_set(stem, img_src,   ROOT/'images'/'val', '.jpg')
    move_set(stem, label_src, ROOT/'labels'/'val', '.txt')

# 单独把 VAL_SEQ 的 xml 文件搬一个就够
move_set(VAL_SEQ, xml_src, ROOT/'xml'/'val', '.xml')

print('✔ 数据裁剪完成')

# 5. 提醒清理 YOLO 缓存
(cache := ROOT/'labels').glob('*.cache')
for c in (ROOT/'labels').glob('*.cache'):
    c.unlink()
print('✔ 已删除旧 .cache，可重新训练')
