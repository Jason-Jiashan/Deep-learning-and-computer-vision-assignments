import shutil, pathlib

# 源目录：解压后的 DETRAC-Images/DETRAC-Images
SRC = pathlib.Path(r'E:\DETRAC-Images\DETRAC-Images')
# 目标目录：你的 YOLO images/train
DST = pathlib.Path(r'E:\Deep learning and computer vision assignments\datasets\UA-DETRAC\images\train')
DST.mkdir(parents=True, exist_ok=True)

for seq_dir in SRC.glob('MVI_*'):
    seq = seq_dir.name               # MVI_20011
    for jpg in seq_dir.glob('img*.jpg'):
        raw = jpg.stem[3:]  # '00001'
        idx = f'{int(raw):04d}'  # → '0001'
        new_name = f'{seq}_frame{idx}.jpg'
        shutil.copy(jpg, DST / new_name)

print('✓  图片已全部复制并重命名到', DST)
