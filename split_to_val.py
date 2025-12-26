import os, random, shutil, pathlib

# ---------- 你自己的数据根 ----------
root = pathlib.Path("datasets/UA-DETRAC")

img_train = root / "images" / "train"
lbl_train = root / "labels" / "train"
img_val   = root / "images" / "val"
lbl_val   = root / "labels" / "val"

img_val.mkdir(parents=True, exist_ok=True)
lbl_val.mkdir(parents=True, exist_ok=True)

# 抽 10 % 做验证
imgs = [p for p in img_train.glob("*.jpg")]
n_val = max(1, int(len(imgs) * 0.1))          # 至少 1 张
val_subset = random.sample(imgs, n_val)

for jpg in val_subset:
    txt = lbl_train / (jpg.stem + ".txt")      # 同名标注
    if not txt.exists():                       # 没对应 label 就跳过
        continue
    shutil.move(str(jpg), img_val / jpg.name)
    shutil.move(str(txt), lbl_val / txt.name)

print(f"Moved {len(val_subset)} images(+labels) -> val/")
