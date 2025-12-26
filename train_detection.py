# train_detection.py
import subprocess, sys, os

# === 可按需修改的常量 ===
YOLO_DIR   = os.path.join('traffic_models', 'yolov5')    # yolov5 代码目录
DATA_YAML  = os.path.join('data', 'ua-detrac.yaml')      # 数据集描述文件
PRETRAIN   = 'yolov5s.pt'                                # 预训练权重（放在项目根）
# =========================


def train_yolo(epochs=50, batch=16, img=640,
               run_name='gui_run', device= 'cpu'):
    """
    调用 Ultralytics yolov5 官方 train.py 进行训练
    参数:
        epochs  : 训练轮数
        batch   : batch size
        img     : 输入分辨率 (正方形边长)
        run_name: runs/train/<run_name>
        device  : 'cpu' / '0' / '0,1' ...
    """
    python_exe = sys.executable              # 当前解释器
    cmd = [
        python_exe, 'train.py',
        '--data', r"..\..\data\ua-detrac.yaml",
        '--weights', PRETRAIN,
        '--img', str(img),
        '--batch', str(batch),
        '--epochs', str(epochs),
        '--device', str(device),
        '--name', run_name,
        '--exist-ok'                         # 目录已存在也继续
    ]

    print('[YOLO]', ' '.join(cmd))
    # 在 YOLO_DIR 目录里执行
    subprocess.run(cmd, cwd=YOLO_DIR, check=True)


# 允许脚本独立运行：python train_detection.py --epochs 30 --batch 8 --device cpu
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Train YOLOv5 on UA-DETRAC')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch',  type=int, default=16)
    ap.add_argument('--img',    type=int, default=640)
    ap.add_argument('--device', default='cpu',
                    help="'cpu' or CUDA id e.g. 0, 0,1")
    ap.add_argument('--name',   default='cli_run')
    args = ap.parse_args()

    train_yolo(epochs=args.epochs,
               batch=args.batch,
               img=args.img,
               run_name=args.name,
               device=args.device)
