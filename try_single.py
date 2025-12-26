import cv2, torch, matplotlib.pyplot as plt

model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path=r'E:/Deep learning and computer vision assignments/traffic_models/yolov5/runs/train/Finished/best_1.pt'
)
img = cv2.cvtColor(cv2.imread('probe.jpg'), cv2.COLOR_BGR2RGB)
res = model(img, size=640)
print(res)                # 看检测数量
res.render()              # 在 res.ims[0] 上画框
plt.imshow(res.ims[0]); plt.axis(False); plt.show()
