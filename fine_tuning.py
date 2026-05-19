from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 开始训练
results = model.train(
    data="/home/shichenqiao/Deep_Learning_and_Spatial_Intelligence/PJ2/task2/datasets/data_1.yaml",
    epochs=100,           # 微调轮数，按需调整
    imgsz=640,            # 输入图像尺寸
    batch=16,             # 批大小，按 GPU 显存调整
    device=0,             # 使用 GPU 0
    pretrained=True,      # 使用预训练权重
    patience=10,          # 早停轮数，若验证损失连续20轮不下降则自动停止[reference:3]
    workers=8             # 数据加载工作线程
)