## 一、项目目录结构

PJ2/
├── task1/                     # 任务1：花卉图像分类
│   ├── data/                  # 分类任务数据集
│   │   ├── jpg/               # 花卉原始图片
│   │   ├── imagelabels.mat    # 图像标签文件
│   │   └── setid.mat          # 数据集划分文件
│   ├── experiment\_results.csv # 模型实验结果统计
│   ├── flower\_classification.py # 分类任务核心代码
│   ├── resnet18\_lr0.001\_bs32\_best.pth        # ResNet18最优权重
│   ├── resnet18\_lr0.0001\_bs32\_best.pth       # ResNet18最优权重
│   ├── resnet18\_lr0.0005\_bs32\_best.pth       # ResNet18最优权重
│   ├── resnet18\_lr1e-05\_bs32\_best.pth        # ResNet18最优权重
│   ├── resnet18\_lr5e-05\_bs32\_best.pth        # ResNet18最优权重
│   ├── resnet18\_scratch\_lr0.001\_bs32\_best.pth # 从零训练ResNet18权重
│   └── swin\_tiny\_lr0.0001\_bs32\_best.pth      # Swin-Tiny最优权重
├── task2/                     # 任务2：场景目标检测与视频多目标跟踪
│   ├── ①/                     # 模块1：YOLO模型微调训练
│   │   ├── datasets/          # 道路车辆数据集
│   │   │   ├── train/         # 训练集
│   │   │   │   ├── images/    # 训练图片
│   │   │   │   └── labels/    # 训练标签
│   │   │   ├── valid/         # 验证集
│   │   │   │   ├── images/    # 验证图片
│   │   │   │   └── labels/    # 验证标签
│   │   │   └── data\_1.yaml    # 数据集配置文件
│   │   ├── runs/              # 训练输出结果
│   │   │   └── detect/
│   │   │       └── train/
│   │   │           ├── weights/               # 模型权重目录
│   │   │           │   ├── best.pt           # 最优模型权重
│   │   │           │   └── last.pt           # 最终轮次权重
│   │   │           ├── args.yaml             # 训练参数配置
│   │   │           ├── BoxF1\_curve.png       # F1分数曲线
│   │   │           ├── BoxP\_curve.png        # 精确率曲线
│   │   │           ├── BoxPR\_curve.png       # PR曲线
│   │   │           ├── BoxR\_curve.png        # 召回率曲线
│   │   │           ├── confusion\_matrix.png  # 混淆矩阵
│   │   │           ├── confusion\_matrix\_normalized.png # 归一化混淆矩阵
│   │   │           ├── labels.jpg            # 标签分布可视化
│   │   │           ├── results.csv           # 训练指标日志
│   │   │           ├── results.png           # 训练指标可视化
│   │   │           ├── train\_batch0.jpg      # 训练批次数据可视化
│   │   │           ├── train\_batch1.jpg
│   │   │           ├── train\_batch2.jpg
│   │   │           ├── val\_batch0\_labels.jpg # 验证集标签可视化
│   │   │           ├── val\_batch0\_pred.jpg   # 验证集预测可视化
│   │   │           ├── val\_batch1\_labels.jpg
│   │   │           ├── val\_batch1\_pred.jpg
│   │   │           ├── val\_batch2\_labels.jpg
│   │   │           └── val\_batch2\_pred.jpg
│   │   ├── fine\_tuning.py     # 模型微调核心代码
│   │   ├── yolo26n.pt         # 自定义训练YOLO权重
│   │   └── yolov8n.pt         # 官方预训练YOLO权重
│   └── ②/                     # 模块2：视频目标追踪与计数
│       ├── video\_data/        # 原始视频素材
│       ├── counting\_log.txt   # 目标越线计数日志
│       ├── output\_counted.mp4 # 计数结果输出视频
│       ├── time\_lapse\_video.mp4 # 原始延时摄影视频
│       └── tracking.py        # 目标追踪与计数脚本
└── task3/                     # 任务3

## 二、task1数据集下载

cd PJ2/task1/data
wget https://www.robots.ox.ac.uk/\~vgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/\~vgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/\~vgg/data/flowers/102/setid.mat
tar -xf 102flowers.tgz

## 三、代码执行

cd PJ2/task1
python flower\_classification.py --data\_root /data

cd PJ2/task2/①
python fine\_tuning.py

cd PJ2/task2/②
python tracking.py


**项目三readme**

1\. 数据 — data.py

数据集：Stanford Background，715 张室外场景，8 类（sky / tree / road / grass / water / building / mountain / foreground）

划分：seed=42 固定 shuffle，80/20 → train 572 / val 143

预处理：240×320；训练侧随机尺度裁剪（0.75\~1.35）+ 水平翻转；验证侧只做确定性 resize；ImageNet 均值方差归一化

标签：负数或越界像素 → ignore\_index=255，不进 loss 也不进指标

类别不平衡：训练集像素直方图 \~7.3M / 6.3M / 9.1M / 2.5M / 3.3M / 8.2M / 1.4M (mountain) / 4.8M — mountain 最稀，所以 Dice 有意义

2\. 模型 — model.py

经典 U-Net，base\_channels=32，通道宽度 32→64→128→256→512：



Encoder：4 次 MaxPool 下采样 + DoubleConv (3×3 Conv → BN → ReLU ×2)

Decoder：ConvTranspose2d 上采样 + skip concat + DoubleConv

输出：1×1 Conv → 8 类 logits

全随机初始化，无 torchvision，无预训练

3\. 损失 — losses.py

CE：nn.CrossEntropyLoss(ignore\_index=255)

Dice：自己实现的多类 soft Dice（softmax → one-hot → 2·∩/(|p|+|t|)，加 smooth=1.0）；ignore 像素通过 mask 置零

CE+Dice：等权相加

4\. 训练 — train.py

AdamW lr=1e-4 wd=1e-4，CosineAnnealingLR

batch=16，AMP=True，num\_workers=4

从 10 epoch best ckpt 续训到 40 epoch（continue\_training.sh，可重置 optimizer/scheduler 防止 cosine 跑到 0）

同时保存 best/last ckpt，按 val mIoU 选 best

5\. 指标 — metrics.py

混淆矩阵算 pixel acc / mean acc / per-class IoU / mIoU（mIoU 为主指标）

另外采样最多 20 万像素跑 one-vs-all mAP（满足作业要求的 accuracy/mAP 曲线）

6\. 关键结果 — task3\_unet\_report.md

Loss	Best Epoch	Val mIoU	Val Pixel Acc	Sampled mAP

CE	34	0.4361	0.6737	0.5995

Dice	33	0.4329	0.6674	0.5127

CE + Dice	28	0.4366	0.6760	0.6234

结论：CE+Dice 在 mIoU、Pixel Acc、mAP 三项均最佳，且收敛更早（28 epoch）。CE 单用平滑但易被大类主导；Dice 单用对小类敏感但波动大、mAP 偏低（因 logits 标定差）。

