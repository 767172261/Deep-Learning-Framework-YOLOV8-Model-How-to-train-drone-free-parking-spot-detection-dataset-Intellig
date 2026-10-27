无人机空闲停车位检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/37ca21a428764d77a1f92d85352ab47c.png)
无人机空闲停车位检测数据集，共计2种类别，分别为：['空闲', '已停车']，共计7500+图像。
数据集已整理成YOLO格式 ，YOLOv5、YOLOv6、YOLOv8、YOLO11、YOLO12等等YOLO系列通用，数据集已划分为训练集和测试集等

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e7f481f46d6d44bdae6f43011e09d83c.png)
以下是您提供的 **空余停车位检测数据集** 的详细信息整理成表格，便于快速了解数据构成与使用方式：

---

### 📊 空余停车位检测数据集概览表

| 项目 | 内容 |
|------|------|
| **数据集名称** | 空余停车位检测数据集（YOLO格式） |
| **应用场景** | 智慧停车、停车场管理、城市交通监控、自动驾驶泊车辅助 |
| **采集视角** | 固定摄像头俯拍 / 路侧高位摄像头 / 无人机航拍（视具体来源而定） |
| **总图像数量** | **7,500+ 张** 高质量标注图像 |
| **数据划分** | ✅ 已预划分为 `train`（训练集）和 `val/test`（验证/测试集）<br>（典型比例：8:2 或 7:3） |
| **标注格式** | **YOLO 格式（.txt）**<br>每张图像对应一个标签文件，含归一化边界框坐标 |
| **兼容模型** | **通用 YOLO 系列框架**：<br>✅ YOLOv5 / YOLOv6 / YOLOv7 / YOLOv8<br>✅ 所谓“YOLOv11” / “YOLOv12”（实际基于 YOLOv8 架构的自定义版本） |
| **类别数量 (nc)** | **2 类** |
| **类别名称（中文）** | `['空闲', '已停车']` |
| **类别说明** | - **空闲**：未被车辆占用的停车位<br>- **已停车**：已被车辆占用的停车位 |
| **是否可直接训练** | ✅ 是！只需配置 `.yaml` 文件即可启动训练 |
| **推荐用途** | - 停车场空位实时监测系统<br>- 智慧城市停车诱导屏<br>- 移动端/边缘设备部署（如 Jetson） |

---

### 📁 推荐目录结构（YOLO标准）

```bash
ParkingDataset/
├── images/
│   ├── train/     # 训练图像（~6000+）
│   └── val/       # 验证/测试图像（~1500+）
├── labels/
│   ├── train/     # 对应 .txt 标签文件
│   └── val/
└── parking.yaml   # 数据配置文件
```

---

### 🗃️ 示例 `parking.yaml` 配置文件

```yaml
# parking.yaml
train: ./images/train
val: ./images/val

nc: 2
names: ['空闲', '已停车']
```

> 💡 注：部分 YOLO 实现（如 Ultralytics YOLOv8）要求类别名使用英文，若需兼容，可改用：
> ```yaml
> names: ['empty', 'occupied']
> ```
> 并在推理后映射回中文显示。

---

以下是针对您提供的 **空余停车位检测数据集（2类：'空闲', '已停车'）** 的完整 **YOLOv8 训练代码**，可直接用于训练模型。

---

### ✅ 前提条件

- 已安装 `ultralytics`：
  ```bash
  pip install ultralytics
  ```
- 数据集按标准 YOLO 格式组织（见前文目录结构）
- 配置好 `parking.yaml` 文件

---

### 📜 YOLOv8 训练代码（`train_parking.py`）

```python
# train_parking.py
from ultralytics import YOLO
import os

def main():
    # 创建保存目录
    os.makedirs("runs/parking_detection", exist_ok=True)

    # 加载预训练模型（YOLOv8s，也可选 yolov8n / yolov8m / yolov8l）
    model = YOLO('yolov8s.pt')  # 自动从 Ultralytics 官方下载

    # 开始训练
    results = model.train(
        data='parking.yaml',          # 数据配置文件路径
        epochs=100,                   # 训练轮数（可根据验证 loss 调整）
        imgsz=640,                    # 输入图像尺寸（可选 416/512/640/1280）
        batch=32,                     # 批大小（根据 GPU 显存调整）
        name='parking_yolov8s',       # 实验名称
        project='runs/parking_detection',  # 保存路径
        optimizer='AdamW',            # 优化器（可选 SGD / Adam / AdamW）
        lr0=0.01,                     # 初始学习率
        lrf=0.01,                     # 最终学习率（lr0 * lrf）
        patience=20,                  # 早停耐心值（验证 loss 不下降时停止）
        save=True,                    # 保存 checkpoint 和 best 模型
        save_period=10,               # 每 N 个 epoch 保存一次（可选）
        device=0,                     # 使用 GPU 0（多卡用 [0,1]，CPU 用 'cpu'）
        workers=8,                    # 数据加载线程数
        cache=False,                  # 是否缓存图像到内存（加速训练，但耗内存）
        hsv_h=0.015,                  # 色调增强
        hsv_s=0.7,                    # 饱和度增强
        hsv_v=0.4,                    # 亮度增强
        degrees=10.0,                 # 旋转增强（±10°）
        translate=0.1,                # 平移增强
        scale=0.5,                    # 缩放增强
        flipud=0.0,                   # 上下翻转概率
        fliplr=0.5,                   # 左右翻转概率
        mosaic=1.0,                   # Mosaic 数据增强比例
        mixup=0.1,                    # MixUp 增强比例
    )

    print(f"✅ 训练完成！最佳模型保存在：{results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()
```

---

### 📂 项目结构示例（确保路径一致）

```
YourProject/
├── parking.yaml
├── train_parking.py
└── ParkingDataset/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/
```

其中 `parking.yaml` 内容如下：

```yaml
# parking.yaml
train: ./ParkingDataset/images/train
val: ./ParkingDataset/images/val

nc: 2
names: ['空闲', '已停车']
```

> ⚠️ 注意：Ultralytics YOLOv8 在训练时内部使用英文类别名更稳定。  
> 若遇到兼容性问题，建议将 `names` 改为英文，并在推理时映射回中文：

```yaml
names: ['empty', 'occupied']
```

并在推理后做映射：
```python
class_map = {'empty': '空闲', 'occupied': '已停车'}
```

---

### ▶️ 运行训练

```bash
python train_parking.py
```

训练完成后，模型将保存在：
```
runs/parking_detection/parking_yolov8s/weights/best.pt
```

---

### 🔍 后续使用（推理示例）

```python
from ultralytics import YOLO

model = YOLO('runs/parking_detection/parking_yolov8s/weights/best.pt')
results = model('test_image.jpg')

# 可视化结果
results[0].show()
```

---

