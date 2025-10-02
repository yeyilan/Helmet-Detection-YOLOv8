# --- 步骤一：解决OMP Error的“魔法代码” ---
# 必须放在所有其他import之前，作为脚本的第一行可执行代码
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 步骤二：导入YOLOv8的核心功能 ---
from ultralytics import YOLO

# --- 步骤三：定义并执行你的训练任务 ---
# 这是一个好习惯，把主程序逻辑放在 if __name__ == '__main__': 下
# 这样可以防止在被其他脚本导入时，代码被意外执行
if __name__ == '__main__':
    # 1. 加载一个模型。
    #    'yolov8s.pt'代表加载YOLOv8的小号(small)预训练模型。
    #    程序会自动下载这个文件（如果本地没有的话）。
    model = YOLO('yolov8m.pt')

    # 2. 调用模型的.train()方法，开始训练。
    #    这里，我们把你之前在命令行里的所有参数，都作为函数的参数传进去。
    results = model.train(

        data='data.yaml',  # 指定你的数据集配置文件
        epochs=100,  # 指定训练的总轮次
        imgsz=640,  # 指定训练的图片尺寸
        # batch=16,          # (可选) 你还可以指定批次大小
        # workers=4          # (可选) 指定数据加载的工作线程数
        patience=20,
        degrees=10.0,  # 随机旋转图片的角度范围 (+/- 10度)
        translate=0.1,  # 随机平移图片的幅度 (+/- 10%)
        scale=0.5,  # 随机缩放图片的幅度 (+/- 50%)
        shear=10.0,  # 随机错切图片的角度 (+/- 10度)
        perspective=0.001,  # 随机透视变换的程度
        flipud=0.5,  # 随机上下翻转的概率 (50%)
        fliplr=0.5,
        optimizer='AdamW',  # 换用AdamW优化器，通常效果更好
        lr0=0.01,  # 初始学习率

        # (5) 给这次实验起个名字，方便区分
        project='runs/detect',  # 结果都保存在这个项目文件夹下
        name='experiment_v4_balanced_data',
    )

    # 3. (可选) 打印一条信息，告诉我们训练完成了
    print("训练完成！结果已保存在runs/detect/目录下。")