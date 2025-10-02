# find_hard_cases.py (V4.0 - 最终修正版)

# ---------------- 准备工作 ----------------
import os
import yaml
import json
from ultralytics import YOLO
# --- 核心修正：我们不再从ultralytics导入工具，而是导入Python自己的工具 ---
import glob  # glob是一个强大的文件查找库

# 解决OMP Error (好习惯)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':

    # ---------------- 1. 加载模型和配置文件 ----------------
    print("--- 正在加载模型和数据配置... ---")

    model_path = 'runs/detect/experiment_v24/weights/best.pt'  # 请确保这是你最好的模型路径
    model = YOLO(model_path)

    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    # 获取训练集图片的根目录。注意：data.yaml中的路径是相对路径
    train_image_dir = data_config['train']
    # 将其转换为绝对路径，以避免混淆
    base_dir = os.path.dirname(os.path.abspath('data.yaml'))
    abs_train_image_dir = os.path.join(base_dir, train_image_dir)

    # ---------------- 2. (新方法) 使用我们自己的函数，查找所有图片文件 ----------------
    print(f"--- 正在从 {abs_train_image_dir} 目录中查找图片... ---")

    # glob.glob会返回一个所有匹配指定模式的文件路径列表
    # f'{...}/**/*' 是一种强大的模式匹配语法：
    # ** 代表“匹配任意层级的子文件夹”
    # *  代表“匹配任意文件名”
    # recursive=True 是必须的，让**能够生效
    image_paths = sorted(glob.glob(os.path.join(abs_train_image_dir, '**/*'), recursive=True))
    # 过滤掉非图片文件
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    # 根据图片路径，推断出对应的标签路径
    label_paths = [p.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt') for p in image_paths]

    if not image_paths:
        print(
            f"错误：在指定的路径 {abs_train_image_dir} 下没有找到任何图片文件！请检查你的data.yaml文件中的'train'路径是否正确。")
        exit()  # 如果一张图都找不到，就直接退出程序

    # ---------------- 3. 开始“嗅探”，找出“嫌疑犯” ----------------
    print(f"\n--- 开始评估训练集中的 {len(image_paths)} 张图片... ---")

    model.eval()
    hard_cases = []

    for i, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
        try:
            # 使用predict模式
            results = model(img_path, verbose=False)

            num_predictions = len(results[0].boxes)

            # 读取对应的标签文件
            with open(label_path, 'r') as f:
                num_true_boxes = len(f.readlines())

            # 计算误差分数
            error_score = abs(num_true_boxes - num_predictions)
            if num_true_boxes > 0 and num_predictions == 0:
                error_score += num_true_boxes * 2

            if error_score > 0:
                hard_cases.append({
                    'image_path': os.path.relpath(img_path, base_dir),  # 记录相对路径，更清晰
                    'true_boxes': num_true_boxes,
                    'pred_boxes': num_predictions,
                    'error_score': error_score
                })
        except Exception as e:
            print(f"\n处理文件 {img_path} 时发生错误: {e}")
            continue  # 如果某张图处理失败，就跳过，继续处理下一张

        if (i + 1) % 200 == 0:
            print(f"已处理 {i + 1}/{len(image_paths)} 张图片...")

    # ---------------- 4. 整理“犯罪档案” ----------------
    print("\n--- 分析完成！整理困难样本报告... ---")

    hard_cases_sorted = sorted(hard_cases, key=lambda x: x['error_score'], reverse=True)

    output_file = 'hard_cases_report.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hard_cases_sorted, f, indent=4, ensure_ascii=False)

    print(f"报告已生成！共找到 {len(hard_cases_sorted)} 个疑似困难/错误样本，已记录在 {output_file} 文件中。")
    print("请打开这个JSON文件，并重点检查排在最前面的那些图片和它们的标签是否正确。")