# tools/inspect_dataset.py

import os

import numpy as np
import yaml
from collections import Counter
import matplotlib.pyplot as plt


def inspect_dataset_distribution(data_yaml_path='./data.yaml'):
    """
    读取YOLO数据集配置文件，统计并可视化训练集中各类别的实例数量。
    """
    print("--- 开始进行数据集分布体检 ---")

    # 1. 加载配置文件
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config['names']
        train_dir = data_config['train']
        base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        train_labels_dir = os.path.join(base_dir, train_dir.replace('images', 'labels'))
        print(f"成功加载配置文件: {data_yaml_path}")
        print(f"标签目录: {train_labels_dir}")
    except Exception as e:
        print(f"错误：无法加载或解析 {data_yaml_path}。请确保文件存在且格式正确。错误信息: {e}")
        return

    if not os.path.isdir(train_labels_dir):
        print(f"错误：找不到标签目录 {train_labels_dir}。请检查路径。")
        return

    # 2. 遍历所有标签文件，统计每个类别的出现次数
    class_counts = Counter()
    total_files = 0
    for filename in os.listdir(train_labels_dir):
        if filename.endswith('.txt'):
            total_files += 1
            filepath = os.path.join(train_labels_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                    except (ValueError, IndexError):
                        print(f"警告：文件 {filename} 中发现格式错误的行: {line.strip()}")

    print(f"\n--- 体检报告 ---")
    print(f"总共扫描了 {total_files} 个标签文件。")
    if not class_counts:
        print("错误：没有统计到任何标签实例。请检查标签文件是否为空或格式错误。")
        return

    # 3. 打印统计结果
    print("各类别实例数量统计：")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)  # 使用.get以防某个类别完全没出现
        print(f"- 类别 '{name}' (ID: {i}): {count} 个实例")

    # 4. 可视化结果
    labels = [class_names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[key] for key in sorted(class_counts.keys())]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
    plt.xlabel('类别 (Class)')
    plt.ylabel('实例数量 (Instances)')
    plt.title('训练集类别分布直方图')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # 在柱状图上显示具体数字
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom')  # va: vertical alignment

    report_image_path = '../dataset_distribution_report.png'
    plt.savefig(report_image_path)
    print(f"\n可视化报告已保存至: {report_image_path}")
    plt.show()


if __name__ == '__main__':
    # 假设data.yaml在项目根目录，而此脚本在tools/下
    # 所以我们使用相对路径'../data.yaml'
    inspect_dataset_distribution()