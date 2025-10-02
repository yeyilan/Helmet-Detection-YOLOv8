# tools/balance_dataset_v2.py

import os
import random
import shutil
from collections import Counter

# -------------------------------------------------------------------
# V2.0 手术方案：在这里定义你的“手术”参数
# -------------------------------------------------------------------
# 1. 我们的“少数派”类别ID
MINORITY_CLASS_IDS = [0, 1, 3]  # 'person', 'helmet', 'no-helmet'
#    我们希望它们过采样后，各自包含它们的“文件数量”能达到这个目标
OVERSAMPLE_FILE_TARGET = 4000

# 2. 我们的“多数派”类别ID
MAJORITY_CLASS_ID = 2  # 'vest'
#    我们希望在所有操作结束后，包含它的“文件数量”上限不超过这个值
UNDERSAMPLE_FILE_CEILING = 5000

# 3. 数据集路径
BASE_DATA_DIR = './datasets'  # 确保这个路径正确


# -------------------------------------------------------------------


def get_class_files_map(labels_dir):
    """扫描标签目录，返回一个字典，key是类别ID，value是包含该ID的文件名列表"""
    class_files = {}
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, 'r') as f:
                present_classes = set(int(line.strip().split()[0]) for line in f if line.strip())
                for cid in present_classes:
                    if cid not in class_files:
                        class_files[cid] = set()
                    class_files[cid].add(filename)
    # 将set转换为list，方便后续操作
    for cid in class_files:
        class_files[cid] = list(class_files[cid])
    return class_files


def balance_dataset_v2():
    """V2.0版本的数据集平衡脚本，分两阶段执行"""
    print("--- 警告：V2.0手术刀将直接修改文件，请务必备份！ ---")
    user_input = input("输入 'yes' 继续执行: ")
    if user_input.lower() != 'yes':
        print("操作已取消。")
        return

    for split in ['train', 'valid']:
        print(f"\n--- 正在处理 {split} 集 ---")

        images_dir = os.path.join(BASE_DATA_DIR, split, 'images')
        labels_dir = os.path.join(BASE_DATA_DIR, split, 'labels')

        if not os.path.isdir(labels_dir):
            print(f"找不到目录 {labels_dir}, 已跳过。")
            continue

        # --- 阶段一：只做“加法” - 过采样少数派 ---
        print("\n--- 阶段一：过采样少数派 ---")
        class_files_map = get_class_files_map(labels_dir)
        for cid in MINORITY_CLASS_IDS:
            if cid not in class_files_map: continue

            files = class_files_map[cid]
            current_count = len(files)

            if current_count == 0 or current_count >= OVERSAMPLE_FILE_TARGET:
                print(f"类别 {cid} 文件数 {current_count}, 无需过采样。")
                continue

            num_to_add = OVERSAMPLE_FILE_TARGET - current_count
            files_to_copy = random.choices(files, k=num_to_add)

            print(f"为类别 {cid} 过采样, 增加 {num_to_add} 个文件...")
            for i, filename in enumerate(files_to_copy):
                base_name = os.path.splitext(filename)[0]
                # 使用更独特的后缀，避免与之前的增强冲突
                new_base_name = f"{base_name}_ovs_{cid}_{i}"

                # 找到对应的图片扩展名（可能是.jpg, .png等）
                original_img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    if os.path.exists(os.path.join(images_dir, base_name + ext)):
                        original_img_path = os.path.join(images_dir, base_name + ext)
                        break

                if original_img_path:
                    # 复制图片
                    shutil.copy(original_img_path,
                                os.path.join(images_dir, new_base_name + os.path.splitext(original_img_path)[1]))
                    # 复制标签
                    shutil.copy(os.path.join(labels_dir, filename), os.path.join(labels_dir, new_base_name + '.txt'))

        # --- 阶段二：再做“减法” - 欠采样多数派 ---
        print("\n--- 阶段二：欠采样多数派 ---")
        # 重新扫描一次文件映射，因为它已经被过采样改变了
        class_files_map_after_oversample = get_class_files_map(labels_dir)
        if MAJORITY_CLASS_ID in class_files_map_after_oversample:
            files = class_files_map_after_oversample[MAJORITY_CLASS_ID]
            current_count = len(files)

            if current_count > UNDERSAMPLE_FILE_CEILING:
                num_to_remove = current_count - UNDERSAMPLE_FILE_CEILING
                # 智能筛选：我们优先删除那些“只包含”多数派的“低价值”样本
                minority_files = set()
                for cid in MINORITY_CLASS_IDS:
                    minority_files.update(class_files_map_after_oversample.get(cid, []))

                majority_only_files = list(set(files) - minority_files)

                # 如果纯多数派样本足够删，就先从这里删
                if len(majority_only_files) >= num_to_remove:
                    files_to_remove = random.sample(majority_only_files, k=num_to_remove)
                else:  # 如果不够，就只能忍痛删除一些混合样本了
                    files_to_remove = majority_only_files
                    remaining_to_remove = num_to_remove - len(majority_only_files)
                    mixed_files_to_consider = list(set(files) & minority_files)
                    files_to_remove.extend(random.sample(mixed_files_to_consider, k=remaining_to_remove))

                print(f"为类别 {MAJORITY_CLASS_ID} 欠采样, 移除 {num_to_remove} 个文件...")
                for filename in files_to_remove:
                    base_name = os.path.splitext(filename)[0]
                    # 找到对应的图片并删除
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = os.path.join(images_dir, base_name + ext)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            break
                    # 删除标签
                    label_path = os.path.join(labels_dir, filename)
                    if os.path.exists(label_path):
                        os.remove(label_path)
            else:
                print(f"类别 {MAJORITY_CLASS_ID} 文件数 {current_count}, 无需欠采样。")

    print("\n--- V2.0 数据集平衡手术完成！ ---")


if __name__ == '__main__':
    balance_dataset_v2()