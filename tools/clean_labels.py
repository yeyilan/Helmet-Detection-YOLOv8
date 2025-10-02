# tools/clean_labels.py

import os
import yaml


def clean_and_remap_labels():
    """
    遍历指定的数据集标签文件夹，根据定义的映射规则，
    清洗并重写所有的YOLO标签文件。
    """
    print("--- 开始执行标签清洗与重映射任务 ---")

    # --- 步骤一：定义我们的“新旧世界”翻译词典 ---
    # 这个字典，就是我们的核心规则
    # key是旧的、混乱的类别ID，value是我们定义的新的、干净的类别ID
    label_map = {
        # --- 映射到新的 'person' (ID: 0) ---
        11: 0,  # 'Person' -> 0
        25: 0,  # 'worker' -> 0

        # --- 映射到新的 'helmet' (ID: 1) ---
        8: 1,  # 'Helmet' -> 1
        17: 1,  # 'helmet' -> 1

        # --- 映射到新的 'vest' (ID: 2) ---
        14: 2,  # 'Safety-Vest' -> 2
        12: 2,  # 'Reflective-Vest' -> 2
        24: 2,  # 'vest' -> 2
        15: 2,  # 'Vest' -> 2
        16: 2,  # 'Yelek' -> 2
        23: 2,  # 'safety vest - v1...' -> 2

        # --- 映射到新的 'no-helmet' (ID: 3) ---
        9: 3,  # 'No-Helmet' -> 3
        18: 3,  # 'no helmet' -> 3
        20: 3,  # 'no_helmet' -> 3

        # 注意：所有不在这里面的旧ID（比如'0','1','Goggles'等），
        # 都会被自动地、优雅地忽略和抛弃掉。
    }
    print("标签映射规则已定义。")

    # --- 步骤二：定位我们的“脏数据”文件夹 ---
    # 假设我们的数据集文件夹名叫 'datasets'
    # 你可以根据你的真实文件夹名修改
    base_data_dir = './datasets'  # '../' 代表从当前文件(在tools里)返回上一级
    train_labels_dir = os.path.join(base_data_dir, 'train', 'labels')
    valid_labels_dir = os.path.join(base_data_dir, 'valid', 'labels')

    directories_to_clean = [train_labels_dir, valid_labels_dir]

    # --- 步骤三：开始“大清洗” ---
    total_files_processed = 0
    total_boxes_remapped = 0

    for label_dir in directories_to_clean:
        if not os.path.isdir(label_dir):
            print(f"警告：找不到目录 {label_dir}，已跳过。")
            continue

        print(f"\n正在清洗目录: {label_dir}")
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(label_dir, filename)
                new_lines = []

                try:
                    with open(filepath, 'r') as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            if not parts: continue  # 跳过空行

                            old_class_id = int(parts[0])

                            # 核心逻辑：进行“翻译”
                            if old_class_id in label_map:
                                new_class_id = label_map[old_class_id]
                                # 重新拼接成YOLO格式的行
                                new_line = f"{new_class_id} {' '.join(parts[1:])}"
                                new_lines.append(new_line)
                                total_boxes_remapped += 1

                    # 用清洗过的新内容，覆盖掉原来的旧文件
                    with open(filepath, 'w') as f_out:
                        for line in new_lines:
                            f_out.write(line + '\n')

                    total_files_processed += 1
                except Exception as e:
                    print(f"处理文件 {filepath} 时发生错误: {e}")

    print(f"\n--- 清洗任务完成！ ---")
    print(f"总共处理了 {total_files_processed} 个标签文件。")
    print(f"总共重映射了 {total_boxes_remapped} 个标注框。")
    print("你的标签数据现在已经干净、统一了！")


# --- 脚本的入口 ---
if __name__ == '__main__':
    # 运行主函数
    clean_and_remap_labels()