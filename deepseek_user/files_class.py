"""
This file is used to process the files and create the JSON file for the front-end.
The code will move 20% of the files in each sub-directory to the validation set and the rest to the training set.
The JSON file will store the cleaned folder names and the original folder names.
The code will also move the files to the corresponding folders.
The code will create the necessary folders for the validation and training sets.
The code will be run by running the following command in the terminal:
"""

import json
import os
import random
import shutil


def process_files_and_create_json(path):
    json_data = {}  # 用于存储清理后的名称和原始文件夹名称

    files_list = os.listdir(path)
    for file in files_list:
        sub_path = os.path.join(path, file)
        if os.path.isdir(sub_path):
            # 去除数字和.号后的文字作为 JSON 的键
            folder_name_cleaned = file.split('.')[1] if '.' in file else file
            json_data[file] = folder_name_cleaned

            # 获取子目录里的文件列表并打乱顺序
            file_list = os.listdir(sub_path)
            random.shuffle(file_list)
            var_count = int(len(file_list) * 0.2)  # 计算20%的文件数量
            var_files = file_list[:var_count]  # 前20%的文件作为验证集
            train_files = file_list[var_count:]  # 剩下的80%的文件作为训练集

            var_dir = os.path.join('../static/img/val/', file)
            train_dir = os.path.join('../static/img/train/', file)

            # 创建验证集和训练集的文件夹
            os.makedirs(var_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)

            # 移动文件到验证集和训练集文件夹
            for var_file in var_files:
                shutil.move(os.path.join(sub_path, var_file), os.path.join(var_dir, var_file))
            for train_file in train_files:
                shutil.move(os.path.join(sub_path, train_file), os.path.join(train_dir, train_file))

    # 确保 JSON 文件夹存在
    json_dir = os.path.join('../static/json/')
    os.makedirs(json_dir, exist_ok=True)

    # 生成和写入 JSON 文件
    json_file_path = os.path.join(json_dir, 'name.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    process_files_and_create_json('../static/images')
