import os


def rename_files_in_directory(directory):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 过滤出MP3文件
    mp3_files = [f for f in files if f.lower().endswith('.mp3')]

    # 确保文件按某种顺序排列（例如按文件名）
    mp3_files.sort()

    # 重命名文件
    for idx, filename in enumerate(mp3_files, start=1):
        old_file_path = os.path.join(directory, filename)
        new_filename = f"train_{idx}.mp3"
        new_file_path = os.path.join(directory, new_filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{old_file_path}' to '{new_file_path}'")


def rename_all_files_in_subdirectories(root_directory):
    for class_name in os.listdir(root_directory):
        class_dir = os.path.join(root_directory, class_name)

        if os.path.isdir(class_dir):
            rename_files_in_directory(class_dir)


# 示例：重命名训练集文件夹中的所有MP3文件
train_directory_path = 'static/dataset/train'
rename_all_files_in_subdirectories(train_directory_path)

# 示例：重命名测试集文件夹中的所有MP3文件
test_directory_path = 'static/dataset/test'
rename_all_files_in_subdirectories(test_directory_path)
