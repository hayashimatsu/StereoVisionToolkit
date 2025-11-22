import os
import csv

def create_folders_from_csv(csv_file_path, target_directory):
    # 確認目標資料夾是否存在，若不存在則創建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 讀取CSV文件
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            folder_name = row[0]  # 假設名字在CSV的第一列
            folder_path = os.path.join(target_directory, folder_name)
            # 創建資料夾
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"資料夾 '{folder_name}' 已創建")
            else:
                print(f"資料夾 '{folder_name}' 已存在")

# 使用範例
csv_file_path = 'folder_name.csv'
target_directory = 'result'
create_folders_from_csv(csv_file_path, target_directory)