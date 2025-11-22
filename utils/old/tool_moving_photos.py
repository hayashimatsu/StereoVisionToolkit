import os
import shutil
from pathlib import Path

def reorganize_folders(source_root):
    source_root = Path(source_root)
    dest_root = source_root.parent / f"{source_root.name}_"
    
    # 遍歷所有 set_* 資料夾
    for set_folder in source_root.glob('set_*'):
        set_number = set_folder.name.split('_')[1]
        
        # 遍歷該 set 資料夾中的所有 jpg 文件
        for img_file in set_folder.glob('*.jpg'):
            # 獲取照片編號
            img_number = img_file.stem.split('_')[1]
            
            # 創建新的目標資料夾
            new_folder = dest_root / f"set_{set_number}" / img_number
            new_folder.mkdir(parents=True, exist_ok=True)
            
            # 移動文件
            shutil.move(str(img_file), str(new_folder / img_file.name))
            
            print(f"Moved {img_file} to {new_folder / img_file.name}")

    print("All files have been moved successfully.")

# 使用示例
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    source_folder = script_dir / "data/picture_set_0620"  # 請根據實際情況修改這個路徑
    reorganize_folders(source_folder)