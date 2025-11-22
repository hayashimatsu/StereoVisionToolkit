import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import os
import shutil

class MultiPointSelector:
    def __init__(self, image_path, img_side, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(str(image_path))
        self.marked_image = self.image.copy()
        self.points = []
        self.canvas_name = 'Mark Multiple Points'
        self.need_get_out = False
        self.is_left = True if "left" in img_side else False

    def select_points(self):
        self._setup_window()
        while True:
            cv2.setMouseCallback(self.canvas_name, self._on_mouse_click)
            cv2.imshow(self.canvas_name, self.marked_image)
            if cv2.waitKey(20) & 0xFF == 27:  # Esc key
                break
        cv2.destroyAllWindows()
        self._save_marked_image()
        return self.points

    def _setup_window(self):
        img_height, img_width = self.image.shape[:2]
        scale_factor = min(1.0, 1600 / max(img_height, img_width))
        window_width = int(img_width * scale_factor)
        window_height = int(img_height * scale_factor)

        cv2.namedWindow(self.canvas_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.canvas_name, window_width, window_height)
        cv2.imshow(self.canvas_name, self.image)

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._print_coordinate_info(x, y)
            if self._confirm_selection(x, y):
                self._update_points(x, y)
                self._draw_point(x, y)
                if not self._continue_marking():
                    self.need_get_out = True

    def _confirm_selection(self, x, y):
        message = f'Do you want to mark this point ({x}, y)?'
        return messagebox.askyesno('Confirm Selection', message)

    def _update_points(self, x, y):
        self.points.append((x, y))

    def _draw_point(self, x, y):
        if self.is_left:
            cv2.circle(self.marked_image, (x, y), 5, (0, 0, 255), -1)
        else:
            cv2.circle(self.marked_image, (x, y), 5, (0, 255, 0), -1)

    def _continue_marking(self):
        return messagebox.askyesno('Continue', 'Do you want to mark another point?')

    def _print_coordinate_info(self, x, y):
        print(f'\nPixel coordinates: x = {x}, y = {y}')

    def _save_marked_image(self):
        cv2.imwrite(str(self.output_path), self.marked_image)
        print(f"Marked image saved as: {self.output_path}")

def process_images(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    for set_folder_1 in input_path.glob('set_*'):
        set_number = set_folder_1.name.split('_')[1]
        for set_folder_2 in set_folder_1.glob('0*'):
            for img_file in set_folder_2.glob('*.jpg'):
                # 创建输出文件夹
                img_side = img_file.stem.split('_')[0]
                img_number = img_file.stem.split('_')[1]
                
                output_subfolder = output_path / f"set_{set_number}" / img_number
                output_subfolder.mkdir(parents=True, exist_ok=True)

                # 设置输出文件路径
                output_file = output_subfolder / f"{img_file.stem}{img_file.suffix}"

                # 处理图像
                selector = MultiPointSelector(img_file, img_side, output_file)
                marked_points = selector.select_points()
                print(f"Marked points for {img_file.name}:", marked_points)
                output_point_log = output_subfolder / f"{img_file.stem}(mark).csv"
                np.savetxt(output_point_log,  marked_points, fmt='%d', delimiter=',')

if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    input_folder = script_dir / "data" / "video_set_chiba_hq"
    output_folder = script_dir / "data" / "video_set_chiba_hq_mark"
    # input_folder = script_dir / "result_0911" / "parameter_0620_case(2)-0/target_pictures_set_rectified"
    # output_folder = script_dir / "result_0911" / "parameter_0620_case(2)-0/target_pictures_set_rectified_marked"

    process_images(input_folder, output_folder)