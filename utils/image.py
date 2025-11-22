import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.signal_processor import SignalProcessor
from typing import Dict, Any
import cv2
import numpy as np
import tkinter.messagebox as tkMessageBox
import logging
import pandas as pd
from pathlib import Path
import shutil
from typing import List, Tuple

class ImageChartGenerator:
    def __init__(self, img, xlabel: str, ylabel: str, 
                 save_path_result: str, need_show: bool=True, 
                 range_max:int=None , range_min:int=None, counter_tick:int=500):
        # create the frame
        self.fig= None
        self.ax = None
        self.figsize = (25, 11)
        self.dpi = 100
        self.fontname = "MS Gothic"
        self.pad_inches = 0.3
        self.fontsize = 25  # Set the desired font size here
        
        # need the data for image
        self.img = img
        
        # need the introduction for x axis and y axis
        self.xlabel=xlabel
        self.ylabel=ylabel
        
        # need the detail for the range of the counter
        if range_max:
            self.max = range_max 
        if range_min:
            self.min = range_min 
        if counter_tick:
            self.tick = counter_tick
        
        # need save folder and name
        self.save_path_result = save_path_result
        self.need_show = need_show
        self.photo_name = None
        
    def create_rectify(self, photo_name=None, textstr=None, corner_info=None):
        self.photo_name = "rectified" if photo_name is None else photo_name
        self._setup_figure()
        # add comment
        if corner_info is not None:
            corner1_min, corner2_min, corner1_max, corner2_max = corner_info
            # these are matplotlib.patch.Patch properties
            if textstr is not None:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
                self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=8,
                            verticalalignment='top', bbox=props)
            self.ax.scatter(corner1_min[0][0], corner1_min[0][1], facecolors='none', edgecolors='magenta', s=100, marker='o')
            self.ax.scatter(corner2_min[0][0], corner2_min[0][1], facecolors='none', edgecolors='magenta', s=100, marker='o')
            self.ax.scatter(corner1_max[0][0], corner1_max[0][1], facecolors='none', edgecolors='magenta', s=100, marker='^')
            self.ax.scatter(corner2_max[0][0], corner2_max[0][1], facecolors='none', edgecolors='magenta', s=100, marker='^')
        im1 = self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), cmap='gray')
        # output picture
        if self.need_show:
            plt.draw()
            self._close_and_save(self.fig)
        else:   
            self.fig.savefig(f'{self.save_path_result}/{self.photo_name}.jpg', bbox_inches='tight', pad_inches=self.pad_inches)
            plt.close()
    
    def create_disparity(self, target_disparity, photo_name=None):
        self.photo_name = "disparity" if photo_name is None else photo_name
        self._setup_figure()
        # set the base for the counter picture 
        cmap_ = plt.cm.jet_r
        cmap_.set_bad(color="black") 

        im1 = self.ax.imshow(self.img, cmap=cmap_, vmin=self.min, vmax=self.max)
        
        # color bar ------------------------------------------ 
        divider = make_axes_locatable(self.ax)  # axに紐付いたAxesDividerを取得
        cax = divider.append_axes("right", size="5%", pad=self.pad_inches)  # append_axesで新しいaxesを作成
        cbar = self.fig.colorbar(im1, ax=self.ax, cax=cax, ticks=[self.min, target_disparity, self.max])
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=self.fontsize)
        # output picture
        if self.need_show:
            plt.draw()
            self._close_and_save(self.fig)
        else:   
            self.fig.savefig(f'{self.save_path_result}/{self.photo_name}.jpg', bbox_inches='tight', pad_inches=self.pad_inches)
            plt.close()
    
    def create_depth(self, photo_name=None):
        self.photo_name = "result_depth(rotate fix)" if photo_name is None else photo_name
        self._setup_figure()
        # set the base for the counter picture 
        cmap_ = plt.cm.jet_r
        cmap_.set_bad(color="black") 
        im1 = self.ax.imshow(self.img, cmap=cmap_, vmin=self.min, vmax=self.max)
        
        # color bar ------------------------------------------ 
        divider = make_axes_locatable(self.ax)  # axに紐付いたAxesDividerを取得
        cax = divider.append_axes("right", size="5%", pad=self.pad_inches)  # append_axesで新しいaxesを作成
        ticks1 = self._get_tick(self.min, self.max, self.tick)
        cbar = self.fig.colorbar(im1, ax=self.ax, cax=cax, ticks=ticks1)
        cbar.ax.tick_params(labelsize=self.fontsize)
        # output picture
        if self.need_show:
            plt.draw()
            self._close_and_save(self.fig)
        else:   
            self.fig.savefig(f'{self.save_path_result}/{self.photo_name}.jpg', bbox_inches='tight', pad_inches=self.pad_inches)
            plt.close()

    def create_height(self, show_rut_shape:np=None):
        self.photo_name = "result_height(fit with rut)"
        self._setup_figure()
        # set the base for the counter picture 
        cmap_ = plt.cm.jet
        cmap_.set_bad(color="black") 
        im1 = self.ax.imshow(self.img, cmap=cmap_, vmin=self.min, vmax=self.max)
        
        # color bar ------------------------------------------ 
        divider = make_axes_locatable(self.ax)  # axに紐付いたAxesDividerを取得
        cax = divider.append_axes("right", size="5%", pad=self.pad_inches)  # append_axesで新しいaxesを作成
        ticks1 = self._get_tick(self.min, self.max, 10)
        cbar = self.fig.colorbar(im1, ax=self.ax, cax=cax, ticks=ticks1)
        cbar.ax.tick_params(labelsize=self.fontsize)
        # if you need to show the rut
        if show_rut_shape:
            self.ax.scatter(self.interpolateLine_imgCoor[0,:][0], self.interpolateLine_imgCoor[-1,:][1], color='magenta', s=100, marker='x', alpha=0.7)
            self.ax.scatter(self.interpolateLine_imgCoor[-1,:][0], self.interpolateLine_imgCoor[-1,:][1], color='magenta', s=100, marker='x', alpha=0.5)
            self.ax.plot(self.interpolateLine_imgCoor[:,0], self.interpolateLine_imgCoor[:,1], color='magenta', linestyle = "-", linewidth = 3)
        # output picture
        if self.need_show:
            plt.draw()
            self._close_and_save(self.fig)
        else:   
            self.fig.savefig(f'{self.save_path_result}/{self.photo_name}.jpg', bbox_inches='tight', pad_inches=self.pad_inches)
            plt.close()

    def create_matching_point(self, photo_name=None, matchPt=None):
        self.photo_name = "matching_point" if photo_name is None else photo_name
        self._setup_figure()
        # add comment
        textstr = '\n'.join((
                    f"y_interval:max('△') = {matchPt[0],matchPt[1]}",
                    "unit:pixel"))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)
        # self.ax.scatter(matchPt[0], matchPt[1], facecolors='none', edgecolors='magenta', s=150, marker="x", alpha=0.5)
        self.ax.scatter(matchPt[0], matchPt[1], facecolors='magenta', s=50, marker="x")
        self.ax.scatter(matchPt[0], matchPt[1], facecolors='magenta', s=150, marker="x", alpha=0.5)
        # output picture
        im1 = self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.draw()
        self._close_and_save(self.fig)


    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi) 
        self.ax.set_xlabel(self.xlabel, fontsize=self.fontsize, fontname=self.fontname)
        self.ax.set_ylabel(self.ylabel, fontsize=self.fontsize, fontname=self.fontname)
        self.ax.tick_params(axis='both', which='major', labelsize=self.fontsize)
        
    def _close_and_save(self, fig):
        """_表した画像を消して保存する。_

        Args:
            fig (_plt_): _内容を決めたfig_
            name (_str_): _保存したい名前_
        """
        while True:
            if plt.waitforbuttonpress(0):
                plt.close()
                if tkMessageBox.askyesno('askyesno', 'この画像を保存しますか?'):
                    fig.savefig(f'{self.save_path_result}/{self.photo_name}.jpg', bbox_inches='tight', pad_inches=self.pad_inches)
                break
    
    def _get_tick(self, min_val, max_val, step):
        # Adjust min_val and max_val to the closest multiple of step 
        min_val = np.floor(min_val / step) * step
        max_val = np.ceil(max_val / step) * step

        # Create a list of ticks from min_val to max_val with a step size of step
        ticks = np.arange(min_val, max_val + step, step).astype(int)
        return ticks
    
    
    
class ImagePairOrganizer:
    def __init__(self, left_folder: str, right_folder: str, output_folder: str):
        self.left_folder = Path(left_folder)
        self.right_folder = Path(right_folder)
        self.output_folder = Path(output_folder)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def organize_image_pairs(self):
        self.logger.info("Starting image pair organization")
        set_folders = self._get_set_folders()
        for set_folder in set_folders:
            self._process_set(set_folder)
        self.logger.info("Image pair organization completed")
        
    def _get_set_folders(self) -> List[str]:
        left_sets = {folder.name for folder in self.left_folder.glob('set_*') if folder.is_dir()}
        right_sets = {folder.name for folder in self.right_folder.glob('set_*') if folder.is_dir()}
        common_sets = left_sets.intersection(right_sets)
        if not common_sets:
            self.logger.warning("No matching set folders found")
        return sorted(common_sets)

    def _process_set(self, set_name: str):
        self.logger.info(f"Processing set: {set_name}")
        left_images = self._get_images(self.left_folder / set_name, "left_*.jpg")
        right_images = self._get_images(self.right_folder / set_name, "right_*.jpg")
        paired_images = self._pair_images(left_images, right_images)
        self._organize_pairs(paired_images, set_name)
        
    def _get_images(self, folder: Path, pattern: str) -> List[Path]:
        return sorted(folder.glob(pattern))

    def _pair_images(self, left_images: List[Path], right_images: List[Path]) -> List[Tuple[Path, Path]]:
        pairs = []
        for left in left_images:
            right = next((r for r in right_images if r.stem[6:] == left.stem[5:]), None)
            # left.stem → 'left_001', left.stem[5:] → 001
            # right.stem → 'right_001', right.stem[6:] → 001
            if right:
                pairs.append((left, right))
            else:
                self.logger.warning(f"No matching right image for {left.name}")
        return pairs
    
    def _organize_pairs(self, pairs: List[Tuple[Path, Path]], set_name: str):
        for left, right in pairs:
            pair_number = left.stem[5:]
            output_dir = self.output_folder / "target_picture_set" / set_name / pair_number
            output_dir.mkdir(parents=True, exist_ok=True)
            self._copy_image(left, output_dir / left.name)
            self._copy_image(right, output_dir / right.name)
            
    def _copy_image(self, source: Path, destination: Path):
        try:
            shutil.copy2(source, destination)
        except IOError as e:
            self.logger.error(f"Error copying {source} to {destination}: {e}")
            

# visualization.py
class RutProfileVisualizer:
    """Handles visualization of rut profiles"""
    
    def __init__(self, signal_processor=None, reference_data=None):
        self.signal_processor = signal_processor or SignalProcessor()
        self.reference_data = reference_data

    
    def plot_profile(self, profile_data):
        """Plots rut profile with proper formatting"""
        plt.figure(figsize=(10, 6))

        # Apply filter and plot main profile
        x_coords = profile_data[:, 0]
        y_coords = self.signal_processor.apply_filter(profile_data[:, 1])
        # plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Current Profile')
        plt.plot(profile_data[:, 0], profile_data[:, 1], 'b-', linewidth=2, label='Current Profile')
        
        # Plot reference data if available
        if self.reference_data is not None:
            plt.plot(self.reference_data[:, 0], 
                    self.reference_data[:, 1], 
                    'r--', 
                    linewidth=1.5, 
                    label='Reference Profile')
            plt.legend()
        
        
        plt.title('Rut Profile')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (mm)')
        plt.grid(True)
        # Set Y-axis limits 
        plt.ylim(-80, 20)
        plt.yticks(range(-80, 21, 10))
        
        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Wait for any key press to close
        plt.waitforbuttonpress()
        plt.close()



def resize_image(image, scale_factor):
    # Calculate new dimensions
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dimensions = (width, height)

    # Resize the image
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)
    return resized_image
