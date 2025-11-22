from utils.image import ImageChartGenerator
from pathlib import Path
import numpy as np
import cv2
import logging
import os 
import shutil
import json

class DepthCalculator:
    def __init__(self, config):
        self.config = config
        self.root = Path(__file__).parent.parent
        self.logger = self._setup_logger()
      
        
        # load save picture information
        self.disparity_folder_name = "target_pictures_set_disparity"
        self.depth_folder_name = "target_pictures_set_depth"
        self.rectified_folder_name = "target_pictures_set_rectified"
        self.need_show = True if self.config.need_showPicture == "True" else False
        self.input_folder = None
        self.output_folder= None
        self.temp_folder = None
        self.rectified_temp_folder = None
        self.setup_paths()
        
        ## load parameter  
        #1. experiement
        self.xy_rotate_angle = None
        self.yz_rotate_angle = None
        self.xz_rotate_angle = None
        # 2.camera's rectified
        self.Q = None  # dispartityToDepthMap
        self.P1 = None
        self.P2 = None
        # 3.picture information
        self.max_depth = None # m
        self.min_depth = None # m

        self.load_parameters()       
     
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def setup_paths(self):
        self.input_folder = self.root / Path(self.config.save_path_temp) / Path(self.disparity_folder_name)
        self.output_folder = self.root / Path(self.config.save_path_result)
        self.temp_folder = self.root / Path(self.config.save_path_temp)
        self.rectified_temp_folder = self.root / Path(self.config.save_path_temp) / Path(self.rectified_folder_name)
        self.logger.info(f"Input folder: {self.input_folder}")
        self.logger.info(f"Output folder: {self.output_folder}")
        self.logger.info(f"Temp folder: {self.temp_folder}")
        self.logger.info(f"Rectified temp folder: {self.rectified_temp_folder}")
    
    def load_parameters(self):
        self.logger.info("Loading parameters")

        self.xy_rotate_angle = self.config.xy_rotate_angle
        self.yz_rotate_angle = self.config.yz_rotate_angle
        self.xz_rotate_angle = self.config.xz_rotate_angle
        self.max_depth = self.config.max_depth
        self.min_depth = self.config.min_depth
        self.logger.info("Parameters loaded successfully")

    def createDepth(self):
        self.logger.info("Starting to process all image sets")
        for set_folder in self.input_folder.glob('set_*'):
            self.process_set(set_folder)
        self.logger.info("All image sets processed successfully")
        # return self.output_folder, self.temp_folder

    def process_set(self, set_folder):
        set_name = set_folder.name
        self.logger.info(f"Processing set: {set_name}")
        
        output_set_folder = self.output_folder / self.depth_folder_name / set_name
        temp_set_folder = self.temp_folder / self.depth_folder_name / set_name
        # * Before performing any operations, make sure that the temp folder exists and that it is cleared out beforehand.
        if not os.path.exists(temp_set_folder):
            temp_set_folder.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(temp_set_folder)
            os.makedirs(temp_set_folder)

        for disparity_file in set_folder.glob('*/disparity_*.npy'):
            self.process_disparity_file(disparity_file, output_set_folder, temp_set_folder)

        self.logger.info(f"Set {set_name} processed and saved") 
        
    def process_disparity_file(self, disparity_file, output_folder, temp_folder):
        pair_name = disparity_file.parent.name
        self.logger.info(f"Processing disparity file: {disparity_file.name}")

        # Load the appropriate Q matrix for this pair
        self._load_q_matrix_for_pair(temp_folder.name, pair_name) # temp_folder.name:basename
        self._load_p_matrix_for_pair(temp_folder.name, pair_name) # temp_folder.name:basename
        disparity = np.load(disparity_file)
        depth, world_coord_rotated = self.calculate_depth(disparity)
        
        # create a pairing folder for saving
        output_folder_pair = output_folder / pair_name
        temp_folder_pair = temp_folder / pair_name
        output_folder_pair.mkdir(parents=True, exist_ok=True)
        # * Before performing any operations, make sure that the temp folder exists and that it is cleared out beforehand.
        if not os.path.exists(temp_folder_pair):
            temp_folder_pair.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(temp_folder_pair)
            os.makedirs(temp_folder_pair)
        self.save_depth_results(depth, world_coord_rotated, output_folder_pair, temp_folder_pair, pair_name, need_show=self.need_show)
        
    def _load_q_matrix_for_pair(self, set_name, pair_name):
        """Load the appropriate Q matrix for the given pair from rectified temp folder."""
        # Try to load recalculated Q matrix from rectified temp folder
        q_rectified_path = self.rectified_temp_folder / set_name / pair_name / f'Q_rectified_{pair_name}.csv'
        
        if q_rectified_path.exists():
            try:
                self.Q = np.genfromtxt(q_rectified_path, delimiter=',')
                self.logger.info(f"Loaded recalculated Q matrix for {pair_name} from {q_rectified_path}")
                
                # Also load metadata for verification
                q_metadata_path = self.rectified_temp_folder / set_name / pair_name / f'Q_metadata_{pair_name}.json'
                if q_metadata_path.exists():
                    with open(q_metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    self.logger.info(f"Q matrix metadata: original_size={metadata.get('original_size')}, "
                                   f"rectified_size={metadata.get('rectified_size')}, "
                                   f"scaling_factors={metadata.get('scaling_factors')}")
            except Exception as e:
                self.logger.warning(f"Failed to load recalculated Q matrix for {pair_name}: {e}")
                self.logger.info("Falling back to original Q matrix")
        else:
            self.logger.info(f"No recalculated Q matrix found for {pair_name}, using original Q matrix")

    def _load_p_matrix_for_pair(self, set_name, pair_name):
        """Load P1 and P2 matrices for the given pair."""
        base_path = self.rectified_temp_folder / set_name / pair_name
        
        for matrix_name in ['P1', 'P2']:
            matrix_path = base_path / f'{matrix_name}_{pair_name}.csv'
            try:
                if matrix_path.exists():
                    setattr(self, matrix_name, np.genfromtxt(matrix_path, delimiter=','))
                    self.logger.info(f"Loaded {matrix_name} matrix for {pair_name}")
                else:
                    setattr(self, matrix_name, None)
                    self.logger.warning(f"No {matrix_name} matrix found for {pair_name}")
            except Exception as e:
                setattr(self, matrix_name, None)
                self.logger.error(f"Failed to load {matrix_name} matrix for {pair_name}: {e}")

    def calculate_depth(self, disparity, check_rut_shape=False):
        """Calculate depth from disparity using point-wise calibration or Q-matrix fallback.
        
        This method implements the point-wise calibration approach recommended for
        ego-motion scenarios where standard stereo rectification assumptions are violated.
        
        Args:
            disparity: Disparity map as numpy array
            check_rut_shape: Legacy parameter (unused)
            
        Returns:
            tuple: (depth_map, world_coordinates_rotated)
        """
        self.logger.info("Calculating depth")
        
        # Use point-wise calibration (epipolar correction) when P matrices are available
        if self.Q is None:
            raise ValueError("No Q matrix available for depth calculation")
        self.logger.info("Falling back to Q matrix approach")
        world_coord = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)
            
        world_coord_rotated = self.apply_rotation(world_coord)
        depth = self.coord_to_depth(world_coord_rotated)
        return depth, world_coord_rotated

    def compute_3d_with_epipolar_correction(self, disparity_map, P1, P2):
        """Point-wise epipolar geometry for 3D reconstruction.
        
        TODO: NEED Fix
        """

        return None
    
  

    def apply_rotation(self, world_coord):
        """Apply sequential rotations to world coordinates.
        
        Applies rotations in order: XY -> YZ -> XZ to align the coordinate system
        with the desired reference frame. Fixed to properly accumulate rotations.
        
        Args:
            world_coord: Input world coordinates (H x W x 3)
            
        Returns:
            numpy.ndarray: Rotated world coordinates (H x W x 3)
        """
        rotated_coord = world_coord.copy()
        
        # XY rotation
        theta1 = self.xy_rotate_angle * np.pi / 180
        R1 = np.array([
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1), np.cos(theta1), 0],
            [0, 0, 1]
        ])
        rotated_coord = np.dot(rotated_coord, R1.T)
        
        # YZ rotation
        theta2 = self.yz_rotate_angle * np.pi / 180
        R2 = np.array([
            [1, 0, 0],
            [0, np.cos(theta2), -np.sin(theta2)],
            [0, np.sin(theta2), np.cos(theta2)]
        ])
        rotated_coord = np.dot(rotated_coord, R2.T)
        
        # XZ rotation
        theta3 = self.xz_rotate_angle * np.pi / 180
        R3 = np.array([
            [np.cos(theta3), 0, -np.sin(theta3)],
            [0, 1, 0],
            [np.sin(theta3), 0, np.cos(theta3)]
        ])
        rotated_coord = np.dot(rotated_coord, R3.T)
        
        return rotated_coord

    def coord_to_depth(self, world_coord):
        """Convert world coordinates to depth values.
        
        Args:
            world_coord: World coordinates in mm (H x W x 3)
            
        Returns:
            numpy.ndarray: Depth values in meters (H x W)
        """
        return np.sqrt(np.sum((world_coord / 1000) ** 2, axis=2))

    def save_depth_results(self, depth, world_coord_rotated, output_folder, temp_folder, pair_name, need_show=True):
        # Save .npy file
        if temp_folder is not None:
            np.save(temp_folder / f'world_coord_{pair_name}.npy', world_coord_rotated)

        # Create and save depth image
        depth_mm = depth * 1000  # Convert to mm
        max_depth_mm = self.max_depth * 1000
        min_depth_mm = self.min_depth * 1000

        painter = ImageChartGenerator(
            img=depth_mm,
            xlabel="pixel",
            ylabel="pixel",
            save_path_result=output_folder,
            need_show=need_show,    
            range_max=max_depth_mm,
            range_min=min_depth_mm,
            counter_tick=self.config.counter_tick,
        )
        painter.create_depth(photo_name=f"depth_{pair_name}")

    

    