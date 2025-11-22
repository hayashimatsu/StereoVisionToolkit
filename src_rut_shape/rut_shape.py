# from .rectify import Rectifier
from .rectify_refactored import Rectifier
# from .disparity import DisparityCalculator
from .disparity_refactored import DisparityCalculator
from .depth import DepthCalculator
# from .height import HeightCounterCalculator
from .height_refactored import HeightCalculator
# from .test_T import Test_T
# from .test_R import Test_R
# from .output import RutShapeOutput

class RutShape:
    def __init__(self, config):
        # necessary variable
        # 0.hyper parameter
        self.config = config
        self.rectifier = None
        self.resize_scale = None
        self.disparity_calculator = None
        self.need_R_tuning =  False
        self.need_T_tuning =  False
        if self.need_R_tuning and self.need_T_tuning:
            raise ValueError("Both R and T tuning cannot be True simultaneously.")


    def create_rectify(self):
        # initialization
        self.rectifier = Rectifier(self.config)
        # progress
        # self.rectifier.createRectifiedStereoPhoto()
        self.rectifier.create_rectified_stereo_photos()
        # save the resize scale for the disparity
        self.resize_scale = self.rectifier.resize_scale[0]

    def create_disparity(self):
        # initialization
        self.disparity_calculator = DisparityCalculator(self.config, self.resize_scale)
        # progress
        self.disparity_calculator.create_disparity()
        
    def create_depth(self):
        # initialization
        self.depth_calculator = DepthCalculator(self.config)
        # progress
        self.depth_calculator.createDepth()

    def create_height_counter_map(self):
        # initialization
        # self.height_calculator = HeightCounterCalculator(self.config)
        self.height_calculator = HeightCalculator(self.config)
        # progress
        self.height_calculator.create_height()


