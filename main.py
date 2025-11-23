from config.config import Config
from src_rut_shape.rut_shape import RutShape


def load_config(config_path: str) -> Config:
    """
    Load configuration from a JSON fil1e.

    Args:1
        config_path (str): Path to the configuration JSON file.

    Returns:
        Config: Loaded configuration object.
    """
    return Config(config_path)


def process_rut_shape(config: Config) -> None:
    """
    Process rut shape analysis using the provided configuration.

    Args:
        config (Config): Configuration object containing processing parameters.
    """
    rut_analyzer = RutShape(config)
    rut_analyzer.create_rectify()
    rut_analyzer.create_disparity()
    rut_analyzer.create_depth()
    rut_analyzer.create_height_counter_map()


def main() -> None:
    """
    Main function to execute the rut shape analysis pipeline.
    """ 
    # Choose the appropriate configuration file
    config_file = "config/config_rut_shape.json"
    # config_file = "con1fig/config_rut_shape_exam.json"
    
    config = load_config(config_file)
    process_rut_shape(config)
    print("Processing completed successfully")


if __name__ == "__main__":
    main() 