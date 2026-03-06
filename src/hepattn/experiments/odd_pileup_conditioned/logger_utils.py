from lightning.pytorch.loggers import CometLogger

class FixedCometLogger(CometLogger):
    def __init__(self, save_dir=None, offline_directory=None, **kwargs):
        # Map save_dir to offline_directory if not provided
        if offline_directory is None and save_dir is not None:
            offline_directory = save_dir
            
        super().__init__(offline_directory=offline_directory, **kwargs)
