"""
ODD Particle Flow Training Script.

Top level training script powered by the Lightning CLI.

Usage:
    python main.py fit --config configs/base.yaml
    python main.py test --config configs/base.yaml --ckpt_path path/to/checkpoint.ckpt
"""

import pathlib

from lightning.pytorch.cli import ArgsType

from hepattn.experiments.odd_pileup_conditioned.lightning_module import ODDPFlow
from hepattn.experiments.odd_pileup_conditioned.pflow_data import ODDDataModule
from hepattn.utils.cli import CLI


# Path to config directory (same directory as this script)
config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    """
    Main entry point for training/testing.

    Args:
        args: Command line arguments (None to use sys.argv)
    """
    CLI(
        model_class=ODDPFlow,
        datamodule_class=ODDDataModule,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}
        },
    )


if __name__ == "__main__":
    main()
