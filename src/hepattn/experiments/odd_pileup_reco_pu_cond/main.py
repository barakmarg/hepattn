"""
ODD Pileup Removal + Reconstruction Training Script.

Top level training script powered by the Lightning CLI.

Usage:
    python main.py fit --config configs/base.yaml
    python main.py test --config configs/base.yaml --ckpt_path path/to/checkpoint.ckpt
"""

import pathlib

from lightning.pytorch.cli import ArgsType

from hepattn.experiments.odd_pileup_reco_pu_cond.lightning_module import ODDPFlowTwoStream
from hepattn.experiments.odd_pileup_reco_pu_cond.pflow_data import ODDDataModule
from hepattn.utils.cli import CLI


config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ODDPFlowTwoStream,
        datamodule_class=ODDDataModule,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}
        },
    )


if __name__ == "__main__":
    main()
