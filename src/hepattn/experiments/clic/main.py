"""Top level training script, powered by the lightning CLI."""

import pathlib


# import sys
# import os

# # Robustly find and add the src directory to sys.path
# current_dir = os.getcwd()
# print(f"Current working directory: {current_dir}")

# # Potential paths to the 'src' directory containing 'hepattn'
# possible_src_paths = [
#     os.path.abspath(os.path.join(current_dir, 'src')),              # If cwd is project root
#     os.path.abspath(os.path.join(current_dir, 'hepattn/src')),      # If cwd is workspace root
#     os.path.abspath(os.path.join(current_dir, '../../../../')),     # If cwd is notebook dir
# ]

# found = False
# for path in possible_src_paths:
#     if os.path.isdir(os.path.join(path, 'hepattn')):
#         if path not in sys.path:
#             sys.path.append(path)
#             print(f"Added {path} to sys.path")
#         found = True
#         break

# if not found:
#     # Fallback: Walk up the directory tree to find 'src/hepattn'
#     d = current_dir
#     while len(d) > 1:
#         check_path = os.path.join(d, 'src')
#         if os.path.isdir(os.path.join(check_path, 'hepattn')):
#             if check_path not in sys.path:
#                 sys.path.append(check_path)
#                 print(f"Added {check_path} to sys.path (found via walk)")
#             found = True
#             break
#         d = os.path.dirname(d)

# if not found:
#     print("Warning: Could not find 'hepattn' package directory. Imports may fail.")
# else:
#     print("sys.path setup complete.")

from lightning.pytorch.cli import ArgsType

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.experiments.clic.pflow_data import PflowDataModule
from hepattn.utils.cli import CLI

config_dir = pathlib.Path(__file__).parent / "configs"

import os

def main(args: ArgsType = None) -> None:
    CLI(
        model_class=MPflow,
        datamodule_class=PflowDataModule,
        args=args,
        parser_kwargs={"default_env": True, "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}},
    )


if __name__ == "__main__":
    main()
