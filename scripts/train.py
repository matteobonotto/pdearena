# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
sys.path.append(os.getcwd())

from pdearena import utils
import torch
from pdearena.data.datamodule import PDEDataModule
from pdearena.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: F401
from pdearena.models.pdemodel import PDEModel

os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')

logger = utils.get_logger(__name__)

torch.set_float32_matmul_precision("highest") #"highest", "high"


def setupdir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "tb"), exist_ok=True)
    os.makedirs(os.path.join(path, "ckpts"), exist_ok=True)


def main():
    cli = utils.PDECLI(
        PDEModel,
        datamodule_class=PDEDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    if cli.trainer.default_root_dir is None:
        logger.warning("No default root dir set, using: ")
        cli.trainer.default_root_dir = os.environ.get("PDEARENA_OUTPUT_DIR", "./outputs")
        logger.warning(f"\t {cli.trainer.default_root_dir}")

    setupdir(cli.trainer.default_root_dir)
    logger.info(f"Checkpoints and logs will be saved in {cli.trainer.default_root_dir}")
    logger.info("Starting training...")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # if not cli.trainer.fast_dev_run:
    #     logger.info("Starting testing...")
    #     cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
