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
from tqdm import tqdm
from time import time
from lightning.pytorch.cli import SaveConfigCallback


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
        # save_config_overwrite=True,
        # save_config_callback = SaveConfigCallback(
        #     overwrite=True,
        #     parser='omegaconf',
        #     config=
        #     ),
        run=False,
        # parser_kwargs={"parser_mode": "omegaconf"},
    )
    # cli.config.data.num_workers = 20
    
    cli.config.data.data_dir = os.path.join(
        os.getcwd(),
        cli.config.data.data_dir
    )

    dm = cli.datamodule
    cli.datamodule.setup()

    t1 = time()
    for i, (X,y) in enumerate(tqdm(cli.datamodule.test_dataloader()[0])):
        # a = X.to(torch.device('cuda')).max()
        a = X.max()
    t2 = time() - t1
    print('time per iteration : {:0.5f}s'.format(t2/i))



    '''
    ON: DELL LATITUDE (NVIDIA GEFORCE MX 550)
    batch_size = 8, num_workers = 0

    1625it [04:38,  5.83it/s]
    time per iteration : 0.17152s

    Pikled dataset
    1625it [01:32, 17.55it/s]
    time per iteration : 0.05703s

    hdf5 compressed (blosc_lz4)
    1625it [12:58,  2.09it/s]
    time per iteration : 0.47940s


    batch_size = 8, num_workers = 20
    hdf5 uncompressed (Original dataset)
    1632it [00:51, 31.54it/s]
    time per iteration : 0.03172s

    Pikled dataset
    1632it [00:42, 38.12it/s]
    time per iteration : 0.02625s

    hdf5 compressed (blosc_lz4)
    1632it [02:45,  9.88it/s]
    time per iteration : 0.10125s
    '''

    '''
    ON: MACBOOK PRO (M1 PRO)
    batch_size = 8, num_workers = 20, persistent_workers = True
    
    Pikled dataset
    1625it [00:11, 143.60it/s]
    time per iteration : 0.01007s
    '''



if __name__ == "__main__":
    main()
