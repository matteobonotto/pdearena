## Standard PDE Surrogate Learning
```
python scripts/train.py -c <path/to/config>
```
For example, to run modern U-Net on Navier Stokes dataset on 4 GPUs use:

```
python scripts/train.py -c configs/navierstokes2d.yaml \
    --data.data_dir=/mnt/data/NavierStokes2D_smoke \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.batch_size=8 \
    --data.time_gap=0 --data.time_history=4 --data.time_future=1 \
    --model.name=Unetmod-64 \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```

## Conditioned PDE Surrogate Learning
```
python scripts/cond_train.py -c <path/to/config>
```

For example, to run modern U-Net on Navier Stokes dataset on 4 GPUs use:

```
python scripts/cond_train.py -c configs/cond_navierstokes2d.yaml \
    --data.data_dir=/mnt/data/NavierStokes2D_cond_smoke_v1 \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.batch_size=8 \
    --model.name=Unetmod-64 \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```

