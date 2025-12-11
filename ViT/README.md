# Vision Transformers (ViTs)

This guide provides instructions for reproducing the Vision Transformer (ViT) experiments as presented in our paper. We provide implementations with Derf (our proposed function), DyT, LayerNorm, and other point-wise functions. Follow the steps below to set up the environment, train the model, and evaluate the results.

## 1. Installation
Set up the Python environment with the following commands:
```
conda create -n ViT python=3.12
conda activate ViT
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 tensorboard
```

## 2. Training & Evaluation
To train and evaluate the ViT models on ImageNet-1K, run the following commands:

### ViT-Base
```
torchrun --nnodes=4 --nproc_per_node=8 main.py \
    --model vit_base_patch16_224 \
    --batch_size 128 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /path/to/imagenet \
    --output_dir /path/to/saving_dir \
    --normtype $NORMTYPE
```


### ViT-Large
```
torchrun --nnodes=8 --nproc_per_node=8 main.py \
    --model vit_large_patch16_224 \
    --drop_path 0.5 \
    --batch_size 64 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --opt_betas 0.9 0.95 \
    --data_path /path/to/imagenet \
    --output_dir /path/to/saving_dir \
    --normtype $NORMTYPE
```
- Here the `effective_batch_size` is 4096, calculated as: `num_nodes` × `num_gpus_per_node` × `batch_size` × `update_freq`. When training with a different number of GPUs, adjust `--batch_size` and `--update_freq` accordingly to maintain the effective batch size of 4096.
- Replace `$NORMTYPE` to choose which point-wise function or normalization layer to use. Available options include: `derf` (our proposed function), `dyt` or `layernorm` (DyT or LayerNorm as baselines), `isru`, `expsign`, etc. (other point-wise functions).


