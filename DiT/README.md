# Diffusion Transformers (DiTs) with Derf

This guide provides instructions for reproducing the Diffusion Transformer (DiT) experiments as presented in our paper. We provide implementations with Derf (our proposed function), DyT, LayerNorm, and other point-wise functions. Follow the steps below to set up the environment, train the model, and evaluate the results.

## 1. Installation
Set up the Python environment with the following commands:
```
conda create -n DiT python=3.12
conda activate DiT
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 diffusers==0.32.2 accelerate==1.4.0
```

## 2. Training
To train the DiT models on ImageNet-1K, run the following command:
```
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model $MODEL \
    --lr $LEARNING_RATE \
    --epochs 80 \
    --data-path /path/to/imagenet/train \
    --results-dir /path/to/saving_dir \
    --normtype $NORMTYPE
```
- Replace `$MODEL` with one of the following options: `DiT-B/4`, `DiT-L/4`, or `DiT-XL/2`.
- Repace `$LEARNING_RATE` with one of the following options: `1e-4`, `2e-4`, or `4e-4`.
- Replace `$NORMTYPE` to choose which point-wise function or normalization layer to use. Available options include: `derf` (our proposed function), `dyt` or `layernorm` (DyT or LayerNorm as baselines), `isru`, `expsign`, etc. (other point-wise functions).

## 3. Evaluation
The evaluation pipeline consists of two stages: sampling images from the trained model and computing the FID score.

### Sampling
To generate samples from the trained DiT model, run the following commands:
```
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
    --model $MODEL \
    --image-size 256 \
    --cfg-scale 1.0 \
    --ckpt /path/to/ckpt \
    --sample-dir /path/to/saving_dir \
    --normtype $NORMTYPE
```
- Replace `$MODEL` and `$NORMTYPE` with the corresponding values used during training to ensure consistency between training and evaluation.


### FID Calculation
The above sampling process generates a folder of samples along with a `.npz` file. We directly use this `.npz` file with [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID scores.
