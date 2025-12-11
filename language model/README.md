# Language Model (GPT-2)

This guide provides instructions for reproducing the GPT-2 language modeling experiments as presented in our paper. We provide implementations with Derf (our proposed function), DyT, and LayerNorm. Follow the steps below to set up the environment, train the model, and evaluate the results.

## 1. Clone the nanoGPT Repository

Clone the nanoGPT repository from GitHub:

```bash
git clone git@github.com:karpathy/nanoGPT.git
```

## 2. Installation

Set up the Python environment with the following commands:
```
conda create -n GPT2 python=3.10
conda activate GPT2
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## 3. Dataset Preparation

Follow the instructions in the original [nanoGPT README](https://github.com/karpathy/nanoGPT/blob/master/README.md) to download and prepare the necessary datasets for trainng GPT2.

## 4. Implement Derf

To reproduce the results using Dynamic erf (Derf), apply the following patch:
```
cp model_derf.patch nanoGPT
cp train_derf.patch nanoGPT
cd nanoGPT
git apply model_derf.patch
git apply train_derf.patch
```

In the patch, we also provide implementations of LayerNorm and DyT.  
You can easily switch between them by setting `norm_type = $NORMTYPE` in `config/train_gpt2.py`.


## 5. Training & Evaluation
To train and evaluate the GPT2 model on OpenWebText, run the following command:

```
srun python train.py \
    config/train_gpt2.py \
    --attn_alpha_init_value $ATTN_ALPHA \
    --ffn_alpha_init_value $FFN_ALPHA \
    --dec_alpha_init_value $DEC_ALPHA
```

- Replace `$ATTN_ALPHA`, `$FFN_ALPHA`, and `$DEC_ALPHA` to assign the desired initialization values for different types of layers.

