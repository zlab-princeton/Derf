# Speech Model (wav2vec 2.0)

This guide provides instructions for reproducing the wav2vec 2.0 speech recognition experiments as presented in our paper. We provide implementations with Derf (our proposed function), DyT, and LayerNorm. Follow the steps below to set up the environment, train the model, and evaluate the results.


## 1. Clone the fairseq Repository

Clone the official fairseq repository from GitHub:
```
git clone https://github.com/facebookresearch/fairseq.git
```

## 2. Installation

Set up the Python environment with the following commands:
```
conda create -n w2v python=3.10
conda activate w2v
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install soundfile

cd fairseq
pip install --editable ./
```

*(Fairseq does not provide a config for wav2vec 2.0 Large with LibriSpeech. We created our own by following the instructions from the original paper.)*
Copy the configuration file for wav2vec 2.0 Large with LibriSpeech:
```
cp wav2vec2_large_librispeech.yaml ./fairseq/examples/wav2vec/config/pretraining/
```

## 3. Dataset Preparation
Follow the instructions in the original [wav2vec 2.0 README](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) to download and prepare the LibriSpeech datasets.

## 4. Implement Derf

To reproduce the results using Dynamic erf (Derf), apply the following patch:
```
cp dynamic_erf.patch fairseq
cd fairseq
git apply dynamic_erf.patch
```

In the patch, we also provide implementations of LayerNorm and DyT. You can easily switch between them by simply commenting and uncommenting the relevant code.

## 5. Training & Evaluation
To train and evaluate the wav2vec 2.0 models on the LibriSpeech dataset, run the following commands:

### wav2vec 2.0 Base

```
torchrun --nnodes=8 --nproc_per_node=8 fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir ./examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```

### wav2vec 2.0 Large

```
torchrun --nnodes=16 --nproc_per_node=8 fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir ./examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librispeech
```

- For further details about wav2vec 2.0, see the [original repository](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md).