# DNA Model (Caduceus/Hyena)

This guide provides instructions for reproducing the genomic sequence modeling experiments as presented in our paper. We provide implementations with Derf (our proposed function), DyT, and LayerNorm. Follow the steps below to set up the environment, train the model, and evaluate the results.

## 1. Clone the Caduceus Repository

Clone the official Caduceus repository from GitHub:

```bash
git clone https://github.com/kuleshov-group/caduceus.git
```

## 2. Installation

Set up the Python environment with the following commands:
```
cd caduceus
conda env create -f caduceus_env.yml
conda activate caduceus_env
```

## 3. Dataset Preparation

Follow the instructions in the original [Caduceus README](https://github.com/kuleshov-group/caduceus/blob/main/README.md) to download and prepare the necessary datasets for DNA sequence modeling.

## 4. Implement Derf

To reproduce the results using Dynamic erf (Derf), apply the following patch:
```
cp dynamic_erf.py caduceus
cp dynamic_erf.patch caduceus
cd caduceus
git apply dynamic_erf.patch
```

In the patch, we also provide implementations of LayerNorm and DyT (`dynamic_tanh.py`). You can easily switch between them by simply commenting and uncommenting the relevant code.

## 5. Training
To train the DNA models using the human reference genome, run the following commands:

### Caduceus

```bash
cd slurm_scripts
sbatch run_pretrain_caduceus.sh
```

### HyenaDNA

```bash
cd slurm_scripts
sbatch run_pretrain_hyena.sh
```
You may need to edit these scripts to adapt them to your computing environment.

## 6. Evaluation
To evaluate the DNA models on the GenomicBenchmarks dataset, run the following command:
```bash
cd slurm_scripts
bash wrapper_run_genomics.sh
```