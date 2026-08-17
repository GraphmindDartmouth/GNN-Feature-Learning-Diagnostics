# MNIST Graph Experiments
This repo is for the manuscript ``Diagnosing Graph Neural Networks: Do They Operate in the Feature Learning Regime and How?`` under review at TMLR. For review purposes we provide illustrative code for the MNIST experiments, including transfer learning and feature learning assessment of various graph models, with or without PE.

## Requirements

A working PyTorch + PyTorch Geometric stack that matches your CUDA setup, plus common Python utilities.

Minimum packages:
- `torch`
- `torchvision`
- `torch_geometric`
- `numpy`
- `wandb`
- `pandas`
- `matplotlib`

The scripts assume local datasets and model output directories; replace the `/path/to/...` placeholders in the code with your paths.

<!-- ## Reviewer setup notes

This is an anonymized illustrative code release for review. Full execution requires replacing local path and W&B placeholders before running experiments.

Common placeholders:

| Placeholder | Purpose |
| --- | --- |
| `/path/to/data` | Raw MNIST, GNNBenchmark MNIST, and saved superpixel tensors. |
| `/path/to/data_rwpe` | MNIST graph data with RWPE positional encodings. |
| `/path/to/data_lappe` | MNIST graph data with LapPE positional encodings. |
| `/path/to/model` | Saved model artifacts from W&B sweep runs. |
| `/path/to/eval` | Pickle outputs from `evaluator.py`. |
| `Your-Entity` | W&B entity name used when querying sweep results. | -->

## Repository layout

- `models.py`: Model definitions (MLPs, MPNNs, Graph Transformer variants).
- `utils.py`: Shared utilities (seed setup, KTA computation, graph transform, and train/test loops for image, superpixel, and PE models).
- `trainer.py`: W&B sweep driver for initial weights rescaling experiments across model families to reach best training accuracy.
- `evaluator.py`: Per-scale evaluation loop that tracks accuracy and kernel target alignment (KTA) during training.
- `kta.py`: Post-training KTA change analysis using W&B-selected trained models.
- `transfer.py`: Train on digits 0-4, then probe transfer to digits 5-9 with a linear classifier.
- `data.ipynb`: Data download and processing.

## Usage

All scripts are standalone CLIs. Replace any `/path/to/...` in the code with your local dataset/model paths before running.

### Training sweeps

```bash
python trainer.py mpnn  --model_name GCN  --epochs 500 --layers 3 --device 0
python trainer.py attn  --model_name GPS  --epochs 500 --layers 3 --device 0
python trainer.py pe    --model_name MLP_RWPE --epochs 500 --layers 3 --device 0
python trainer.py image --epochs 500 --device 0
```

### KTA Evaluation

```bash
python evaluator.py --base gcn --device 0
python evaluator.py --base gps --device 0
python evaluator.py --base mlp_image --device 0
```

### Post-training KTA change

```bash
python kta.py --base gcn --device 0
python kta.py --base gps --device 0
python kta.py --base mlp_image --device 0
```

### Transfer/Probing

```bash
python transfer.py --base gcn --device 0
python transfer.py --base gps --device 0
```

Notes:
- First run data.ipynb to load and process data.
- PE runs expect a `pe` field in the dataset (e.g., RWPE/LapPE variants).
- Image runs use torchvision MNIST or pre-saved superpixel tensors, both assumed to be 28x28.
