# Rethinking Communication-Efficient Federated Learning with Biased Gradient Compression

This repository provides the implementation for communication-efficient Federated Learning (FL) using **biased gradient compression** techniques. It is based on the paper **"Rethinking Communication-Efficient Federated Learning with Biased Gradient Compression"**, which proposes the **ACTION** condition (Acute-Angle Condition) to optimize the use of biased gradient compression, ensuring better convergence while maintaining efficient communication.

## Key Contributions

- **ACTION Condition**: This algorithm-agnostic condition ensures that biased gradient compression methods achieve optimal convergence in FL, overcoming the issue of suboptimal convergence seen in conventional biased methods.
- **QSL Compression Techniques**: We extend existing gradient compression methods, such as **Quantization (Q)**, **Sparsification (S)**, and **Low-Rank Approximation (L)**, by incorporating the ACTION condition to enhance convergence and communication efficiency.
- **Theoretical and Empirical Validation**: Rigorous theoretical analysis and extensive experiments across multiple models and datasets confirm the effectiveness of the ACTION condition in both computer vision (CV) and natural language processing (NLP) tasks.

## Requirements

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Overview of the System

### 1. Federated Learning with Biased Gradient Compression

The goal of this work is to reduce the communication cost in Federated Learning by using gradient compression methods that are biased but still ensure convergence to the optimal solution. The key idea is that biased compressors can reduce communication overhead, but they introduce a compression error (bias) that typically prevents convergence. Our proposed **ACTION condition** resolves this by ensuring the compressed gradients form an acute angle with the original gradient, which guarantees that the compression error does not prevent optimal convergence.

### 2. Compression Methods

The code implements the following gradient compression methods:

- **Vanilla Quantization (VQ)**
- **Vanilla Sparsification (VS)**
- **Vanilla Low-Rank Approximation (VLR)**
- **Enhanced Versions (EQ, ES, ELR)**, which satisfy the ACTION condition.
- **Degraded Versions (DQ, DS, DLR)**, which violate the ACTION condition.
- **QuantileQuantization** and **EnhancedQuantileQuantization**
- **THCQuantization** and **EnhancedTHCQuantization**
- **FedTC** and **EnhancedFedTC**

## How to Use

### 1. **Main Entry Point: `main.py`**

The entry point for the federated learning process is the `main.py` script. You can run the script with different parameters for experimentation.

### Arguments:

- **`--benchmark`**: Choose the dataset/model to use. Options include:
  - `'vgg11_cifar10'`, `'vgg16_cifar10'`, `'resnet18_cifar100'`, `'vit_tinyin'`, `'bert_yahoo'`, `'bert_agnews'`, `'bert_emotion'`, `'tibert_trec'`
     Default: `'vgg16_cifar10'`
- **`--num_classes`**: Number of classes in the dataset. Default: `10`
- **`--bs`**: Batch size for training. Default: `256`
- **`--comp`**: Compression method to use. Options:
  - `'baseline'`, `'vs'`, `'cs'`, `'vq'`, `'cq'`, `'vlr'`, `'clr'`, `'vqq'`, `'eqq'`, `'rt'`, `'vtq'`, `'etq'`, `'vtc'`, `'etc'`
     Default: `'vq'`
- **`--num_client`**: Number of clients in the federated learning setup. Default: `10`
- **`--alpha`**: Alpha parameter for data distribution among clients. Default: `5`
- **`--global_epoch`**: Number of global epochs. Default: `100`
- **`--local_epoch`**: Number of local epochs per client. Default: `1`
- **`--device`**: Device for training (e.g., `'cuda:0'` or `'cpu'`). Default: `'cuda:0'`
- **`--output`**: Prefix for result files (loss, accuracy). Default: `None`

### Example Command:

```bash
python main.py --benchmark vgg16_cifar10 --num_classes 10 --bs 256 --comp vq --num_client 10 --alpha 5 --global_epoch 100 --local_epoch 1 --output vgg16_cifar10_vq
```

### 2. **Results**:

The results (loss and accuracy) will be saved in the `results/` directory with the following files:

- `*_loss.npy`: Loss values during training
- `*_acc1.npy`: Top-1 accuracy values
- `*_acc5.npy`: Top-5 accuracy values