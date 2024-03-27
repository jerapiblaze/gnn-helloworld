# Graph Neural Network - Hello world

A graph neural network getting started project using `torch_geometric`.

## Setup

Conda is required.

```bash
conda create -n gnn-helloworld python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 numpy networkx ipykernel matplotlib pandas -c pytorch -c nvidia -c conda-forge -y
conda activate gnn-helloworld
conda install pyg -c pyg -y
conda install pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pytorch -c pyg -y
```