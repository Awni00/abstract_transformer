This code is largely based on: https://github.com/omihub777/ViT-CIFAR/tree/main. They achieve good performance with small ViT models on CIFAR10 dataset, despite weak spatial inductive biases.

We swap out their implementation of ViT for a Dual Attention Transformer to evaluate the effect of relational inductive biases.