# ResNet-Recreation
An implementation of the CIFAR-10 part of ResNet in PyTorch. For more Information on this network and training parameters, please refer to the original paper [here](https://arxiv.org/pdf/1512.03385).

# Usage
Training residual and non-residual networks for all n = 3, 5, 7 and 9:

`python train.py`

Training a list of specific networks:

`python train.py 3,True 8,False` will train a residual network for n=3 and a non-residual network for n=8.

# Results

![Graphs](https://github.com/paulblum00/ResNet-Recreation/blob/main/graphs/graph.png)

The code successfully replicates the main results: Increasing depth beyond n=3 will result in an increase in error rate for the non-residual network and a decrease in error rate for the residual network. Compare with Figure 6 in the original paper.

# Pre-trained models

Pre-trained models are available in the models folder. The naming scheme is analogous to the usage above.
