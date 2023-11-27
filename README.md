# TCN library (`tcn_lib`)

This is a library for Temporal Convolutional Networks (TCNs) in PyTorch. It is based on the TCN as described in the paper [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun. The code is largely based on the [original PyTorch implementation](https://github.com/locuslab/TCN) from the authors.

Compared to the original code base, the following extra features were added:

- Support for batch normalization or/and weight normalization
- Possible to disable residual connections
- Support for ResNeXt blocks
- Support for depthwise separable convolutions
- Performing [Kaiming](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) initialization for ReLU activated networks

## Installation

```bash
pip install git+https://github.com/V0XNIHILI/TCN-library.git

# Or:

git clone git@github.com:V0XNIHILI/TCN-library.git
cd TCN-library
pip install -e .
```

## Usage

```python
from tcn_lib import TCN

# Model for sequential MNIST task
seq_mnist_tcn = TCN(1, 10, [25] * 8, 7)

# Same model, but with batch normalization
seq_mnist_tcn_bn = TCN(1, 10, [25] * 8, 7, batch_norm=True)

# Same model, but with weight normalization
seq_mnist_tcn_wn = TCN(1, 10, [25] * 8, 7, weight_norm=True)

# Same model, but without residual connections
seq_mnist_tcn_no_res = TCN(1, 10, [25] * 8, 7, residual=False)

# MNIST classification model with bottleneck blocks
seq_mnist_tcn_bottle = TCN(1, 10, [(64, 256)] * 8, 7, bottleneck=True)

# Same model, but with ResNeXt blocks
mnist_tcn_resnext = TCN(1, 10, [(64, 256)] * 8, 7, bottleneck=True, groups=32)

# Same model but with depthwise separable convolutions
mnist_tcn_depthwise = TCN(1, 10, [(64, 256)] * 8, 7, bottleneck=True, groups=-1)

# Same model, but with dropout
seq_mnist_tcn_dropout = TCN(1, 10, [25] * 8, 7, dropout=0.1)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
