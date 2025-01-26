from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height: int = height // kh
    new_width: int = width // kw

    """
    Operation
    1  2  3  4  5  6
    7  8  9 10 11 12 with kernel(2, 2)

    Result:
    1  2  7  8
    3  4  9 10
    5  6 11 12
    """

    # output_lst = []
    # for n in range(batch):
    #     for m in range(channel):
    #         for i in range(new_height):
    #             for j in range(new_width):
    #                 for u in range(i * kh, (i + 1) * kh):
    #                     for v in range(j * kw, (j + 1) * kw):
    #                         output_lst.append(input[n, m, u, v])
    # shape_lst = (batch, channel, new_height, new_width, kh * kw)
    output = input.permute(0, 1, 3, 2)
    output = output.contiguous()
    output = output.view(batch, channel, width, new_height, kh)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)
    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    output, _, _ = tile(input, kernel)
    output = output.mean(4)
    output = output.contiguous()
    output = output.view(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    return output


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        result = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (input, result) = ctx.saved_values
        return (grad_output * (result == input)), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    t = input.exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    t = input.exp()
    t = t.sum(dim)
    t = t.log()
    return input - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    t, _, _ = tile(input, kernel)
    t = max(t, 4)
    t = t.view(batch, channel, t.shape[2], t.shape[3])
    return t


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    if not ignore:
        rand_tensor = rand(input.shape)
        random_drop = rand_tensor > rate
        return input * random_drop
    else:
        return input
