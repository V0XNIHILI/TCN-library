from typing import List, Union

import networkx as nx

def get_receptive_field_size(kernel_size: int,
                             num_layers: int,
                             dilation_exponential_base: int = 2):
    """Calculate the receptive field size of a TCN. We assume the TCN structure of the paper
    from Bai et al.

    Due to: https://github.com/locuslab/TCN/issues/44#issuecomment-677949937

    Args:
        kernel_size (int): Size of the kernel.
        num_layers (int): Number of layers in the TCN.
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        int: Receptive field size.
    """

    return sum([
        2 * dilation_exponential_base**(l - 1) * (kernel_size - 1)
        for l in range(num_layers, 0, -1)
    ]) + 1

def get_kernel_size_and_layers(required_receptive_field_size: int, kernel_sizes: List[int] = [3,5,7,9], dilation_exponential_base: int = 2):
    """Get the configuration of kernel size and number of layers that has a receptive field size closest
    (but always larger) than the required receptive field size.

    Args:
        required_receptive_field_size (int): Required receptive field size.
        kernel_sizes (List[int], optional): List of kernel sizes to choose from. Defaults to [3,5,7,9].
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        Tuple[int, int]: Tuple of kernel size and number of layers.
    """

    configurations = []

    # Find for each kernel size the number of layers that has a receptive field size closest to the required receptive field size
    for kernel_size in kernel_sizes:
        num_layers = 1

        while get_receptive_field_size(kernel_size, num_layers, dilation_exponential_base) < required_receptive_field_size:
            num_layers += 1

        configurations.append((kernel_size, num_layers))

    # Find the configuration with the smallest receptive field size that is larger than the required receptive field size
    min_receptive_field_size = float('inf')
    configuration = None

    for kernel_size, num_layers in configurations:
        receptive_field_size = get_receptive_field_size(kernel_size, num_layers, dilation_exponential_base)

        if receptive_field_size < min_receptive_field_size:
            min_receptive_field_size = receptive_field_size

            configuration = (kernel_size, num_layers)

    return configuration
