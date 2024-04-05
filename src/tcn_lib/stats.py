from typing import List, Union, Tuple


def get_receptive_field_size(kernel_size: Union[int, List[int]],
                             num_layers: int,
                             dilation_exponential_base: int = 2):
    """Calculate the receptive field size of a TCN. We assume the TCN structure of the paper
    from Bai et al.

    Due to: https://github.com/locuslab/TCN/issues/44#issuecomment-677949937

    Args:
        kernel_size (Union[int, List[int]]): Size of the kernel(s).
        num_layers (int): Number of layers in the TCN.
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        int: Receptive field size.
    """

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_layers

    return sum([
        2 * dilation_exponential_base**(l - 1) * (kernel_size[l-1] - 1)
        for l in range(1,num_layers+1)
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


def create_graph(kernel_size: int, num_layers: int, input_size: Union[int, None] = None, colors: Tuple[str, str] = None, keep_ancestors_only: bool = False):
    """Create computation graph of a TCN network. Assumes the structure of the TCN as proposed by Bai et al.

    Args:
        kernel_size (int): Size of the kernel
        num_layers (int): Total number of layers in a TCN
        input_size (Union[int, None], optional): Maximum length of sequence as input into the TCN. Defaults to None.
        colors (Tuple[str, str], optional): Tuple of colors to use for the graph (base color, ancestor color). Defaults to None.
        keep_ancestors_only (bool, optional): Whether to keep only the ancestors of the last node in the last layer. Defaults to False.

    Returns:
        Tuple[nx.DiGraph, Dict[(int, int), (int. int)]: Tuple of NetworkX graph of the TCN and a dictionary of
        the positions of the nodes in the graph
    """

    import networkx as nx

    base_color, ancestor_color = ('blue', 'red') if colors is None else colors

    G = nx.DiGraph()
    color_map = []

    if input_size is None:
        input_size = get_receptive_field_size(kernel_size, num_layers, 2)

    # Add nodes
    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            G.add_node((i, j))

            color_map.append(base_color)

    # Create pos dictionary
    pos = {}

    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            pos[(i, j)] = (i, j)

    for i in range(input_size):
        for j in range(2 * num_layers + 1):
            # Always add edge to previous layer with the same i value
            # Also add edge to previous layer with i value of i - kernel_size * n, where n is the dilation
            # This dilation is calculated by the formula: dilation ** (num_layers - j - 1)

            # Add residual connection
            if j % 2 == 0 and j > 0:
                G.add_edge((i, j - 2), (i, j))

            if j > 0:
                # Add connection to previous layer
                G.add_edge((i, j - 1), (i, j))

                dilation = 2**((j + 1) // 2 - 1)

                for k in range(1, kernel_size):
                    if i - k * dilation >= 0:
                        G.add_edge((i - k * dilation, j - 1), (i, j))

    # Find all ancestors of the last node in the last layer
    ancestors = nx.ancestors(
        G, (input_size - 1, num_layers * 2)) | {(input_size - 1, num_layers * 2)}

    if keep_ancestors_only:
        # Remove all nodes that are not ancestors
        for i in range(input_size):
            for j in range(2 * num_layers + 1):
                if (i, j) not in ancestors:
                    G.remove_node((i, j))

        color_map = [ancestor_color] * len(ancestors)
    else:
        # Color all ancestors
        for node in ancestors:
            i, j = node

            color_map[i * (num_layers * 2 + 1) + j] = ancestor_color                

    return G, pos, color_map


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    k = 3
    total_layers = 3

    G, pos, color_map = create_graph(k, total_layers, keep_ancestors_only=False)

    plt.figure(figsize=(10, 5))
    nx.draw(G, pos, with_labels=False, font_size=7, node_color=color_map, node_size=90)  #, node_size=30)
    plt.show()
