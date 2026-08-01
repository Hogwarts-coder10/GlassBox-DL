import numpy as np


def backward(root_tensor):
    """
    Executes backpropagation by traversing the computational graph
    in reverse topological order.

    Args:
        root_tensor (Tensor): The final tensor in the graph (usually the scalar loss).
    """

    if not root_tensor.requires_grad:
        raise RuntimeError(
            "Cannot call backward() on a tensor where requires_grad = 'False'"
        )

    # Initialzing the gradient of root tensor
    # If it's a scalar loss, the derivative of a function with respect to itself is 1
    if root_tensor.grad is None or np.all(root_tensor.grad == 0):
        root_tensor.grad = np.ones_like(root_tensor.data, dtype=float)

    # Building topological order of the graph using DFS (Depth-First Search)
    topo_order = []
    visited = set()

    def build_topo(node):
        if node not in visited:
            visited.add(node)

            # Traverse all parent nodes (operands that created this node)
            for child in node._prev:
                build_topo(child)

            # Append the node only after all it's dependencies have been processed
            topo_order.append(node)

    build_topo(root_tensor)

    # Traversing the graph in reverse topological order
    # This gurantees that a node's gradient are fully computed before passing them
    for node in reversed(topo_order):
        node._backward()
