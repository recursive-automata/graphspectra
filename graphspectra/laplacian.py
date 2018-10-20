import numpy as np

def compute_in_degrees(adjacency_matrix):
    """ TODO
    """
    sum_row = lambda v: v.sum()
    return np.apply_along_axis(sum_row, 0, adjacency_matrix)


def compute_out_degrees(adjacency_matrix):
    """ TODO
    """
    sum_col = lambda v: v.sum()
    return np.apply_along_axis(sum_col, 1, adjacency_matrix)


def compute_undirected_normalized_laplacian(adjacency_matrix):
    """ TODO
    """
    # undirected graph => out-degree = in-degree = degree
    degrees = compute_out_degrees(adjacency_matrix)
    deg = np.diag(degrees)
    lap = deg -adjacency_matrix
    deg_sqrt_inv = np.diag(degrees ** -0.5)
    return deg_sqrt_inv * lap * deg_sqrt_inv


def compute_directed_laplacian(adjacency_matrix):
    """ TODO
    """
    degrees = compute_out_degrees(adjacency_matrix)
    length = degrees.shape[0]
    degrees_tiled = np.tile(np.asmatrix(degrees).transpose(),
                            [1, length])
    nonzero_degrees = degrees_tiled != 0
    egdes = np.divide(adjacency_matrix, degrees_tiled,
                      out=np.zeros_like(adjacency_matrix),
                      where=nonzero_degrees)
    # np.multiply instead of `*` b/c they're matrices
    diag = np.multiply(nonzero_degrees, np.diag(np.ones(length)))
    return diag - egdes
