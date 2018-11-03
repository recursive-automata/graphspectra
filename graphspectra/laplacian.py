import numpy as np

def compute_in_degrees(adjacency_matrix):
    """Compute the in-degrees of the vertices in a directed graph,
    the sums of incoming weights in a weighted directed graph,
    or the degrees of vertices in a simple graph.
    
    Args:
        adjacency_matrix: a numpy 2D array or matrix, representing the
            adjacency matrix of the graph.
      
    Returns: a 1D numpy array
    """
    sum_row = lambda v: v.sum()
    return np.apply_along_axis(sum_row, 0, adjacency_matrix)


def compute_out_degrees(adjacency_matrix):
    """Compute the out-degrees of the vertices in a directed graph,
    the sums of outgoing weights in a weighted directed graph,
    or the degrees of vertices in a simple graph.
    
    Args:
        adjacency_matrix: a numpy 2D array or matrix, representing the
            adjacency matrix of the graph.
      
    Returns: a 1D numpy array
    """
    sum_col = lambda v: v.sum()
    return np.apply_along_axis(sum_col, 1, adjacency_matrix)


def compute_laplacian(adjacency_matrix):
    """Compute the Laplacian matrix of a simple graph.
    
    Args:
        adjacency_matrix: a numpy 2D array or matrix, representing the
            adjacency matrix of a simple graph.
      
    Returns: same type and shape as adjacency_matrix, representing the
        Laplacian matrix of the graph.
    """
    degrees = compute_out_degrees(adjacency_matrix)
    deg = np.diag(degrees)
    return deg - adjacency_matrix


def compute_normalized_laplacian(adjacency_matrix):
    """Compute the normalized Laplacian matrix of a simple graph.
    
    Args:
        adjacency_matrix: a numpy 2D array or matrix, representing the
            adjacency matrix of a simple graph.
      
    Returns: same type and shape as adjacency_matrix, representing the
        normalized Laplacian matrix of the graph.
    """
    # undirected graph => out-degree = in-degree = degree
    degrees = compute_out_degrees(adjacency_matrix)
    deg = np.diag(degrees)
    lap = deg - adjacency_matrix
    deg_sqrt_inv = np.diag(degrees ** -0.5)
    return deg_sqrt_inv * lap * deg_sqrt_inv
