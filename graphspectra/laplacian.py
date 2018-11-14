import numpy as np
import scipy.sparse as ss


def count_out_degrees(adjacency_matrix):
    """Count the out-degree (or the total outgoing weights) for 
    each vertex of a graph.
    
    Args:
        adjacency_matrix: either a numpy matrix or a scipy sparse
            matrix, representing the graph's adjacencies.
      
    Returns: a 1D numpy array
    """
    degrees = adjacency_matrix.sum(axis=1, dtype = np.float)
    if len(degrees.shape) == 2:
        degrees = np.array(degrees)[:, 0]
    return degrees


def _lookup_diag_and_identity_functions(matrix):
    """Look up diag and identity functions for the type of matrix
    given (sparse or dense)."""
    sparse = ss.issparse(matrix)
    diag = ss.diags if sparse else np.diag
    identity = ss.identity if sparse else np.identity
    return diag, identity
    

def normalize_adjacency(adjacency_matrix, normalize='sym'):
    """Normalize the adjacency matrix of a graph.
    
    Args:
        adjacency_matrix: either a numpy 2D array or a scipy sparse
            matrix, representing the graph's adjacencies.
        normalize: One of 'rw' (random walk normalization),
            or 'sym' (symmetric normalization, default).
      
    Returns: either a numpy 2D array or a sparse scipy matrix
    """
    assert(normalize in ['rw', 'sym'])
    diag, _ = _lookup_diag_and_identity_functions(adjacency_matrix)
    degrees = count_out_degrees(adjacency_matrix)
    degrees[degrees == 0] = np.inf  # handle disconnected vertices
    if normalize == 'rw':
        degrees = diag(1 / degrees)
        return degrees @ adjacency_matrix
    if normalize == 'sym':
        degrees = diag(degrees ** -0.5)
        return degrees @ adjacency_matrix @ degrees


def compute_laplacian(adjacency_matrix, normalize=False):
    """Compute the Laplacian matrix of a graph from its adjacency
    matrix.
    
    Args:
        adjacency_matrix: either a numpy 2D array or a scipy sparse
            matrix, representing the graph's adjacencies.
        normalize: how and whether to normalize. Either False (no
            normalization, default) or one of the values supported
            by normalize_adjacency_matrix.
      
    Returns: either a numpy 2D array or a sparse scipy matrix
    """
    diag, identity = _lookup_diag_and_identity_functions(adjacency_matrix)
    if not normalize:
        degrees = count_out_degrees(adjacency_matrix)
        return diag(degrees) - adjacency_matrix
    else:
        n = degrees.shape[0]
        normalized = normalize_adjacency(adjacency_matrix, normalize)
        return identity(n) - normalized
