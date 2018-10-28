import numpy as np
import scipy.linalg as la


def _reorder_eigensystem(eigensystem, key=None):
    key = key or (lambda x: x)
    values, vectors = eigensystem
    indices = np.argsort(key(values))
    values = values[indices]
    vectors = vectors[:, indices]
    return values, vectors
    

def calculate_symmetric_eigensystem(symmetric_matrix, k=None):
    """ Calculate the smallest eigenvalues and eigenvectors of
    a symmetric matrix.
    
    Args:
      symmetric_matrix: a symmetric numpy matrix
      k: an integer, how many eigenvalues and
        eigenvectors to calculate.
      
    Returns: values, vectors. Eigenvalues are in ascending order.
    """
    # symmetric => use `eigh`
    k = k or symmetric_matrix.shape[0]
    k -= 1
    eigensystem = la.eigh(symmetric_matrix, eigvals=(0, k))
    return _reorder_eigensystem(eigensystem, np.abs)
