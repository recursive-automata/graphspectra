import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg.eigen.arpack import eigsh


def _reorder_eigensystem(eigensystem, key=None):
    key = key or (lambda x: x)
    values, vectors = eigensystem
    indices = np.argsort(key(values))
    values = values[indices]
    vectors = vectors[:, indices]
    return values, vectors
    

def calculate_small_eigens(pos_def_matrix, k=1, dense=False):
    """ Calculate the smallest eigenvalues and corresponding eigenvectors of
    a symmetric (or hermitian), positive definite matrix.
    
    Args:
        pos_def_matrix: a numpy 2D array or matrix, assumed to be positive definite
        k: an integer, how many eigenvalues and eigenvectors to calculate. Default is 1
        dense: a boolean, whether to use scipy.linalg.eigh to solve eigensystem.
          If False (default), uses scipy.sparse.linalg.eigen.arpack.eigsh.
      
    Returns: values, vectors. Eigenvalues are in ascending order.
    """
    if dense:
        eigensystem = eigh(pos_def_matrix, eigvals=(0, k - 1))
    else:
        eigensystem = eigsh(pos_def_matrix, k, sigma = 0)
    return _reorder_eigensystem(eigensystem, key=np.abs)
