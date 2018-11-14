import numpy as np

from scipy.sparse import issparse
from scipy.linalg import eig, eigh
from scipy.sparse.linalg.eigen.arpack import eigs, eigsh


def reorder_eigensystem(eigensystem, key=None):
    key = key or (lambda x: x)
    values, vectors = eigensystem
    indices = np.argsort(key(values))
    values = values[indices]
    vectors = vectors[:, indices]
    return values, vectors
    

def calculate_small_eigens(hermitian_matrix, k=1):
    """Calculate the smallest eigenvalues and corresponding eigenvectors of
    a symmetric (or hermitian) matrix.
    
    Args:
        hermitian_matrix: a numpy 2D array or scipy sparse matrix, assumed to be Hermitian.
        k: an integer, how many eigenvalues and eigenvectors to calculate. Default is 1
      
    Returns: values, vectors. Eigenvalues are in ascending order.
    """
    sparse = issparse(hermitian_matrix)
    if sparse:
        eigensystem = eigsh(hermitian_matrix, k, sigma = 0)
    else:
        eigensystem = eigh(hermitian_matrix, eigvals=(0, k - 1))
    return reorder_eigensystem(eigensystem, key=np.abs)


def calculate_large_eigens(matrix, k=1):
    """Calculate the largest-magnitude eigenvalues and corresponding eigenvectors of
    a matrix.
    
    Args:
        matrix: a numpy 2D array or scipy sparse matrix.
        k: an integer, how many eigenvalues and eigenvectors to calculate. Default is 1
      
    Returns: values, vectors. Eigenvalues are in descending order of magnitude.
    """
    sparse = issparse(matrix)
    key = lambda v: -np.abs(v)  # opposite of magnitude
    if sparse:
        eigensystem = eigs(matrix, k=k, which='LM')
        eigensystem = reorder_eigensystem(eigensystem, key=key)
        return eigensystem
    else:
        eigensystem = eig(matrix)
        values, vectors = reorder_eigensystem(eigensystem, key=key)
        return values[0:k], vectors[:, 0:k]
