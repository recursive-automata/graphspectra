import numpy as np
import scipy.linalg as la


def _reorder_eigen_solution(eigen_soln, key=None):
    key = key or (lambda x: x)
    values = eigen_soln[0]
    indices = np.argsort(key(values))
    values = values[indices]
    vectors = eigen_soln[1]
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
    eigen_soln = la.eigh(symmetric_matrix, eigvals=(0, k))
    return _reorder_eigen_solution(eigen_soln, np.abs)
