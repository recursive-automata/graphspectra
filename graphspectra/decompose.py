import numpy as np


def _reorder_eigen_solution(eigen_soln, key=None):
    key = key or (lambda x: x)
    values = eigen_soln[0]
    indices = np.argsort(key(values))
    values = values[indices]
    vectors = eigen_soln[1]
    vectors = vectors[:, indices]
    return values, vectors
    

def calculate_symmetric_eigensystem(symmetric_matrix):
    """ Calculate eigenvalues and eigenvectors of a symmetric
    matrix. Returns the eigenvalues in ascending order.
    """
    # symmetric => use `eigh`
    eigen_soln = np.linalg.eigh(symmetric_matrix)
    return _reorder_eigen_solution(eigen_soln)


def calculate_eigensystem(matrix):
    """ Calculate eigenvalues and eigenvectors of a matrix.
    Returns the eigenvalues in ascending order of magnitude.
    """
    eigen_soln = np.linalg.eig(matrix)
    return _reorder_eigen_solution(eigen_soln, np.abs)