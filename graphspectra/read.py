import numpy as np
from sklearn.metrics.pairwise import distance_metrics, pairwise_kernels

from networkx import Graph, DiGraph
from networkx.convert_matrix import from_numpy_matrix


def read_graph(filepath, directed=True, attr_names=None):
    """ Read a graph from file.
    
    Params:
        filepath -- a string, representing whence to read the graph.
        directed -- a boolean, whether the graph is directed. Defaults to True.
        attr_names -- names of any edge attributes. Assumes there aren't any.
        
    Returns: a networkx Graph or DiGraph
    """
    graph = DiGraph() if directed else Graph()
    with open(filepath, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            split_line = [x.strip() for x in line.split()]
            x, y = split_line[0:2]
            attr_names = attr_names or []
            attr = dict(zip(attr_names, split_line[2:]))
            graph.add_edge(x, y, **attr)
    return graph


def make_disk_graph(X, radius, metric='euclidean'):
    """Make a generalized disk graph, in which points whose distance is less
    than a certain radius are considered adjacent.
    
    Params:
        X: a 2D numpy array of shape (n_observations, n_features).
        radius: the radius of disks for adjacency. 
        metric: string, representing which metric. Options are given by
            sklearn.metrics.pairwise.distance_metrics. Default is 'euclidean'.
        
    Returns: a networkx simple Graph
    """
    metric = distance_metrics()[metric]
    dist = metric(X)
    adj = np.asarray(dist < radius, dtype=np.float)
    return from_numpy_matrix(adj, create_using=Graph)


def make_kernel_graph(X, metric='rbf', cutoff=0, **kwargs):
    """Make a weighted graph, using the a pairwise kernel function
    for weights.
    
    Params:
        X: a 2D numpy array of shape (n_observations, n_features).
        metric: string or function, the metric to use when calculating kernel.
            Options are given by sklearn.metrics.pairwise.pairwise_kernels.
            Default is 'rbf'.
        cutoff: float, optional kernal truncation value, entries below which
            are set to 0.
        **kwargs: passed to pairwise_kernels
        
    Returns: a networkx weighted Graph
    """
    kernel = pairwise_kernels(X, metric=metric, **kwargs)
    if cutoff:
        kernel[kernel < cutoff] = 0
    return from_numpy_matrix(kernel, create_using=Graph)
