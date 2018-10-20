from networkx import Graph, DiGraph
from networkx.convert_matrix import to_numpy_matrix


def read_graph(filepath, directed=True, attr_names=None):
    """ Read a graph from file.
    
    Params:
        filepath -- a string, representing whence to read the graph.
        directed -- a boolean, whether the graph is directed. Defaults to True.
        attr_names -- names of any edge attributes. Assumes there aren't any.
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


# Paper thin wrapper around networkx.convert_matrix.to_numpy_matrix. Saves
# ya the trouble of reimporting it.
get_adjacency_matrix = to_numpy_matrix
    