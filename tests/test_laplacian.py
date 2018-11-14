import numpy as np
import scipy.sparse as ss

import graphspectra as gs

n = 5
adj = ss.lil_matrix((n, n), dtype=np.int)
for i in range(n):
    j = (1 + i) % n
    adj[i, j] = 1

adj[0, int(n/2)] = 1

# directed adjacency matrix
adj_ds = adj.copy()
adj_d = adj.toarray()

# undirected adjacency matrix
adj_s = adj + adj.transpose()
adj = adj_s.toarray()


def test_directed_out_degrees():
    out_degrees = np.ones(n)
    out_degrees[0] = 2
    assert (gs.count_out_degrees(adj_ds) == out_degrees).all()
    assert (gs.count_out_degrees(adj_d) == out_degrees).all()


def test_undirected_out_degrees():
    out_degrees = 2 * np.ones(n)
    out_degrees[0] = 3
    out_degrees[int(n/2)] = 3
    assert (gs.count_out_degrees(adj_s) == out_degrees).all()
    assert (gs.count_out_degrees(adj) == out_degrees).all()

# TODO more tests
