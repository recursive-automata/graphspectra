# Graph Spectra

### Simon R Schneider, 2018

A set of tools for using spectral graph theory to analyze data.

```
# https://snap.stanford.edu/data/ego-Facebook.html
data_file = 'facebook_combined.txt'

import graphspectra as gs
graph = gs.read_graph(data_file, directed = False)
adj = gs.get_adjacency_matrix(graph)
lap = gs.compute_undirected_normalized_laplacian(adj)
values, vectors = gs.calculate_symmetric_eigensystem(lap)
```
