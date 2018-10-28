# Graph Spectra

### Simon R Schneider, 2018

A set of tools for using spectral graph theory to analyze data.

```
import graphspectra as gs
import networkx as nx

# https://snap.stanford.edu/data/ego-Facebook.html
data_file = 'facebook_combined.txt'

graph = gs.read_graph(data_file, directed = False)
adj = nx.to_numpy_matrix(graph)
lap = gs.compute_laplacian(adj)
values, vectors = gs.calculate_symmetric_eigensystem(lap)
gs.plot_graph(vectors, adj)
```
