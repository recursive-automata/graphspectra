# Graph Spectra

## Simon R Schneider, 2018

A set of tools for using spectral graph theory to analyze data.

### Installation

From the repo's root folder: `pip3 install -r requirements.txt; python3 setup.py install` or similar.

### Quickstart

```
import networkx as nx
import graphspectra as gs

## read a graph from file
## https://snap.stanford.edu/data/ego-Facebook.html
# data_file = 'facebook_combined.txt'
# graph = gs.read_graph(data_file, directed=False)

## make a weighted graph from numeric observations
# import numpy as np
# X = np.random.random((1000, 3)) - 0.5
# graph = gs.make_kernel_graph(X, metric='rbf')

## generate a random graph
graph = nx.connected_watts_strogatz_graph(1000, 25, 0.05)

adj = nx.to_numpy_matrix(graph)
lap = gs.compute_laplacian(adj)
values, vectors = gs.calculate_small_eigens(lap, k=3)
gs.plot_graph(vectors[:, 1:], adj)
```
