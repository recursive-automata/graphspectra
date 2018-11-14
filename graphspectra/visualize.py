import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pylab import subplots

def plot_graph(coordinates, adjacency_matrix, title = '',
               vertex_colors='black', vertex_alpha=0.25,
               edge_alpha=0.2, edge_weight_cutoff = 0.9):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    
    if edge_alpha:
        edges = adjacency_matrix > edge_weight_cutoff
        which_edges = np.asarray(np.argwhere(edges))
        line_coords = [[(x[a], y[a]),
                        (x[b], y[b])]
                       for (a, b) in which_edges]
        
        lines = LineCollection(line_coords, linewidths=0.1,
                               color = np.array([0, 0, 0, edge_alpha]))

        fig, ax = subplots()
        ax.add_collection(lines)
        ax.autoscale()
        ax.margins(0.1)
    if title:
        plt.title(title)

    plt.scatter(x, y, s=40, c=vertex_colors, alpha=vertex_alpha)
    plt.show()
