#!/bin/env python

import networkx as nx
import warnings
import matplotlib.pyplot as plt


    









            

    
transition_list = [ 
    ((0,-1), (0,0),   1),
    ((10,0), (10,1),  0.5),
    ((20,1), (20,0), 0.9),
    ((5,0), (5,2), 0.5),
    ((30,2), (-1,-1), 1),
    ((30,0), (-1,-1), 1),
    ((30,1), (-1,-1), 1)
    ]

x = TransitionMap()
x.add_weighted_edges_from(transition_list)
y = x.to_fluxgraph(half_life_scanning=10, half_life_translation=10)

pos = nx.planar_layout(y)

nx.draw_networkx_edge_labels(y, pos)
print(y.edges.data())
nx.draw(y, pos=pos, with_labels=True)
plt.show()