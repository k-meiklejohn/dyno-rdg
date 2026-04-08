#!/bin/env python

import networkx as nx
import matplotlib.pyplot as plt
import dynordg as dr


    
transition_list = [ 
    ((0,-1), (0,0),   1),
    ((10,0), (10,1),  0.5),
    ((20,1), (20,0), 0.9),
    ((30,2), (30,-1), 1),
    ((30,0), (30,-1), 1),
    ((30,1), (30,-1), 1)
    ]

x = dr.TransitionMap()
x.add_weighted_edges_from(transition_list)
y = dr.RiboGraphFlux.from_transition_map(x)



z = dr.RiboGraphVis(y)
for u, v, data in z.edges(data=True):
    print('=====================================')
    print(u, v)
    print(data)

# z.show()