#!/bin/env python

import networkx as nx
import matplotlib.pyplot as plt
import dynordg as dr
from Bio import SeqIO


    
transition_list = [ 
    ((0,-1), (0,0),   1),
    ((10,0), (10,1),  0.5),
    ((20,1), (20,0), 0.5),
    ((5,0), (5,3), 0.3),
    ((7,-1), (7,2), 1),
    ((16,1), (17,2), 0.5),
    ((18,3), (17,2), 0.4),
    ((17,2), (17,0), 0.5),
    ((20,1), (20,-1), 0.2),
    ((30,2), (30,-1), 1),
    ((30,0), (30,-1), 1),
    ((30,1), (30,-1), 1),
    ((30,3), (30,-1), 1)
    ]

sequence = SeqIO.read('nsun2.fasta', format='fasta')
nsun2 = dr.RNASequence(sequence.seq)

y = dr.RiboGraphFlux.from_transition_map(nsun2.transition_map
                                        #  , half_life_scanning=50, half_life_translation=5
                                         )
# y.remove_node(y.bulk_node)
# pos = nx.planar_layout(y)
# nx.draw(y, pos=pos, with_labels=True)
# plt.show()

z = dr.RiboGraphVis(y)


z.show()