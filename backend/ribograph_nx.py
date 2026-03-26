import networkx as nx
import warnings
import matplotlib

class RiboNode():
    def __init__(self, coords: tuple[int, int]):
        if isinstance(coords, RiboNode):
            # Copy constructor — just take the coords from the existing node
            self.coords = coords.coords
        elif isinstance(coords, tuple):
            if not (len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int)):
                raise ValueError('RiboNode tuples must be of form (int, int)')
            self.coords = coords
        else:
            raise TypeError(f'Expected a (int, int) tuple or a RiboNode, got {type(coords).__name__!r}')

        self.position = self.coords[0]
        self.phase = self.coords[1]
    
    def __repr__(self):
        return f"(Pos:{self.position}, Phase:{self.phase})"
    
    def __eq__(self, value):
        if not isinstance(value, RiboNode):
            return NotImplemented
        return self.position == value.position and self.phase == value.phase
    
    def __hash__(self):
        return hash((self.position, self.phase))
    
class list_of_RiboNodes(list):
    """
    A list of Node objects with bulk access to position and phase
    """
    def __init__(self, iterable):
        for n in iterable:
            if not isinstance(n, RiboNode):
                raise ValueError(f"list_of_RiboNodes must contain only RiboNodes, got {type(n).__name__!r}")
        super().__init__(iterable)

    @property
    def positions(self) -> list[int]:
        return [n.position for n in self]
    
    @property
    def phases(self) -> list[int]:
        return [n.phase for n in self]

class MissingEdgeWeightWarning(UserWarning):
    pass
    
class RiboGraph(nx.DiGraph):
    """
    Digraph that only accepts RiboNode instances as nodes
    """

    def __init__(self, incoming_graph_data = None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.bulk_node = RiboNode((-1,-1))

    def add_node(self, node_for_adding, **attr):
        if not isinstance(node_for_adding, RiboNode):
            node_for_adding = RiboNode(node_for_adding)
        else:
            raise TypeError(f"Ribograph type only accepts RiboNodes, or RiboNode-like tuples, got {type(node_for_adding).__name__!r}")
        super().add_node(node_for_adding, **attr)
    
    def add_nodes_from(self, nodes_for_adding, **attr):
        for node in nodes_for_adding:
            n = nodes_for_adding[0] if isinstance(node, tuple) else node
            if not isinstance(n, RiboNode):
                raise TypeError(f'RiboGraph type only accepts RiboNodes or RiboNode-like tuples, got {type(n).__name__!r}')
        super().add_nodes_from(nodes_for_adding, **attr)
    
    
    def add_edge(self, u_of_edge: RiboNode|tuple[int,int], v_of_edge: RiboNode|tuple[int,int], **attr):
        return super().add_edge(RiboNode(u_of_edge), RiboNode(v_of_edge), **attr)

    def successor_edges(self, node, data=False):
        return self.edges([node, *self.successors(node)], data=data)






class TransitionMap(RiboGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._is_valid()
        

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        raise NotImplementedError("Use add_weighted_edge() instead.")

    def add_edges_from(self, ebunch_to_add, **attr):
        raise NotImplementedError("Use add_weighted_edge() instead.")

    def add_weighted_edge(self, u_of_edge, v_of_edge, weight):
        super().add_edge(u_of_edge, v_of_edge, weight=weight)
        self._is_valid()

    def add_weighted_edges_from(self, ebunch_to_add):
        """ebunch_to_add: iterable of (u, v, weight) tuples"""
        for u, v, weight in ebunch_to_add:
            self.add_weighted_edge(u, v, weight)

    def _is_valid_weight(self):

        for node in self.nodes:

            if node.phase == -1:
                continue

            if not any(True for _ in self.successors(node)):
                continue

            total_weight = 0

            for (u, v, w) in self.successor_edges(node, data='weight'):
                total_weight += w

            if total_weight <= 0 or total_weight > 1:
                raise ValueError(f'Total weight of edges from a single node (not from phase=-1) must be greater than 0 and less than or equal to 1\n' \
                f'Offending Node: {node}')
            
    def _is_valid(self):
        self._is_valid_weight()

    def to_fluxgraph(self):
        return RiboGraphFlux(self)


class RiboGraphFlux(RiboGraph):
    def __init__(self, incoming_graph_data=None, map=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.tranitions = map
        self.begun = False
        if map:
            self._construct()   
    
    def _construct(self):
        node = RiboNode((-1,-1))
        flux = 1

        self._iterate_graph(node, flux)

    def _downstream_node(self, node: RiboNode):
        return RiboNode(( min([p.position
                            for p in self.tranitions.nodes 
                            if p.position > node.position 
                            and p.phase == node.phase]),

                            node.phase))
    
    def _iterate_graph(self, node: RiboNode, flux):

        if self.begun and node == self.bulk_node:
            return None
        elif self.bulk_node:
            self.begun = True

        next_node = self._downstream_node(node)

        self.add_edge(node, next_node, flux=flux)

        total_weight = 1

        for u, v, w in self.tranitions.successor_edges(next_node, data='weight'):
            total_weight -= w
            new_flux = flux * w
            self.add_edge(u, v, flux=new_flux)
            self._iterate_graph(v, new_flux)

        if total_weight == 0:
            return None
        else:
            self._iterate_graph(next_node, new_flux)


    
transition_list = [ 
    ((1,-1), (1,0),   1),
    ((10,0), (10,1),  0.4),
    ((10,0), (-1,-1), 0.4),
    ((20,0), (-1,-1), 1),
    ((20,1), (-1,-1), 1)
    ]

x = TransitionMap()
x.add_weighted_edges_from(transition_list)
y = x.to_fluxgraph()
nx.display(y)