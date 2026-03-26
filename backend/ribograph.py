

class Node():
    def __init__(self, coords: tuple[int,int]):
        self.coords = coords
        self.position = coords[0]
        self.phase = coords[1]
    
    def __repr__(self):
        return f"Node(Position:{self.position}, Phase:{self.phase})"
    
    def __eq__(self, value):
        if not isinstance(value, Node):
            return NotImplemented
        return self.position == value.position and self.phase == value.phase
    
    def __hash__(self):
        return hash((self.position, self.phase))


class Edge:
    def __init__(self, source: Node|tuple, target: Node|tuple, values: dict):
        if isinstance(source, Node):
            self.source = source
        elif isinstance(source, tuple):
            self.source = Node(source)
        else:
            raise ValueError('Edge must be made of nodes, or node-like tuples')
        if isinstance(target, Node):
            self.target = target
        elif isinstance(target, tuple):
            self.target = Node(target)
        else:
            raise ValueError('Edge must be made of nodes, or node-like tuples')
        if isinstance(values, dict):
            self.values = values
        elif isinstance(values, (float, int)):
            self.values = {'weight':values}
        
        if self.source.phase == self.target.phase:
            self.phase = str(self.source.phase)
        else:
            self.phase = str(str(self.source.phase) + '/' +  str(self.target.phase))

    def __eq__(self, value):
        if not isinstance(value, Edge):
            return NotImplemented
        return self.source == value.source and self.target == value.target and self.weight == value.weight
    

    def __getattr__(self, name):
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"'Edge' object has no attribute '{name}'")
    
    def __hash__(self):
        return hash ((self.source, self.target, self.weight))

    def __repr__(self):
        return f"Edge({(self.source.position, self.source.phase)}->" \
                    f"{(self.target.position, self.target.phase)}, values={self.values})"
        

class list_of_edges(list):
    """A list of Edge objects with bulk field access."""
    
    @property
    def sources(self) -> list[Node]:
        return list_of_nodes(e.source for e in self)

    @property
    def targets(self) -> list[Node]:
        return list_of_nodes(e.target for e in self)

    @property
    def weights(self) -> list[float]:
        return [e.weight for e in self]
    
    @property
    def empty(self):
        return not bool(self)  # empty list is falsy
    
    @property
    def nodes(self):
        nodes = []
        nodes.extend(self.sources)
        nodes.extend(self.targets)
        nodes = set(nodes)
        return list_of_nodes(nodes)

        
class list_of_nodes(list):
    """
    A list of Node objects with bulk access to position and phase
    """

    @property
    def positions(self) -> list[int]:
        return [n.position for n in self]
    
    @property
    def phases(self) -> list[int]:
        return [n.phase for n in self]
    
class Graph:
    def __init__(self, edges=None):
        if edges == None:
            self.edges = list_of_edges()
        elif isinstance(edges, (Edge, list)):
            self.edges = list_of_edges()
            self.add(edges)
        else:
            raise ValueError('The input must be an edge or list of edges')
        
    def add(self, edge):
        self._add_internal(edge)

    def _add_internal(self, edge):
        if isinstance(edge, Edge):
            self.edges.append(edge)

        elif isinstance(edge, tuple):
            self.edges.append(Edge(*edge))

        elif isinstance(edge, list):
            for e in edge:
                self._add_internal(e)

        else:
            raise ValueError("Invalid input")
    

    def parents(self, node: Node) -> list[Edge]:
        """
        Returns all parents of a given node
        """ 
        return list_of_nodes(e.source for e in self.edges if e.target == node )
    

    def children(self, node: tuple[int, int]) -> list[tuple[int,int]]:
        """
        Returns all children of a given node
        """
        return list_of_nodes(e.target for e in self.edges if e.source == node )
    
    def child_edges(self, node: Node) -> list[Edge]:
        return list_of_edges(e for e in self.edges if e.source == node)


class Transition_Map(Graph):
    def __init__(self, edges=None):
        super().__init__(edges)
        self._is_valid()

    def add(self, edge):
        self._add_internal(edge)
        self._is_valid()
    
    def _is_valid(self):
        for node in self.edges.nodes:
            if node.phase == -1:
                continue
            if self.child_edges(node).empty:
                continue
            total_weight = sum(self.child_edges(node).weights)
            if total_weight <= 0 or total_weight > 1:
                raise ValueError(f'Weight from node must be greater than 0 and less than or equal to 1\n' \
                f'Offending Edges: {self.child_edges(node)}')
            
        
    def by_source(self, source: Node) -> list[Edge]:
        """Return all edges with the given source node."""
        return list_of_edges(e for e in self.edges if e.source == source)

    def by_target(self, target: Node) -> list[Edge]:
        """Return all edges with the given target node."""
        return list_of_edges(e for e in self.edges if e.target == target)

    def by_weight(self, weight: float) -> list[Edge]:
        """Return all edges with the given weight."""
        return list_of_edges(e for e in self.edges if e.weight == weight)

    def by_node(self, node: Node) -> list[Edge]:
        """Return all edges where node appears as either source or target."""
        return list_of_edges(e for e in self.edges if e.source == node or e.target == node)

    def by_field(self, field: str, value) -> list[Edge]:
        """Generic lookup — return all edges where field matches value.
        
        e.g. x.by_field('source', (0,0))
        """
        return list_of_edges(e for e in self.edges if getattr(e, field) == value)
            
class Ribograph(Graph):
    def __init__(self, map: Transition_Map):
        self.map = map
        self.edges = list_of_edges()
        self.begun = False
        self._construct_graph()


    def _construct_graph(self):
        origin = Node((-1,-1))
        flux = 1
        self._iterate(origin, flux)


    def _find_downstream(self, node: Node):
        """
        Findes the next node (or nodes if phase = -1) for a given node
        """
        phase_edges = list_of_edges(e for e in self.map.edges if e.source.position > node.position and 
                                   e.source.phase == node.phase)
        
        if phase_edges.empty:
            return list_of_edges()
        
        if node.phase == -1:
            return phase_edges
        
        minimum_position = min(phase_edges.sources.positions)
        next_edges = self.map.by_source(Node((minimum_position, node.phase)))
        return next_edges
    
    def _iterate_flux_search(self, node, flux, cont = False):
        """
        Iteratively searchs through the edges to calculate all paths of the ribosome
        """

        if node == Node((-1,-1)) and self.begun:
            return None
        elif node == Node((-1,-1)):
            self.begun = True

        next_edges = self._find_downstream(node)

        if next_edges.empty:
            return None
        remaining_weight = 1
        self.edges.append(Edge(node, next_edges[0].source, {'weight': 1, 'flux': flux}))

        for edge in next_edges:
            if not cont:
                remaining_weight -= edge.weight
            edge.flux = flux * edge.weight
            self._iterate_flux_search(edge.target, edge.flux)

        if not (remaining_weight == 0):
            self._iterate_flux_search(next_edges[0].source, remaining_weight * flux, True)
    
    def paths(self):
        node = node

    def _combine_flux(self):




transition_list = [ 
    Edge((1,-1), (1,0),   1),
    Edge((10,0), (10,1),  0.4),
    Edge((10,0), (-1,-1), 0.4),
    Edge((20,0), (-1,-1), 1),
    Edge((20,1), (-1,-1), 1)
    ]


x = Transition_Map(transition_list)
print(x.edges)
print(Ribograph(x).edges)

