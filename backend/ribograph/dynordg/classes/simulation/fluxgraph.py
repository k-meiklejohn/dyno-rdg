from ..graph import RiboGraph
from .transitionmap import TransitionMap
from ..core import RiboEvent, RiboNode, RiboTransition
import networkx as nx

class RiboGraphFlux(RiboGraph):
    def __init__(self, transition_map: TransitionMap, incoming_graph_data=None, half_life_scanning = None, half_life_translation = None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.transitions = transition_map
        self.begun = False
        self.half_life_translation = half_life_translation
        self.half_life_scanning = half_life_scanning
        if map:
            self._construct()   

    @classmethod
    def from_transition_map(cls, transition_map, half_life_translation=None, half_life_scanning=None):
        return cls(transition_map=transition_map)
    
    def _construct(self):
        node = RiboNode((-1,-1))
        flux = 1

        self._iterate_graph(node, flux)
        self._is_valid()


    def _downstream_node(self, node: RiboNode):
        if not any([p.position > node.position 
                            and p.phase == node.phase
                            for p in self.transitions.nodes ]):
            return None
            
        return RiboNode(( min([p.position
                            for p in self.transitions.nodes 
                            if p.position > node.position 
                            and p.phase == node.phase]),

                            node.phase))
    
    def _iterate_graph(self, node: RiboNode, flux, weight=1):

        ###### Handle return to starting node ######
        if self.begun and node == self.bulk_node:
            return None
        elif self.bulk_node:
            self.begun = True

        next_node = self._downstream_node(node)
        if next_node is None:
            return None
    
        #### Calculate decay of ribosomes based on half life ####

        endflux = flux * self.edge_decay(node, next_node)
        drop_flux = flux - endflux
        if drop_flux != 0:
            self.add_edge(next_node, self.bulk_node, flux_start=drop_flux, flux_end=drop_flux, weight=weight*drop_flux/flux) # this is the drop edge
        self.add_edge(node, next_node, flux_start=flux, flux_end = endflux, weight=weight*endflux/flux) # this is the horizontal edge

        #### calculate flux for each edge off next node ####

        remaining_weight = 1
        for u, v, w in self.transitions.out_edges(next_node, data='weight'):
            remaining_weight -= w
            new_flux = endflux * w
            self.add_edge(u, v, flux_start=new_flux, flux_end=new_flux, weight=w)
            self._iterate_graph(v, new_flux)

        #### Continue graph on same phase if weight remaining ####
        if remaining_weight == 0:
            return None
        else:
            self._iterate_graph(next_node, endflux*remaining_weight, weight=remaining_weight)
    
    def add_transition(self, source, target, probability):
        """
        Adds new tranistion to graph. 
        """
        self.transitions.add_weighted_edge(source, target, probability)
        self.clear_edges()
        self._construct()

    def add_transitions_from(self, tbunch):
        """
        Adds transitions from an iterable. Must be of form (source, target, weight)
        Flux is recalculated after adding.
        """
        self.transitions.add_weighted_edges_from(tbunch)
        self.clear_edges
        self._construct

    def edge_decay(self, u: RiboNode, v: RiboNode):
        if u.phase == v.phase or (u.phase > 0 and v.phase > 0):
            if u.phase > 0:
                half_life = self.half_life_translation
            elif u.phase == 0:
                half_life = self.half_life_scanning
            else:
                return 1
            
            if half_life == None:
                return 1
            
            return 0.5 ** (abs(u.position-v.position) / half_life )
        
        else:
            ValueError(f"Edge: {u}, {v} is not capable of decay")

    def _is_valid(self):
        self._valid_in_out()

    @property
    def ribopaths(self) -> list:
        """
        Returns a list of the paths with continued 40S association, each as a list of edge tuples.
        """
        paths = []
        for loading in self.successors(self.bulk_node):
            paths.extend(nx.all_simple_edge_paths(self, loading, self.bulk_node))
        return paths
    
    @property
    def translons(self) -> list:
        """
        Returns a list of all translons in the graph (continued 60S association) as a list of edge tuples
        """
        translon_list = []
        for path in self.ribopaths:
            translon = False
            current_translon = []
            for edge in path:

                if translon:
                    if edge[1].phase < 1:
                        translon = False
                        translon_list.append(current_translon)

                    else:
                        current_translon.append(edge)

                elif edge[0].phase > 0:
                    translon = True
                    current_translon.append(edge)

        return translon_list

    def flux_proportion(self, path: list[tuple[RiboNode,RiboNode]]) -> float|None:
        if path is not None:
            weights = nx.get_edge_attributes(self, name='weight')
            paths_to_start = nx.all_simple_edge_paths(self, self.bulk_node, path[0][0])
            proportion = 1.0


            if paths_to_start is None:
                raise ValueError(f"No path exists from {self.bulk_node} to {self[0][0]}")
            
            start_proportion = 0
            for _path in paths_to_start:
                for edge in _path:
                    proportion *= weights[edge]
                start_proportion += proportion


            proportion = start_proportion
            if path[-1][1] != self.bulk_node:
                for finish_edge in self.out_edges(path[-1][1]):
                    if finish_edge[1].phase < 1:
                        new_path = path.copy()
                        for edge in new_path:
                            proportion *= weights[edge]
                
            else:
                for edge in path:
                    proportion *= weights[edge]
        else:
            raise ValueError('No weight associated with empty path')

        return proportion

    def _valid_in_out(self):
        out_flux = 0
        for u, v, flux in self.in_edges(self.bulk_node, data='flux_end'):
            out_flux += flux

        in_flux = 0
        for node in self.successors(self.bulk_node):
            for u, v, flux in self.out_edges(node, data='flux_start'):
                in_flux += flux
        if out_flux != in_flux:
            # return None
            raise ValueError(f'Flux in does not equal flux out In:{in_flux} vs Out:{out_flux}')