from ..graph import RiboGraph
from .transitionmap import TransitionMap
from ..core import RiboEvent, RiboNode, RiboTransition


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
    
    def _iterate_graph(self, node: RiboNode, flux):

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
        print(node, 'flux:', flux, 'end_flux:', endflux, 'drop_flux:', drop_flux)
        if drop_flux != 0:
            self.add_edge(next_node, self.bulk_node, flux_start=drop_flux, flux_end=drop_flux) # this is the drop edge

        self.add_edge(node, next_node, flux_start=flux, flux_end = endflux) # this is the horizontal edge

        #### calculate flux for each edge off next node ####

        total_weight = 1
        for u, v, w in self.transitions.out_edges(next_node, data='weight'):
            total_weight -= w
            new_flux = endflux * w
            self.add_edge(u, v, flux_start=new_flux, flux_end=new_flux)
            self._iterate_graph(v, new_flux)

        #### Continue graph on same phase if weight remaining ####
        if total_weight == 0:
            return None
        else:
            self._iterate_graph(next_node, endflux*total_weight)
    
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
        if u.phase == v.phase:
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