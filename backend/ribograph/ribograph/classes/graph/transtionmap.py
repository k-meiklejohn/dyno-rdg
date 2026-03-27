class TransitionMap(RiboGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._is_valid()
        

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        raise NotImplementedError("Use add_weighted_edge() instead.")

    def add_edges_from(self, ebunch_to_add, **attr):
        raise NotImplementedError("Use add_weighted_edge() instead.")

    def add_weighted_edge(self, *args):
        # Coerce to RiboTransition, which handles validation
        if len(args) == 1 and isinstance(args[0], (tuple, RiboTransition)):
            transition = RiboTransition(args[0])
        elif len(args) == 3:
            transition = RiboTransition(*args)
        else:
            raise ValueError(f'add_weighted_edge requires a RiboTransition, a length-3 tuple, or 3 arguments, got: {args}')

        super().add_edge(transition.source, transition.target, weight=transition.probability)
        self._is_valid()

    def add_weighted_edges_from(self, ebunch_to_add):
        """ebunch_to_add: iterable of RiboTransitions or (source, target, weight) tuples"""
        for item in ebunch_to_add:
            self.add_weighted_edge(item)


    def _is_valid_weight(self):

        for node in self.nodes:

            if node.phase == -1:
                continue

            if not any(True for _ in self.successors(node)):
                continue

            total_weight = 0

            for (u, v, w) in self.out_edges(node, data='weight'):
                total_weight += w

            if total_weight <= 0 or total_weight > 1:
                raise ValueError(f'Total weight of edges from a single node (not from phase=-1) must be greater than 0 and less than or equal to 1\n' \
                f'Offending Node: {node}, Weight: {total_weight}')
            
    def _is_valid(self):
        self._is_valid_weight()

    def to_fluxgraph(self, half_life_translation = None, half_life_scanning = None):
        return RiboGraphFlux(transition_map=self, half_life_translation=half_life_translation, half_life_scanning=half_life_scanning)

