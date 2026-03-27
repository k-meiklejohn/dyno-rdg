from .nodes import RiboNode
from .transitions import RiboTransition

class RiboEvent(tuple):
    """
    Tuple in the form position:int, type: str, probability: float, drop_probability: float
    Probabilities must be 0 <= p <= 1 and must not add to be greater than 1
    """


    def _initiation_transition(self):
        transitions = []
        if self.probability > 0:
            transitions.append(RiboTransition(RiboNode(self.position, 0), 
                                              RiboNode(self.position, self.frame), 
                                              self.probability))
        if self.drop_probability > 0:
            transitions.append(self.bulk_transition)
        return transitions
    
    def _termination_transition(self):
        transitions = []
        if self.probability > 0:
            transitions.append(RiboTransition(RiboNode(self.position, self.frame), 
                                              RiboNode(self.position, 0), 
                                              self.probability))
        if self.drop_probability > 0:
            transitions.append(self.bulk_transition)
        return transitions
    
    def _frameshift_transition(self):
        transitions = []
        if self.probability > 0:
            transitions.append(RiboTransition(RiboNode(self.position, self.frame),
                                               RiboNode(self.position + self.shift, 0), 
                                               self.probability))
        if self.drop_probability > 0:
            transitions.append(self.bulk_transition)
        return transitions
    
    def _ires_transition(self):
        transitions = []
        if self.probability > 0:
            transitions.append(RiboTransition(RiboNode(self.position, -1), 
                                            RiboNode(self.position, self.frame), 
                                            self.probability))
        return transitions

    def _cap_transition(self):
        transitions = []
        if self.probability > 0:
            transitions.append(RiboTransition(RiboNode(self.position, -1), 
                                            RiboNode(self.position, 0), 
                                            self.probability))
        return transitions


    TRANSITION_MAP = {
        'initiation':  _initiation_transition,
        'termination':    _termination_transition,
        'frameshift': _frameshift_transition,
        'ires':  _ires_transition,
        'cap': _cap_transition
    }

    @property
    def bulk_transition(self):
        return RiboTransition(RiboNode(self.position, 0), RiboNode(-1, -1), self.drop_probability)


    @property
    def frame(self):
        return self.position % 3 + 1
    
    @property
    def shift(self):
        return self.type.str.extract(r'shift([+-]?\d+)').astype(int)

    def __new__(cls, *args):
            
            if len(args) == 1 and isinstance(args[0], (tuple, RiboEvent)):
                data = args[0]
            elif len(args) == 4:
                data = args
            else:
                raise ValueError(f'RiboEvent requires either or a length-4 tuple, got: {args}')

            if len(data) != 4:
                raise ValueError(f'RiboEvent tuple must be of length 4, got length: {len(data)}')
            

            if not isinstance(data[0], int):
                raise ValueError(f"Position of RiboEvent tuple must be 'int', got {type(data[0]).__name__!r}")
            elif not isinstance(data[1], str):
                raise ValueError(f"Type of RiboEvent tuple must be 'str', got {type(data[1]).__name__!r}")
            elif not all(isinstance(x, (float, int)) for x in data[2:]):
                raise ValueError(f"Probabilities of RiboEvent must be 'float' or 'int', \
                                 got {type(x for x in data[:2] if not isinstance(x, (float, int)))}")
            elif all(x == 0 for x in data[2:]):
                raise ValueError(f'At least one probability must be non-zero')
            if sum(data[2:]) > 1:
                raise ValueError(f"Sum of Probabilities for RiboEvent cannot exceed 1 or be 0, got {sum(data[2:])}")

            return super().__new__(cls, data)
    
    def __init__(self, *args):
        super().__init__()
        self.position = self[0]
        self.type = self[1]
        self.probability = self[2]
        self.drop_probability = self[3]

    def __repr__(self):
        return f"(Pos:{self.position}, type:{self.type}, prob:{self.probability}, drop:{self.drop_probability})"
    
    def _to_transition(self):
        """
        Returns a list of RiboTransitions coressponding to the event
        """
        handler = self.TRANSITION_MAP.get(self.type)
        if handler is None:
            raise ValueError(f"No transition defined for event type {self.type!r}")
        return handler(self)