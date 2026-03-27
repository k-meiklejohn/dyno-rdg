from .nodes import RiboNode

class RiboTransition(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, RiboTransition)):
            data = args[0]
        elif len(args) == 3:
            data = args
        else:
            raise ValueError(f'RiboTransition requires either 3 arguments or a length-3 tuple, got: {args}')

        if len(data) != 3:
            raise ValueError(f'RiboTransition must be of length 3, got length: {len(data)}')

        source, target, prob = data

        # Coerce RiboNode-like tuples
        if not isinstance(source, RiboNode):
            if isinstance(source, tuple):
                source = RiboNode(source)
            else:
                raise TypeError(f"Source must be 'RiboNode' or RiboNode-like tuple, got {type(source).__name__!r}")
        if not isinstance(target, RiboNode):
            if isinstance(target, tuple):
                target = RiboNode(target)
            else:
                raise TypeError(f"Target must be 'RiboNode' or RiboNode-like tuple, got {type(target).__name__!r}")

        if not isinstance(prob, (int, float)):
            raise TypeError(f"Probability must be 'float' or 'int', got {type(prob).__name__!r}")
        if not 0 < prob <= 1:
            raise ValueError(f"Probability must be in range (0, 1], got {prob}")

        return super().__new__(cls, (source, target, prob))

    def __init__(self, *args):
        super().__init__()
        self.source = self[0]
        self.target = self[1]
        self.probability = self[2]

    def __repr__(self):
        return f"(Source:{self.source}, Target:{self.target}, Probability:{self.probability})"