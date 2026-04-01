class RiboNode(tuple):
    def __new__(cls, *args):
        
        if len(args) == 1 and isinstance(args[0], (tuple, RiboNode)):
            coords = args[0]
        elif len(args) == 2:
            coords = args
        else:
            raise ValueError(f'RiboNode requires 2 ints or a length-2 tuple, got: {args}')

        if len(coords) != 2:
            raise ValueError(f'RiboNode tuple must be of length 2, got length: {len(coords)}')
        for coord in coords:
            if not isinstance(coord, int):
                raise ValueError(f"RiboNode coordinates must be 'int', got {type(coord).__name__!r}")

        return super().__new__(cls, coords)

    def __init__(self, *args):
        super().__init__()
        self.position = self[0]
        self.phase = self[1]

    def __repr__(self):
        return f"(Pos:{self.position}, Phase:{self.phase})"
    