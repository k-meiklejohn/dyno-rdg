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

        x, y = coords

        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("RiboNode coordinates must be ints")

        if not (-1 <= y < 4):
            raise ValueError("position must be between -1 and 3")

        return super().__new__(cls, (x, y))

    @property
    def position(self) -> int:
        return self[0]

    @property
    def phase(self) -> int:
        return self[1]

    def __repr__(self):
        return f"(Pos:{self.position}, Phase:{self.phase})"