class Obstacle:
    __slots__ = ["id", "x", "y"]

    def __init__(self, id=0, x=0, y=0):
        self.id = id
        self.x = x
        self.y = y