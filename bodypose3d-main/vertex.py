class Bodyparts():
    name = ''
    landmarkList = []

    def __init__(self, name, landmarkList):
        self.name = name
        self.landmarkList = landmarkList

class Vertex(Bodyparts):
    index = 0
    x = 0
    y = 0
    z = 0

    def __init__(self, index, x, y, z):
        self.index = index
        self.x = x
        self.y = y
        self.z = z