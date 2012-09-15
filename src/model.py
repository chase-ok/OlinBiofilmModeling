
import numpy as np
import cv2
import cv
from matplotlib import pyplot as plt
import random as rdm


ALIVE = 1
EMPTY = 0


class CellModel2D(object):

    def __init__(self, numCells=(256, 256),
                       boundaryLayerThickness=4,
                       lightPenetrationDepth=4,
                       mediaConcentration=1.0,
                       mediaPenetrationDepth=2,
                       divisionConstant=0.5):
        self.numCells = np.array(numCells, dtype=int)
        self.boundaryLayerThickness = boundaryLayerThickness
        self.lightPenetrationDepth = lightPenetrationDepth
        self.mediaConcentration = mediaConcentration
        self.mediaPenetrationDepth = mediaPenetrationDepth
        self.divisionConstant = divisionConstant

        self.time = 0
        self.invalidate()
        self._createCells()

    def _createCells(self):
        self.biofilm = self._zeros(np.uint8)
        self.boundaryLayer = self._zeros(np.uint8)
        self.media = self._zeros(float)
        self.light = self._zeros(float)
        self.lifetime = self._zeros()
        self.divisionProbability = self._zeros(float)
        self.dividing = self._zeros(bool)
        self.distances = self._zeros(int) # distance to nearest empty cell

    def placeRandomCellsAtBottom(self, probability=0.2):
        for column in range(self.numColumns):
            if rdm.random() < probability:
                self.biofilm[0, column] = ALIVE

    def placeCellsRegularly(self, spacing=8):
        for column in range(0, self.numColumns, spacing):
            self.biofilm[0, column] = ALIVE

    def _calculateMedia(self):
        boundaryKernel = _makeCircularKernel(self.boundaryLayerThickness)
        cv2.filter2D(self.biofilm, -1, boundaryKernel, self.boundaryLayer)

        np.logical_not(self.boundaryLayer, out=self.media)
        self.media *= self.mediaConcentration

        cv2.GaussianBlur(self.media, (0, 0), self.mediaPenetrationDepth,
                         dst=self.media)

    def _calculateLight(self):
        np.cumsum(self.biofilm, axis=0, out=self.light)
        self.light /= -self.lightPenetrationDepth
        np.exp(self.light, out=self.light)

    def _calculateDividingCells(self):
        self.divisionProbability = self.divisionConstant*self.media*self.light
        self.divisionProbability[np.logical_not(self.biofilm)] = 0

        self.dividing = np.random.ranf(self.numCells) <= \
                        self.divisionProbability

    def _doDivision(self):
        cv2.distanceTransform(self.biofilm, cv.CV_DIST_L2, 5, 
                              dst=self.distances)
        rows, columns = map(list, self.dividing.nonzero())
        displacements = [0 for _ in rows]

        while rows:
            cellIndex = np.array((rows.pop(), columns.pop()))
            displacement = displacements.pop()

            # this also takes care of the situation where updating the biofilm
            # in-place breaks the distance map implmentation.
            if displacement > 10: continue

            foundEmpty = False
            for neighbor in _randomNeighbors():
                index = tuple(cellIndex + neighbor)
                if not self._validIndex(index): continue

                if self.biofilm[index] == EMPTY:
                    foundEmpty = True
                    self.biofilm[index] = ALIVE

            if not foundEmpty:
                minDistance = 999999
                minNeighbor = None
                for neighbor in _neighbors:
                    index = tuple(cellIndex + neighbor)
                    if not self._validIndex(index): continue

                    if self.distances[index] <= minDistance:
                        minDistance = self.distances[index]
                        minNeighbor = index
                rows.append(minNeighbor[0])
                columns.append(minNeighbor[1])
                displacements.append(displacement + 1)

    def step(self):
        self.invalidate()

        self._calculateMedia()
        self._calculateLight()
        self._calculateDividingCells()
        self._doDivision()

    def invalidate(self):
        """Invalidates any cached computations using the current matrices."""
        self._contours = None

    def _zeros(self, dtype=int):
        return np.zeros(self.numCells, dtype=dtype)

    def _validIndex(self, index):
        return (index >= _topLeft).all() and (index < self.numCells).all()

    @property
    def numRows(self):
        return self.numCells[0]

    @property
    def numColumns(self):
        return self.numCells[1]

    @property
    def area(self):
        return self.biofilm.sum().sum()

    @property
    def contours(self):
        if self._contours is None:
            # findContours modifies the image
            self._contours, _ = cv2.findContours(np.copy(self.biofilm), 
                                                 cv2.RETR_LIST, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
        return self._contours

    @property
    def perimeter(self):
        return sum(cv2.arcLength(c, True) for c in self.contours)

    @property
    def perimeterToAreaRatio(self):
        return self.perimeter/float(self.area)


def _makeCircularKernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

_neighbors = []
for dRow in [1, 0, -1]:
    for dColumn in [1, 0, -1]:
        if not (dRow == 0 and dColumn == 0):
            _neighbors.append(np.array([dRow, dColumn]))

def _randomNeighbors():
    rdm.shuffle(_neighbors)
    return _neighbors

_topLeft = np.array([0, 0])