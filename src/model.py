
import numpy as np
import cv2
import cv
from matplotlib import pyplot as plt
import random as rdm


ALIVE = 1
EMPTY = 0

class BiofilmModel2D(object):

    def __init__(self, numCells=(256, 256)):
        self.numCells = numCells
        self.biofilm = self._zeros(np.uint8)

        self.time = 0
        self.invalidate()

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

    @property
    def maxHeight(self):
        for row in reversed(range(self.numRows)):
            if self.biofilm[row, :].any():
                return row

    @property
    def meanHeight(self):
        rowSums = self.biofilm.sum(axis=1)
        return np.dot(np.arange(0, self.numRows), rowSums)/float(rowSums.sum())

    def rowFFT(self, row):
        return np.fft.fftfreq(self.numColumns), np.fft.fft(self.biofilm[row, :])

class CAModel2D(BiofilmModel2D):

    def __init__(self, boundaryLayerThickness=4,
                       lightPenetrationDepth=4,
                       mediaConcentration=1.0,
                       mediaPenetrationDepth=2,
                       divisionConstant=0.5,
                       **biofilmModelArgs):
        BiofilmModel2D.__init__(self, **biofilmModelArgs)

        self.boundaryLayerThickness = boundaryLayerThickness
        self.lightPenetrationDepth = lightPenetrationDepth
        self.mediaConcentration = mediaConcentration
        self.mediaPenetrationDepth = mediaPenetrationDepth
        self.divisionConstant = divisionConstant

        self.boundaryLayer = self._zeros(np.uint8)
        self.media = self._zeros(float)
        self.light = self._zeros(float)
        self.divisionProbability = self._zeros(float)
        self.dividing = self._zeros(bool)
        self.surfaceTension = self._zeros(float)

    def step(self):
        raise NotImplemented()

    def placeRandomCellsAtBottom(self, probability=0.2):
        for column in range(self.numColumns):
            if rdm.random() < probability:
                self.biofilm[0, column] = ALIVE

    def placeCellsRegularly(self, spacing=8):
        start = int(spacing/2)
        end = self.numColumns - int(spacing/2)
        for column in range(start, end, spacing):
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

    def _calculateSurfaceTension(self, centerFactor=0):
        k = centerFactor
        tensionKernel = np.array([[1, 2, 1],
                                  [2, k, 2],
                                  [1, 2, 1]], dtype=np.uint8)
        localSum = cv2.filter2D(self.biofilm, -1, tensionKernel)
        self.surfaceTension = localSum/np.float(tensionKernel.sum().sum())

    def _calculateDivisionProbability(self):
        self.divisionProbability = self.divisionConstant*\
                                   self.media*self.light
        self.divisionProbability[np.logical_not(self.biofilm)] = 0

    def _calculateDividingCells(self):
        self.dividing = np.random.ranf(self.numCells) <= \
                        self.divisionProbability

    def _calculateThroughDivisionProbability(self):
        self._calculateMedia()
        self._calculateLight()
        self._calculateDivisionProbability()


class MinimumDistanceModel2D(CAModel2D):

    def __init__(self, surfaceTensionFactor=10,
                       **caModelArgs):
        CAModel2D.__init__(self, **caModelArgs)

        self.surfaceTensionFactor = surfaceTensionFactor
        self.distances = self._zeros(int) # distance to nearest empty cell

    def step(self):
        self._calculateThroughDivisionProbability()
        self._calculateSurfaceTension(self.surfaceTensionFactor)
        self.divisionProbability *= self.surfaceTension
        self._calculateDividingCells()
        self._divide()

    def _divide(self):
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

class ProbabilisticModel2D(CAModel2D):
    def __init__(self, distancePower=0.5,
                       tensionPower=2.5,
                       blockSize=7,
                       **caModelArgs):
        CAModel2D.__init__(self, **caModelArgs)

        self.distancePower = distancePower
        self.tensionPower = tensionPower
        self.blockSize = blockSize

    def step(self):
        self._calculateThroughDivisionProbability()
        self._calculateSurfaceTension()
        self._calculateDividingCells()
        self._divide()

    def _divide(self):
        blockSize = self.blockSize # shortcut
        halfBlock = (blockSize - 1)/2
        rows, columns = map(list, self.dividing.nonzero())
        distanceKernel = _generateDistanceKernel(blockSize)
        connectedKernel = np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=np.uint8)
        probability = np.empty((blockSize, blockSize), dtype=np.float32)

        for row, column in zip(rows, columns):
            biofilm = _getBlock(self.biofilm, row, column, blockSize)

            cv2.filter2D(biofilm, cv.CV_32F, connectedKernel, probability)
            cv2.threshold(probability, 0.1, 1.0, cv2.THRESH_BINARY, probability)
            probability[biofilm] = 0
            probability *= distanceKernel**self.distancePower
            probability *= _getBlock(self.surfaceTension, row, column, 
                                     blockSize, dtype=float)**self.tensionPower

            # now select at random
            flattened = probability.flatten()
            total = flattened.sum()
            if total < 1.0e-12:
                # no viable placements, we'll have precision problems anyways
                continue 
            flattened /= total

            index = np.random.choice(np.arange(len(flattened)), p=flattened)
            localRow, localColumn = np.unravel_index(index, biofilm.shape)

            self.biofilm[row + (localRow - halfBlock),
                        column + (localColumn - halfBlock)] = 1

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

def _generateDistanceKernel(size=7):
    kernel = np.empty((size, size), dtype=float)
    center = (size - 1)/2
    for row in range(size):
        for column in range(size):
            dx = row - center
            dy = column - center
            #kernel[row, column] = np.sqrt(dx**2 + dy**2)
            kernel[row, column] = dx**2 + dy**2

    # avoid a 0 divide
    kernel[center, center] = 1.0
    kernel = 1.0/kernel
    kernel[center, center] = 0.0

    #return kernel/kernel.sum().sum()
    return kernel # we don't need to normalize here, we'll do it later

def _getBlock(matrix, row, column, blockSize, dtype=np.uint8):
    halfBlock = (blockSize - 1)/2
    left = max(0, column - halfBlock)
    right = min(matrix.shape[1] - 1, column + halfBlock + 1)
    top = max(0, row - halfBlock)
    bottom = min(matrix.shape[0] - 1, row + halfBlock + 1)

    block = np.zeros((blockSize, blockSize), dtype=dtype)
    block[halfBlock - (row - top):halfBlock + (bottom - row),
          halfBlock - (column - left):halfBlock + (right - column)] \
         = matrix[top:bottom, left:right]
    return block
