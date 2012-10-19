
import numpy as np
import cv2
import cv
#from matplotlib import pyplot as plt
import random as rdm
import collections
from itertools import product
import pickle


ALIVE = 1
EMPTY = 0

class Parameters(object):

    def __init__(self, specified):
        self.__dict__ = specified

    def addDefaults(self, **defaults):
        for key, value in defaults.iteritems():
            if key not in self.__dict__:
                self.__dict__[key] = value

    def __repr__(self):
        return "Parameters(%s)" % self.__dict__

    def copy(self):
        return Parameters(self.__dict__.copy())

def calculateMaxHeight(biofilm):
    for row in reversed(range(biofilm.shape[0])):
        if biofilm[row, :].any():
            return row
    return 0

def calculateHeights(biofilm):
    heights = np.zeros(biofilm.shape[1], dtype=int)
    for row in reversed(range(biofilm.shape[0])):
        heights[np.logical_and((heights == 0), biofilm[row, :])] = row
    return heights

def stopOnTime(finalTime):
    return lambda time, _: time >= finalTime

def stopOnMass(maxMass):
    return lambda _, model: np.sum(model.biofilm) >= maxMass

def stopOnHeight(maxHeight):
    return lambda _, model: calculateMaxHeight(model.biofilm) >= maxHeight


class BiofilmModel(object):

    def __init__(self, **params):
        self.params = Parameters(params)
        self._addDefaults()

        self.numCells = self.params.numCells
        self.biofilm = self._zeros(np.uint8)

    def _addDefaults(self):
        self.params.addDefaults(numCells=(256, 256))

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

    def step(self):
        raise NotImplemented()

    def run(self, stopCondition=stopOnTime(100), 
                  maxIterations=10000,
                  videoOutput=None,
                  showProgress=True):
        result = ModelResult(self, videoOutput=videoOutput)
        result.record(0)

        for time in range(1, maxIterations):
            if showProgress and time % 10 == 0:
                print "Step #%i" % time

            if stopCondition(time, self): break
            self.step()
            result.record(time)

        result.finalize()
        return result


class ModelResult(object):

    def __init__(self, model, videoOutput=None):
        self.model = model
        self._collectBasicInfo()

        self.times = []
        self.areas = []

        if videoOutput is not None:
            self.video = cv2.VideoWriter(videoOutput, 
                                        fps=30,
                                        fourcc=cv2.cv.CV_FOURCC(*"PIM1"),
                                        frameSize=self.numCells, 
                                        isColor=False)
        else:
            self.video = None

    def record(self, time):
        self.times.append(time)
        self._calculateArea()

        if self.video is not None:
            self.video.write(self._biofilm.astype(np.uint8)*255)

    def finalize(self):
        self._calculateContours()
        self._calculatePerimeter()
        self._calculateHeights()
        self._calculateMaxHeight()
        self._calculateMeanHeight()
        self._calculateRowFFTs()
        self._calculateCoverage()
        self._calculateConvexityDefects()
        self._calculateXCorrelations()

        self.finalBiofilm = np.copy(self._biofilm)
        del self.model
        del self.video

    def save(self, output):
        with open(output, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _collectBasicInfo(self):
        self.numCells = self.model.numCells
        self.params = self.model.params

    def _calculateArea(self):
        self.areas.append(np.sum(self._biofilm))

    def _calculateContours(self):
        self.contours, _ = cv2.findContours(np.copy(self._biofilm), 
                                            cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)

    def _calculatePerimeter(self):
        self.perimeter = sum(cv2.arcLength(c, True) for c in self.contours)

    def _calculateHeights(self):
        self.heights = calculateHeights(self._biofilm)

    def _calculateMaxHeight(self, top=0.05):
        heights = np.sort(self.heights)
        self.maxHeight = np.mean(heights[-np.ceil(top*len(heights)):])

    def _calculateMeanHeight(self):
        self.meanHeight = np.mean(self.heights)

    def _calculateRowFFTs(self):
        freqs = np.fft.fftfreq(self.model.numColumns)
        rows = np.vstack(np.fft.fft(self._biofilm[row, :])
                         for row in range(self.model.numRows))
        self.rowFFTs = freqs, rows

    def _calculateCoverage(self):
        self.coverages = self._biofilm.sum(axis=1)/float(self.model.numColumns)

    def _calculateConvexityDefects(self):
        self.convexityDefects = []
        for contour in self.contours:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None: continue

            # defects is an Nx1x4 matrix
            for row in range(defects.shape[0]):
                depth = defects[row, 0, 3]/256.0
                self.convexityDefects.append(depth)

    def _calculateXCorrelations(self):
        distances = range(1, self.model.numColumns/2)
        self.xCorrelations = []

        for row in range(self.model.numRows):
            if not self._biofilm[row, :].any(): 
                break

            found = np.zeros(len(distances), dtype=int)
            count = np.zeros_like(found)

            for col in range(self.model.numRows):
                cell = self._biofilm[row, col]

                for i, distance in enumerate(distances):
                    for direction in [-1, 1]:
                        offset = col + distance*direction
                        if offset < 0 or offset >= self.model.numColumns:
                            continue

                        count[i] += 1
                        if self._biofilm[row, offset] == cell:
                            found[i] += 1

            probability = found.astype(float)/count
            self.xCorrelations.append((distances, probability))


    @property # shortcut
    def _biofilm(self): return self.model.biofilm

def loadResult(fileName):
    with open(fileName, "rb") as f:
        return pickle.load(f)


class TestPlan(object):

    def __init__(self, processNumber=0, poolSize=1):
        self.paramValues = dict()
        self.processNumber = processNumber
        self.poolSize = poolSize

    def add(self, param, *values):
        self.paramValues.setdefault(param, []).extend(values)
        return self

    @property
    def numTests(self):
        return np.product([len(v) for v in self.paramValues.itervalues()])

    @property
    def paramSets(self):
        return map(dict, product(*([(name, value) for value in values]
                                   for name, values 
                                   in self.paramValues.iteritems())))

    def run(self, baseFilePath, createModel, *runArgs, **runKwargs):
        for i, params in enumerate(self.paramSets):
            if i % self.poolSize != self.processNumber:
                continue

            print "Starting test %i!" % i

            model = createModel(**params)
            results = model.run(*runArgs, **runKwargs)
            results.save("%s-%i.results" % (baseFilePath, i))
            
            print "Done with test %i!" % i
    

class CAModel(BiofilmModel):

    def __init__(self, **params):
        BiofilmModel.__init__(self, **params)

        self.boundaryLayer = self._zeros(np.uint8)
        self.media = self._zeros(float)
        self.light = self._zeros(float)
        self.divisionProbability = self._zeros(float)
        self.dividing = self._zeros(bool)
        self.surfaceTension = self._zeros(float)

        if self.params.initialCellSpacing:
            self.placeCellsRegularly(self.params.initialCellSpacing)

    def _addDefaults(self):
        BiofilmModel._addDefaults(self)
        self.params.addDefaults(boundaryLayerThickness=8,
                                lightPenetrationDepth=16,
                                mediaConcentration=1.0,
                                mediaPenetrationDepth=8,
                                divisionConstant=1.0,
                                initialCellSpacing=64)

    def placeRandomCells(self, probability=0.2):
        for column in range(self.numColumns):
            if rdm.random() < probability:
                self.biofilm[0, column] = ALIVE

    def placeCellsRegularly(self, spacing=8):
        start = int(spacing/2)
        end = self.numColumns - int(spacing/2)
        for column in range(start, end, spacing):
            self.biofilm[0, column] = ALIVE

    def _calculateMedia(self):
        kernel = _makeCircularKernel(self.params.boundaryLayerThickness)
        cv2.filter2D(self.biofilm, -1, kernel, self.boundaryLayer)

        np.logical_not(self.boundaryLayer, out=self.media)
        self.media *= self.params.mediaConcentration

        cv2.GaussianBlur(self.media, (0, 0), self.params.mediaPenetrationDepth,
                         dst=self.media)

    def _calculateLight(self):
        if self.params.lightPenetrationDepth is not None:
            np.cumsum(self.biofilm, axis=0, out=self.light)
            self.light /= -self.params.lightPenetrationDepth
            np.exp(self.light, out=self.light)
        else:
            self.light = np.ones(self.numCells, dtype=float)

    def _calculateSurfaceTension(self, centerFactor=0):
        k = centerFactor
        tensionKernel = np.array([[1, 2, 1],
                                  [2, k, 2],
                                  [1, 2, 1]], dtype=np.uint8)
        localSum = cv2.filter2D(self.biofilm, -1, tensionKernel)
        self.surfaceTension = localSum/np.float(tensionKernel.sum().sum())

    def _calculateDivisionProbability(self):
        self.divisionProbability = self.params.divisionConstant*\
                                   self.media*self.light
        self.divisionProbability[np.logical_not(self.biofilm)] = 0

    def _calculateDividingCells(self):
        self.dividing = np.random.ranf(self.numCells) <= \
                        self.divisionProbability

    def _calculateThroughDivisionProbability(self):
        self._calculateMedia()
        self._calculateLight()
        self._calculateDivisionProbability()

class MinimumDistanceModel(CAModel):

    def __init__(self, **params):
        CAModel2D.__init__(self, **params)
        self.distances = self._zeros(int) # distance to nearest empty cell

    def _addDefaults(self):
        CAModel._addDefaults(self)
        self.params.addDefaults(surfaceTensionFactor=10)

    def step(self):
        self._calculateThroughDivisionProbability()
        self._calculateSurfaceTension(self.params.surfaceTensionFactor)
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

class ProbabilisticModel(CAModel):

    def __init__(self, **params):
        CAModel.__init__(self, **params)

        self.distancePower = self.params.distancePower
        self.tensionPower = self.params.tensionPower
        self.blockSize = self.params.blockSize

    def _addDefaults(self):
        CAModel._addDefaults(self)
        self.params.addDefaults(distancePower=0.5,
                                tensionPower=2.5,
                                blockSize=5)

    def step(self):
        self._calculateThroughDivisionProbability()
        self._calculateSurfaceTension()
        self._calculateDividingCells()
        self._divide()

    def _divide(self):
        blockSize = self.params.blockSize # shortcut
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
            probability *= distanceKernel**self.params.distancePower
            probability *= _getBlock(self.surfaceTension, row, column, 
                                     blockSize, dtype=float)\
                           **self.params.tensionPower

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

class UniformModel(BiofilmModel):

    def __init__(self, **params):
        BiofilmModel.__init__(self, **params)

        self.heights = np.zeros(self.numColumns, dtype=int)
        self.step()

    def _addDefaults(self):
        BiofilmModel._addDefaults(self)
        self.params.addDefaults(meanGrowth=1.0, 
                                lightPenetrationDepth=32,
                                smoothSigma=0.5)

    def step(self):
        light = np.exp(self.heights/-self.params.lightPenetrationDepth)

        rawGrowth = [rdm.expovariate(1.0/self.params.meanGrowth)
                     for _ in range(self.numColumns)]
        growth = np.round(light*rawGrowth).astype(int)
        self.heights += growth

        self.heights = _gaussSmooth(self.heights, self.params.smoothSigma, 
                                    (0, self.numRows - 1))

        self.biofilm.fill(0)
        for column, height in enumerate(self.heights):
            self.biofilm[0:height, column] = 1

class ColumnModel(BiofilmModel):

    def __init__(self, **params):
        BiofilmModel.__init__(self, **params)

        self.heights = np.zeros(self.numColumns, dtype=int)
        self.columnHeight = 0
        self.step()

    def _addDefaults(self):
        BiofilmModel._addDefaults(self)
        self.params.addDefaults(meanGrowth=3.0, 
                                lightPenetrationDepth=32,
                                columnWidth=16,
                                columnSpacing=12,
                                floorRatio=0.1,
                                smoothSigma=1.0,
                                noiseSigma=3.0)

    def step(self):
        light = np.exp(self.columnHeight/-self.params.lightPenetrationDepth)

        rawGrowth = rdm.expovariate(1.0/self.params.meanGrowth)
        self.columnHeight += np.round(light*rawGrowth).astype(int)
        floorHeight = round(self.columnHeight*self.params.floorRatio)

        wavelength = self.params.columnSpacing + self.params.columnWidth
        end = self.numColumns - 1
        for offset in range(0, end + 1, wavelength):
            columnStart = min(end, offset + self.params.columnSpacing)
            self.heights[offset:columnStart] = floorHeight

            if columnStart == end:
                break

            columnEnd = min(end, columnStart + self.params.columnWidth)
            height = round(self.columnHeight + 
                           rdm.gauss(0, self.params.noiseSigma))
            self.heights[columnStart:columnEnd] = height

        self.heights = _gaussSmooth(self.heights, self.params.smoothSigma, 
                                    (0, self.numRows - 1))

        self.biofilm.fill(0)
        for column, height in enumerate(self.heights):
            self.biofilm[0:height, column] = 1

def _makeCircularKernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

def _makeGaussKernel(sigma, minN=8):
    n = max(minN, round(4*sigma))
    mu = n/2.0
    kernel = np.exp(-(np.arange(1, n) - mu)**2/(2*sigma**2))
    return kernel/kernel.sum()

def _gaussSmooth(values, sigma, clip=None):
    extended = np.concatenate((values[0]*np.ones_like(values),
                               values,
                               values[-1]*np.ones_like(values)))
    kernel = _makeGaussKernel(sigma)
    smoothed = np.convolve(extended, kernel, mode='same')
    values = smoothed[len(values):2*len(values)]

    if clip is not None:
        np.clip(values, clip[0], clip[1], out=values)

    return values


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
