
import model
import numpy as np
import cv2
from matplotlib import pyplot as plt

def lightPenetrationVsMediaConcentration(valueFunc, 
                                         valueName, 
                                         numSteps=100,
                                         numSamples=3,
                                         mediaPenetrationDepth=4,
                                         divisionConstant=0.5,
                                         initialCellSpacing=24):
    light = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    media = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    lightMesh, mediaMesh = np.meshgrid(light, media)
    valueMesh = np.empty_like(lightMesh, dtype=float)

    for column, lightPenetration in enumerate(light):
        for row, mediaConcentration in enumerate(media):
            print "%s/%s" % (1 + row + column*len(media), valueMesh.size)

            values = []
            for _ in range(numSamples):
                m = model.CellModel2D(numCells=(128, 128),
                                      lightPenetrationDepth=lightPenetration,
                                      mediaConcentration=mediaConcentration,
                                      mediaPenetrationDepth=mediaPenetrationDepth,
                                      divisionConstant=divisionConstant)
                m.placeCellsRegularly(initialCellSpacing)
                for _ in range(numSteps): m.step()
                values.append(valueFunc(m))

            value = np.mean(values)
            print "Light=%s, Media=%s, %s=%s" % \
                  (lightPenetration, mediaConcentration, valueName, value)
            valueMesh[row, column] = value

    plt.figure()
    plt.pcolor(lightMesh, mediaMesh, valueMesh)
    plt.loglog()
    plt.xlim(light.min(), light.max())
    plt.ylim(media.min(), media.max())
    plt.xlabel('Light Penetration Depth')
    plt.ylabel('Media Concentration')
    plt.title(valueName)
    plt.show()

def parameterSweep():
    def makeWriter(name):
        return cv2.VideoWriter(name, 
                               fps=30,
                               fourcc=cv2.cv.CV_FOURCC(*"PIM1"),
                               frameSize=(256, 256), 
                               isColor=False)

    i = 0
    for light in [16, 32]:
        for media in [1.0, 2.0]:
            for boundary in [4, 8, 12]:
                for penetration in [4, 8, 12]:
                    for distancePower in [0.5, 1.0, 2.0]:
                        for tensionPower in [0.5, 1.0, 2.0]:
                            i += 1
                            print i

                            m = model.ProbabilisticModel2D(numCells=(256, 256),
                                                           lightPenetrationDepth=light,
                                                           mediaConcentration=media,
                                                           boundaryLayerThickness=boundary,
                                                           mediaPenetrationDepth=penetration,
                                                           divisionConstant=1.0,
                                                           blockSize=5,
                                                           distancePower=distancePower,
                                                           tensionPower=tensionPower)
                            m.placeCellsRegularly(64)

                            name = "-".join(map(str, [light, media, boundary, penetration, distancePower, tensionPower]))
                            print name

                            biofilmW = makeWriter("../movies/%s-biofilm.avi" % name)
                            probsW = makeWriter("../movies/%s-probabilities.avi" % name)
                            mediaW = makeWriter("../movies/%s-media.avi" % name)
                            lightW = makeWriter("../movies/%s-light.avi" % name)

                            for t in range(1000):
                                m.step()
                                if t % 5: continue

                                biofilmW.write(m.biofilm.astype(np.uint8)*255)
                                probsW.write((m.divisionProbability*255/m.divisionProbability.max()).astype(np.uint8))
                                mediaW.write((m.media*255/m.media.max()).astype(np.uint8))
                                lightW.write((m.light*255/m.light.max()).astype(np.uint8))
    print "done!"


def movies():
    m = model.ProbabilisticModel2D(numCells=(256, 256),
                                   lightPenetrationDepth=32,
                                   mediaConcentration=1.0,
                                   boundaryLayerThickness=12,
                                   mediaPenetrationDepth=8,
                                   divisionConstant=2.0,
                                   blockSize=5,
                                   distancePower=0.5,
                                   tensionPower=2.5)
    #m.placeRandomCellsAtBottom(0.025)
    m.placeCellsRegularly(64)

    def makeWriter(name):
        return cv2.VideoWriter(name, 
                               fps=30,
                               fourcc=cv2.cv.CV_FOURCC(*"PIM1"),
                               frameSize=tuple(m.numCells), 
                               isColor=False)

    biofilm = makeWriter("../movies/biofilm.avi")
    probs = makeWriter("../movies/probabilities.avi")
    media = makeWriter("../movies/media.avi")
    light = makeWriter("../movies/light.avi")

    for t in range(1000):
        print t
        m.step()
        if t % 5: continue

        biofilm.write(m.biofilm.astype(np.uint8)*255)
        probs.write((m.divisionProbability*255/m.divisionProbability.max()).astype(np.uint8))
        media.write((m.media*255/m.media.max()).astype(np.uint8))
        light.write((m.light*255/m.light.max()).astype(np.uint8))

    print m.meanHeight
    freq, sp = m.rowFFT(round(m.meanHeight))
    plt.plot(freq, np.abs(sp))
    plt.show()

if __name__ == '__main__':
    #lightPenetrationVsMediaConcentration(lambda m: m.perimeterToAreaRatio,
    #                                     "Perimeter to Area Ratio")
    #movies()
    parameterSweep()
