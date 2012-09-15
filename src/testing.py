
import model
import numpy as np
import cv2
from matplotlib import pyplot as plt

def lightPenetrationVsMediaConcentration(valueFunc, 
                                         valueName, 
                                         numSteps=100,
                                         mediaPenetrationDepth=4,
                                         divisionConstant=0.5,
                                         initialCellSpacing=24):
    light = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    media = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    lightMesh, mediaMesh = np.meshgrid(light, media)
    valueMesh = np.empty_like(lightMesh, dtype=float)

    for column, lightPenetration in enumerate(light):
        for row, mediaConcentration in enumerate(media):
            print "%s/%s" % (1 + row + column*len(media), valueMesh.size)

            m = model.CellModel2D(numCells=(128, 128),
                                  lightPenetrationDepth=lightPenetration,
                                  mediaConcentration=mediaConcentration,
                                  mediaPenetrationDepth=mediaPenetrationDepth,
                                  divisionConstant=divisionConstant)
            m.placeCellsRegularly(initialCellSpacing)
            for _ in range(numSteps): m.step()

            value = valueFunc(m)
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

def movies():
    m = model.CellModel2D(numCells=(256, 256),
                          lightPenetrationDepth=32,
                          mediaConcentration=0.5,
                          mediaPenetrationDepth=3,
                          divisionConstant=0.5)
    m.placeRandomCellsAtBottom(0.05)

    def makeWriter(name):
        return cv2.VideoWriter(name, 
                               fps=20,
                               fourcc=cv2.cv.CV_FOURCC(*"PIM1"),
                               frameSize=tuple(m.numCells), 
                               isColor=False)

    biofilm = makeWriter("biofilm.avi")
    probs = makeWriter("probabilities.avi")
    media = makeWriter("media.avi")
    light = makeWriter("light.avi")

    for t in range(400):
        print t
        m.step()
        print "Ratio: ", m.perimeterToAreaRatio

        biofilm.write(m.biofilm.astype(np.uint8)*255)
        probs.write((m.divisionProbability*255/m.divisionProbability.max()).astype(np.uint8))
        media.write((m.media*255/m.media.max()).astype(np.uint8))
        light.write((m.light*255/m.light.max()).astype(np.uint8))

if __name__ == '__main__':
    lightPenetrationVsMediaConcentration(lambda m: m.perimeterToAreaRatio,
                                         "Perimeter to Area Ratio")
