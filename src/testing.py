
import model
import numpy as np
import cv2

if __name__ == '__main__':
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