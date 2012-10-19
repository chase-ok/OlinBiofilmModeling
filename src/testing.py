
import model
import sys

plan = model.TestPlan(processNumber=int(sys.argv[1]),
                      poolSize=4)
plan.add("numCells", (96, 256))
plan.add("boundaryLayerThickness", 8, 12, 16)
plan.add("lightPenetrationDepth", 16, None)
plan.add("mediaConcentration", 1.0)
plan.add("mediaPenetrationDepth", 8, 12, 16)
plan.add("divisionConstant", 0.01, 0.1, 0.5, 1.0, 2.0)
plan.add("initialCellSpacing", 64)
plan.add("distancePower", 0.1, 0.5, 1.0, 1.5, 2.0)
plan.add("tensionPower", 0.1, 0.5, 1.0, 1.5, 2.0)
plan.add("blockSize", 5)

plan.run("../results/probabalistic", 
         model.ProbabilisticModel,
         stopCondition=model.stopOnHeight(90),
         maxIterations=1000,
         showProgress=True)

print "PROCESS DONE!"
