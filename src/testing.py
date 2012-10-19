
import model

plan = model.TestPlan()
plan.add("numCells", (128, 256))
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
         stopCondition=model.stopOnHeight(100),
         maxIterations=1500)

print "REALLY DONE!"
