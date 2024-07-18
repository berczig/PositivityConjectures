import importlib
from SPC.Restructure.GlobalUIODataPreparer import GlobalUIODataPreparer
from SPC.Restructure.ModelLogger import ModelLogger
from SPC.Restructure.ml_algorithms.LearningAlgorithm import LearningAlgorithm
from SPC.Restructure.ml_algorithms.RLAlgorithm import RLAlgorithm
from SPC.Restructure.ml_models.MLModel import MLModel
from SPC.Restructure.ml_models.RLNNModel_CorrectSequence import RLNNModel_CorrectSequence


partition = (3,2,1)
core_generator_type = "EscherCoreGeneratorTripple" # EscherCoreGeneratorBasic  EscherCoreGeneratorTripple

uio_length = sum(partition)
# core_data_type

# 1) get Training Data
Preparer = GlobalUIODataPreparer(uio_length)
#Preparer.computeTrainingData(partition, core_generator_type)
Preparer.loadTrainingData("SPC/Saves,Tests/Trainingdata/partition_3_2_1_symmectric_core_9_7_2024.bin")

counts = Preparer.getInputdataAsCountsMatrix()

def saveToFile(counts, sums, path):
    with open(path, "w+") as file:
        T = [",".join([str(sums[i])] + [str(x) for x in counts[i]]) for i in range(len(sums))]
        file.write("\n".join(T))

saveToFile(counts, Preparer.getTrainingData()[1], "SPC/Saves,Tests/subsetsum/subsetsum3_2_1_symmetric.txt")
