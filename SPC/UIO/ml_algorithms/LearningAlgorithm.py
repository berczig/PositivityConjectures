from SPC.UIO.ModelLogger import ModelLogger
from SPC.UIO.GlobalUIODataPreparer import GlobalUIODataPreparer

class LearningAlgorithm:

    def __init__(self, model_logger:ModelLogger):
        self.model_logger = model_logger

    def setTrainingData(self,X:dict,y:list):
        self.trainingdata_input = X
        self.trainingdata_output = y

    def train(self, iterations, model_save_path="", model_save_time=0):
        pass

    def predict(self, input):
        pass

    def UIO_preparer(self, GLobalUIODataPreparer:GlobalUIODataPreparer):
        self.UIO_preparer = GLobalUIODataPreparer
        pass

    # Changes from previous version before restructure:
    # DataSaver class -> ModelLogger class
    # ConditionEvaluator class ->   { FilterEvaluator class (evaluation features)
    #                               { GlobalUIODataPreparere class (collecting cores)