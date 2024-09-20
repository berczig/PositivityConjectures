from SPC.UIO.ModelLogger import ModelLogger
from SPC.UIO.FilterEvaluator import FilterEvaluator
from SPC.UIO.GlobalUIODataPreparer import GlobalUIODataPreparer
import time
import numpy as np

class LearningAlgorithm:

    def __init__(self, model_logger:ModelLogger):
        self.model_logger = model_logger
        self.saveable_model = False
        self.UIO_preparer:GlobalUIODataPreparer
 
    def setTrainingData(self,X:dict,y:list):
        self.trainingdata_input = X
        self.trainingdata_output = y
        self.FE = FilterEvaluator(self.trainingdata_input, self.trainingdata_output, FilterEvaluator.DEFAULT_IGNORE_VALUE, self.model_logger)

    def train(self, iterations, model_save_path="", model_save_time=0):
        pass

    def predict(self, input):
        pass

    def setUIO_preparer(self, GLobalUIODataPreparer:GlobalUIODataPreparer):
        self.UIO_preparer = GLobalUIODataPreparer

    def updateModelLogger(self, current_bestscore, res_score, residuals, bestmatrix, super_rewards, tic0, some_wrong_uios):
        V, E = self.FE.convertConditionMatrixTo_VerticesAndEdges(bestmatrix)
        self.model_logger.graph_vertices = V
        self.model_logger.graph_edges = E
        self.model_logger.bestscore_history.append(current_bestscore)
        self.model_logger.residual_score_history.append(res_score)
        self.model_logger.perfect_coef_history.append(np.sum(residuals==0))
        self.model_logger.last_modified = time.time()
        self.model_logger.graphsize_history.append(np.sum(bestmatrix!=self.FE.ignore_edge))
        if super_rewards:
            self.model_logger.meanscore_history.append(np.mean(super_rewards))
        self.model_logger.calculationtime_history.append(time.time()-tic0)
        self.model_logger.residuals = residuals
        self.model_logger.current_bestgraph = self.FE.convertConditionMatrixToText(bestmatrix)
        self.model_logger.current_bestgraph_matrix = bestmatrix
        self.model_logger.some_wrong_uios = some_wrong_uios
        self.model_logger.coeffsum = np.sum(self.FE.true_coefficients)
        

    # Changes from previous version before restructure:
    # DataSaver class -> ModelLogger class
    # ConditionEvaluator class ->   { FilterEvaluator class (evaluation features)
    #                               { GlobalUIODataPreparere class (collecting cores)