from SPC.UIO.ml_algorithms.LearningAlgorithm import LearningAlgorithm
from SPC.UIO.ml_models.RLNNModel_CorrectSequence import RLNNModel_CorrectSequence
from SPC.UIO.FilterEvaluator import FilterEvaluator
from SPC.UIO.UIO import UIO
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from SPC.UIO.ModelLogger import ModelLogger
import time
from datetime import datetime

class CheckSpecificAlgorithm(LearningAlgorithm):
    def __init__(self, model_logger:ModelLogger):
        super().__init__(model_logger)
        self.saveable_model = True
    
    def train(self, iterations, model_save_path="", model_save_time=0):
        self.model : RLNNModel_CorrectSequence
        self.model = self.model_logger.get_model_structure()


        
        filter = self.FE.getTestMatrix()
        score = self.FE.evaluate(filter)
        
        residuals = self.FE.evaluate(filter, verbose=False, return_residuals=True)
        res_score = np.sum(np.abs(residuals))

        # get 5 wrong uios
        encodings = self.UIO_preparer.getAllUIOEncodings()
        some_wrong_uios = []
        for uiodid, res in enumerate(residuals):
            if res != 0:
                some_wrong_uios.append((encodings[uiodid], res))
                if len(some_wrong_uios) == 5:
                    break
        #print("matrix:", filter)
        print("filter: ", self.FE.convertConditionMatrixToText(filter))
        print("score: ", score)
        print("residual score: ", res_score)
        print("res: ", residuals)

        self.updateModelLogger(res_score, res_score, residuals, filter, None, 0, some_wrong_uios)