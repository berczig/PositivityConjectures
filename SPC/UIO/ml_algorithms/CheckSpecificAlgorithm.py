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

        starts = ["subescher uv start", "subescher vw start", "subescher uw start", "subescher uz start","subescher vz start", "subescher wz start",
              "subescher uv_w start", "subescher uw_v start", "subescher vw_u start","subescher uv_z start", "subescher uz_v start", "subescher vz_u start",
              "subescher uw_z start", "subescher uz_w start", "subescher wz_u start", "subescher vw_z start", "subescher vz_w start","subescher wz_v start",
              "subescher uvw_z start","subescher uvz_w start", "subescher uwz_v start", "subescher vwz_u start","subescher uv_wz start",
              "subescher uw_vz start","subescher vw_uz start"]
        starts = ["subescher wz start"] 
        
        # LESS:  subescher wz start subescher uv_z start subescher wz_u start subescher vz_w start
        # GREATER:  'subescher uz_v start' 'subescher uv_w start' 'subescher uv start'
        # res score 4185 subescher vw_u start' 

        for start in starts:
            # modifications to filter
            new_item = (("0", start), UIO.LESS)
            filter = self.FE.getTestMatrix(new_item) 





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
            print()

            self.updateModelLogger(self.FE.evaluate(filter), res_score, residuals, filter, None, 0, some_wrong_uios)