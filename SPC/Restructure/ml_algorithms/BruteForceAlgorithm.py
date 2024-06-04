from SPC.Restructure.ml_algorithms.LearningAlgorithm import LearningAlgorithm
from SPC.Restructure.ml_models.RLNNModel_CorrectSequence import RLNNModel_CorrectSequence
from SPC.Restructure.FilterEvaluator import FilterEvaluator
from SPC.Restructure.UIO import UIO
import itertools
import numpy as np
import math
import time
from datetime import datetime

class BruteForceAlgorithm(LearningAlgorithm):
    def train(self, iterations, model_save_path="", model_save_time=0):
        self.model : RLNNModel_CorrectSequence
        self.model = self.model_logger.get_model()

        self.FE = FilterEvaluator(self.trainingdata_input, self.trainingdata_output, FilterEvaluator.DEFAULT_IGNORE_VALUE, self.model.CORE_LENGTH, self.model_logger)

        total_filters = self.getTotalNumberOfFilters(self.model.ALPHABET_SIZE, self.model.ROWS_IN_CONDITIONMATRIX, self.model.COLUMNS_IN_CONDITIONMATRIX, self.model.MAX_EXPECTED_EDGES)
        print("will generate {} matrices".format(total_filters))
        MatrixGenerator = self.generate_matrices(self.model.ROWS_IN_CONDITIONMATRIX, self.model.COLUMNS_IN_CONDITIONMATRIX, self.model.MAX_EXPECTED_EDGES, 
                                                 [UIO.LESS, UIO.EQUAL, UIO.GREATER], FilterEvaluator.DEFAULT_IGNORE_VALUE)
        bestscore = -FilterEvaluator.INF
        bestfilter = None
        starttime = time.time()
        for i, filter in enumerate(MatrixGenerator, start=1):
            score = self.FE.evaluate(filter)
            #print(filter, score)

            if score > bestscore or bestfilter is None:
                bestscore = score
                bestfilter = filter
                print("bestfilter:", self.FE.convertConditionMatrixToText(bestfilter))
            if i % 10000 == 0:
                print("{}/{}. bestscore: {}".format(i, total_filters, bestscore))
                current_work_time = time.time()-starttime
                processing_speed = i/current_work_time
                ETC = time.time() + (total_filters-i)/ processing_speed
                print("ETC: {}".format(datetime.fromtimestamp(ETC)))
        print("bestfilter:", self.FE.convertConditionMatrixToText(bestfilter))
        print("bestscore:", bestscore)

        # [[ -1  -1  -1  -1  -1  -1  -1 101 101  -1]] -1506.0

        # [[ -1  -1  -1  -1  -1  -1  -1  -1  -1  -1]
 # [ -1  -1  -1  -1  -1  -1  -1 101 101  -1]] -5670.0
  
    def getTotalNumberOfFilters(self, ALPHABET_SIZE, ROWS_IN_CONDITIONMATRIX, EDGES, MAX_EXPECTED_EDGES):
        count = 0
        num_entries = ROWS_IN_CONDITIONMATRIX*EDGES
        for num_edges in range(MAX_EXPECTED_EDGES+1):
            count += math.comb(num_entries, num_edges) * (ALPHABET_SIZE-1)**(num_edges)
        # if MAX_EXPECTED_EDGES == num_entries then ALPHABET_SIZE**(ROWS_IN_CONDITIONMATRIX*EDGES) will be returned
        return count

    def generate_matrices(self, rows, cols, max_non_zero, non_values, zero_value):
        """
        Generates matrices with the given constraints.

        :param rows: Number of rows in the matrix
        :param cols: Number of columns in the matrix
        :param max_non_zero: Maximum number of non-zero entries allowed
        :param values: List of possible values for entries in the matrix
        :return: Generator of matrices
        """
        # Generate all possible positions for non-zero entries
        all_positions = [(i, j) for i in range(rows) for j in range(cols)]

        for non_zero in range(max_non_zero+1):
            # Generate combinations of positions where non-zero entries will be placed
            for non_zero_positions in itertools.combinations(all_positions, non_zero):
                # Generate all possible values for the chosen positions
                for values_combo in itertools.product(non_values, repeat=non_zero):
                    matrix = np.empty((rows, cols), dtype=int)
                    matrix.fill(zero_value)
                    for pos, val in zip(non_zero_positions, values_combo):
                        matrix[pos] = val
                    yield matrix
