from SPC.UIO.ml_algorithms.LearningAlgorithm import LearningAlgorithm
from SPC.UIO.ml_models.RLNNModel_CorrectSequence import RLNNModel_CorrectSequence
from SPC.UIO.FilterEvaluator import FilterEvaluator
from SPC.UIO.UIO import UIO
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from datetime import datetime

class BruteForceAlgorithm(LearningAlgorithm):
    def train(self, iterations, model_save_path="", model_save_time=0):
        self.model : RLNNModel_CorrectSequence
        self.model = self.model_logger.get_model_structure()

        self.FE = FilterEvaluator(self.trainingdata_input, self.trainingdata_output, FilterEvaluator.DEFAULT_IGNORE_VALUE, self.model_logger)

        total_filters = self.getTotalNumberOfFilters(self.model.ALPHABET_SIZE, self.model.ROWS_IN_CONDITIONMATRIX, self.model.COLUMNS_IN_CONDITIONMATRIX, self.model.MAX_EXPECTED_EDGES)
        print("will generate {} matrices".format(total_filters))
        MatrixGenerator = self.generate_matrices(self.model.ROWS_IN_CONDITIONMATRIX, self.model.COLUMNS_IN_CONDITIONMATRIX, self.model.MAX_EXPECTED_EDGES, 
                                                 [UIO.LESS, UIO.EQUAL, UIO.GREATER], FilterEvaluator.DEFAULT_IGNORE_VALUE)
        bestscore = -FilterEvaluator.INF
        bestfilter = None
        starttime = time.time()
        my_filter = np.ones((self.model.ROWS_IN_CONDITIONMATRIX, self.model.COLUMNS_IN_CONDITIONMATRIX))*self.FE.DEFAULT_IGNORE_VALUE
        # ["0", "subescher start interval", "subescher end interval", "1.insert", "2. insert", "n-1", "n+k-1"]
        #my_filter[0][0] = UIO.LESS
        #my_filter[0][11] = UIO.LESS
        #my_filter[1][7] = UIO.LESS
        #my_filter[1][9] = UIO.EQUAL

        #my_filter[0][0] = UIO.LESS
        #my_filter[0][11] = UIO.LESS
        #my_filter[1][7] = UIO.LESS
        #my_filter[1][9] = UIO.EQUAL

        #my_filter[0][0] = UIO.LESS
        #my_filter[0][11] = UIO.LESS
        #my_filter[0][16] = UIO.LESS
        #my_filter[1][16] = UIO.GREATER
        #my_filter[1][19] = UIO.GREATER
        #for i, filter in [(1, my_filter)]: # enumerate(MatrixGenerator, start=1):
        for i, filter in enumerate(MatrixGenerator, start=1):
            score = self.FE.evaluate(filter)

            if score > bestscore or bestfilter is None:
                bestscore = score
                bestfilter = filter
                print("bestfilter({}):\n".format(bestscore), self.FE.convertConditionMatrixToText(bestfilter))
            if i % 10000 == 0:
                print("{}/{}. bestscore: {}".format(i, total_filters, bestscore))
                current_work_time = time.time()-starttime+0.0001
                processing_speed = i/current_work_time
                ETC = time.time() + (total_filters-i)/ processing_speed
                print("ETC: {}".format(datetime.fromtimestamp(ETC)))
        print("bestfilter:", self.FE.convertConditionMatrixToText(bestfilter))
        print("bestscore:", bestscore)
        print()

        res = self.FE.evaluate(bestfilter, verbose=False, return_residuals=True)
        perfect_indices = np.abs(res) < 0.01
        perfect_predict = sum(perfect_indices)
        print("perfect predict: {}/{} ~ {:.2f}%".format(perfect_predict, len(res), 100*perfect_predict/len(res)))
        res_notp = res[~perfect_indices]
        print("res mean:", np.mean(res))
        print("res sd:", np.var(res)**0.5)
        print("non-zero res mean:", np.mean(res_notp))
        print("non-zero res sd:", np.var(res_notp)**0.5)
        tcnz = self.FE.true_coefficients[self.FE.true_coefficients!=0]
        tcm = np.mean(self.FE.true_coefficients)
        print("true coeff mean:", tcm)
        tcnzm = np.mean(tcnz)
        print("true coeff mean:", tcnzm)
        print("residuals in 10% mean range: {:.2f}".format(100*sum(res <= (0.1*tcnzm))/len(res)))

        print("non zero coefficients: {:.2f}%".format( 100*sum(self.FE.true_coefficients!=0)/len(self.FE.true_coefficients)  ))
        #plt.plot(range(len(res_notp)), res_notp+self.FE.true_coefficients[~perfect_indices], range(len(res_notp)), self.FE.true_coefficients[~perfect_indices])

        plt.plot(range(len(res)), res+self.FE.true_coefficients, range(len(res)), self.FE.true_coefficients)
        plt.title("correct coeffs vs predictions")
        plt.show()


        fig, ax = plt.subplots()
        ax.plot(range(len(res_notp)), res_notp)
        ax.hlines(y=0.1*tcnzm, xmin=0.0, xmax=len(res_notp), color='r')
        plt.title("non-zero residuals")
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(res[self.FE.true_coefficients == 0])
        plt.title("residuals of the zero coefficients")
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(res[self.FE.true_coefficients != 0])
        plt.title("residuals of the non-zero coefficients")
        plt.show()

    def ressi(self, true, res):
        pass
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
