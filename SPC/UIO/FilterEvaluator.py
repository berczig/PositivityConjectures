import numpy as np
from SPC.UIO.UIO import UIO
from SPC.UIO.ModelLogger import ModelLogger
from SPC.UIO.cores.EscherCoreGeneratorBasic import EscherCoreGeneratorBasic
from SPC.UIO.cores.EscherCoreGeneratorTripple import EscherCoreGeneratorTripple
#from SPC.UIO.cores.EscherCoreGenerator2 import EscherCoreGenerator
from SPC.UIO.cores.CorrectSequenceCoreGenerator import CorrectSequenceCoreGenerator

class FilterEvaluator: 

    INF = 9999999999
    DEFAULT_IGNORE_VALUE = -1

    def __init__(self, coreRepresentationsCategorizer:dict, true_coefficients, ignore_edge:int, model_logger:ModelLogger, verbose=True):
        """
        coreRepresentationsCategorizer: {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2}, ...}
        """
        self.coreRepresentationsCategorizer = coreRepresentationsCategorizer # {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2}}
        self.true_coefficients = np.array(true_coefficients)
        self.ignore_edge = ignore_edge
        if verbose:
            print("true coefficients:", true_coefficients)
            print("true coefficients length:", len(true_coefficients))
            print("true coefficients max coef:", max(true_coefficients))
            print(np.sum(true_coefficients))
        self.coreRep2 = {}
        for primerep in coreRepresentationsCategorizer:
            counts = np.zeros(len(true_coefficients))
            uio_counts = coreRepresentationsCategorizer[primerep]
            for uioID in uio_counts:
                counts[uioID] += uio_counts[uioID]
            self.coreRep2[primerep] = counts

        self.core_labels = model_logger.core_generator_class_.getCoreLabels(model_logger.partition)
        model_logger.core_generator_class_.calculate_comp_indices(model_logger.partition)
        self.comp_indices = model_logger.core_generator_class_.comp_indices
        self.edgePenalty = model_logger.edgePenalty

        self.model_logger:ModelLogger
        self.model_logger = model_logger

    def getTestMatrix(self, new_item):
        desc = self.model_logger.core_generator_class_.getTestMatrixDescription(self.model_logger.partition)
        desc[0][new_item[0]] = new_item[1]
        rows = len(desc)
        columns = self.model_logger.core_generator_class_.getCoreRepresentationLength(self.model_logger.partition)
        print(f"getTestMatrix create a {rows}x{columns} matrix")
        matrix = np.empty((rows, columns), dtype=int)
        matrix.fill(FilterEvaluator.DEFAULT_IGNORE_VALUE)

        ordered_cc = self.model_logger.core_generator_class_.getOrderedCoreComparisions(self.model_logger.partition)
        for rowindex, comparisons in enumerate(desc):
            for edgecomp in comparisons:
                colindex = ordered_cc.index(edgecomp)
                matrix[rowindex][colindex] = comparisons[edgecomp]
                print("edge:", edgecomp, comparisons[edgecomp], colindex)
        return matrix

    def coreFitsConditions(self, correp, Conditions): # ANDs conditions in row together

        for rowcondition in Conditions:
            fits = True
            for edgeIndex, edgevalue in rowcondition:
                #print("correp:", len(correp), correp)
                #print("edgeIndex:", edgeIndex)
                if correp[edgeIndex] != edgevalue:
                    fits = False
                    break
            if fits:
                return True
        # can only be here if all rows are ignore
        return False
    

    def evaluate(self, filter, verbose=False, return_residuals=False):
        # for each uio of length l+k, check how many of its cores comply  with 
        # the Condition_matrix and compare that amount with the true coefficient c_{l,k}
        if verbose:
            print("evaluate filter / Condition_matrix:", filter)
        score = 0 # bigger is better, negative

        # prune the Condition_matrix - remove the edges that are of type self.ignore_edge
        Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != self.ignore_edge] 
                    for conditionrow in filter]
        #Conditions = [x for x in Conditions if len(x) != 0] # removes [] from Conditions - a important decision, can mess up the RL for correct sequences! (1 single condition will always count to few correps)
        counted = np.zeros(len(self.true_coefficients)) # the i'th entry is the number of correps associated to the i'th uio that fit the Conditions

        if len(Conditions) == 0: # count everything
            for primeCoreRep in self.coreRep2:
                counted += self.coreRep2[primeCoreRep]
        else:
            for primeCoreRep in self.coreRep2:
                #print("filter:", filter)
                #print("primeCoreRep:", primeCoreRep, self.coreRepresentationsCategorizer[primeCoreRep], self.coreFitsConditions(primeCoreRep, Conditions))
                if primeCoreRep == "GOOD" or (primeCoreRep != "BAD" and self.coreFitsConditions(primeCoreRep, Conditions) == True):
                #if not primeCoreRep or self.coreFitsConditions(primeCoreRep, Conditions) == True:
                    #print("count!")
                    counted += self.coreRep2[primeCoreRep]
                    if verbose:
                        print(primeCoreRep, "Good")
                else:
                    if verbose:
                        print(primeCoreRep, "bad")

        residuals = counted - self.true_coefficients

        if return_residuals:
            return residuals
        
        if (residuals > 0).any():
            return -5000
            
        """if -sum(abs(residuals)) > -10:  
            for id, x in enumerate(residuals):  
                if x > 0:
                    print("UIOID:", id, counted[id], self.true_coefficients[id])"""
        
        from SPC.UIO.ml_algorithms.RLAlgorithm import RLAlgorithm
        num_edges = np.sum(filter != self.DEFAULT_IGNORE_VALUE) 
    
        has_trivial_row = np.any([np.all(row==self.DEFAULT_IGNORE_VALUE) for row in filter])

        return -sum(abs(residuals)) - self.edgePenalty*num_edges + (-1000 if has_trivial_row else 0)
    
    def getCorrectUIOs(self, filter, verbose=False):
        residuals = self.evaluate(filter, verbose, return_residuals=True)
        return np.sum(residuals != 0) 
      
    def evaluate_old(self, filter, verbose=False):
        # for each uio of length l+k, check how many of its cores comply  with 
        # the Condition_matrix and compare that amount with the true coefficient c_{l,k}
        if verbose:
            print("evaluate filter / Condition_matrix:", filter)
        score = 0 # bigger is better, negative

        # Condition_matrix is not so straight to the point when one wants to check the conditions, so let's prune it a bit so it's easier to do the checking
        Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != self.ignore_edge] 
                    for conditionrow in filter]

        counted = np.zeros(len(self.true_coefficients)) # the i'th entry is the number of correps associated to the i'th uio that fit the Conditions
        for primeCoreRep in self.coreRepresentationsCategorizer:
            #print("primeCoreRep:", primeCoreRep, self.coreRepresentationsCategorizer[primeCoreRep], self.coreFitsConditions(primeCoreRep, Conditions))
            if self.coreFitsConditions(primeCoreRep, Conditions) == True:
                dict_ = self.coreRepresentationsCategorizer[primeCoreRep]
                for uioID in dict_:
                    a = dict_[uioID]
                    counted[uioID] += a
        #print()
        difference = counted - np.array(self.true_coefficients)
        for x in difference:
            if x < 0:
                return -self.INF
        return -sum(difference)
    
    def convertCorerepToText(self, corerep):
        for index, (i,j) in enumerate(self.comp_indices):
            edge = corerep[index]
            print( self.core_labels[i], UIO.RELATIONTEXT[edge], self.core_labels[j])
    
    def convertConditionMatrixToText(self, Condition_matrix):
        rows, columns = Condition_matrix.shape
        rowtexts = []
        for row in range(rows):
            rowtext = []
            for index, (i,j) in enumerate(self.comp_indices):
                edge = int(Condition_matrix[row][index])
                if edge != self.ignore_edge:
                    rowtext.append("[{}{}{}]".format(self.core_labels[i],UIO.RELATIONTEXT[edge],self.core_labels[j]))
            if rowtext:
                rowtexts.append(" AND ".join(rowtext))
        return 3*" " + 16*"-" + "\n" + "\nOR\n".join(rowtexts) + "\n"+ 3*" " + 16*"-"
    
    def convertConditionMatrixTo_VerticesAndEdges(self, Condition_matrix) -> tuple:
        # returns the corelabels (the vertices of the graph) and all edges of the connected components in the graph
        V = self.core_labels
        edges_all_rows = []

        rows, columns = Condition_matrix.shape
        for row in range(rows):
            edges = []
            for index, (i,j) in enumerate(self.comp_indices):
                edge = int(Condition_matrix[row][index])
                if edge != self.ignore_edge:
                    edges.append((i,j, edge))
            edges_all_rows.append(edges)

        return V,edges_all_rows
    
    def narrowCoreTypeSelection(self, random_uios):
        # only use a portion of the coreTypes corresponding to n_uios random uios when evaluating a conditionMatrix
        #print("n:", self.n, "C_n:", C_n(self.n))
        #total_uios = C_n(self.n)
        #random_uios = random.sample(range(total_uios), n_uios)

        # copy all the coretypes which appear in at least one of the selected uios
        print("narrowCoreTypeSelection")
        self.activeuios_n = len(random_uios)
        self.activeTrueCoefficients = self.trueCoefficients[random_uios]
        self.activeCoreTypes = {}
        for coretype in self.coreTypes:
            counts = self.coreTypes[coretype]
            for uioID in counts:
                if uioID in random_uios: # is one of the randoms
                    if coretype not in self.activeCoreTypes:
                        self.activeCoreTypes[coretype] = {uioID:counts[uioID]}
                    else:
                        self.activeCoreTypes[coretype][uioID] = counts[uioID]
        print(len(self.coreTypes))
        print("activeCoreTypes:", len(self.activeCoreTypes))

    