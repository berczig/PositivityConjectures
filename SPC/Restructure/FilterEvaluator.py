import numpy as np
from SPC.Restructure.UIO import UIO

class FilterEvaluator: 

    INF = 9999999999
    DEFAULT_IGNORE_VALUE = -1

    def __init__(self, coreRepresentationsCategorizer:dict, true_coefficients, ignore_edge:int, core_length:int):
        """
        coreRepresentationsCategorizer: {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2}, ...}
        """
        self.coreRepresentationsCategorizer = coreRepresentationsCategorizer # {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2}}
        self.true_coefficients = true_coefficients
        self.ignore_edge = ignore_edge
        self.core_length = core_length
        self.coreRep2 = {}
        for primerep in coreRepresentationsCategorizer:
            counts = np.zeros(len(true_coefficients))
            uio_counts = coreRepresentationsCategorizer[primerep]
            for uioID in uio_counts:
                counts[uioID] += uio_counts[uioID]
            self.coreRep2[primerep] = counts

    def coreFitsConditions(self, correp, Conditions): # ANDs conditions in row together
        for rowcondition in Conditions:
            fits = True
            for edgeIndex, edgevalue in rowcondition:
                if correp[edgeIndex] != edgevalue:
                    fits = False
                    break
            if fits:
                return True
        return False
    

    def evaluate(self, filter, verbose=False):
        # for each uio of length l+k, check how many of its cores comply  with 
        # the Condition_matrix and compare that amount with the true coefficient c_{l,k}
        if verbose:
            print("evaluate filter / Condition_matrix:", filter)
        score = 0 # bigger is better, negative

        # prune the Condition_matrix - remove the edges that are of type self.ignore_edge
        Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != self.ignore_edge] 
                    for conditionrow in filter]

        counted = np.zeros(len(self.true_coefficients)) # the i'th entry is the number of correps associated to the i'th uio that fit the Conditions
        for primeCoreRep in self.coreRep2:
            #print("primeCoreRep:", primeCoreRep, self.coreRepresentationsCategorizer[primeCoreRep], self.coreFitsConditions(primeCoreRep, Conditions))
            if not primeCoreRep or self.coreFitsConditions(primeCoreRep, Conditions) == True:
                counted += self.coreRep2[primeCoreRep]
        #print()
        difference = counted - np.array(self.true_coefficients)
        if (difference < 0).any():
            return -self.INF
        return -sum(difference)
    
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
    
    def convertConditionMatrixToText(self, Condition_matrix):
        rows, columns = Condition_matrix.shape
        rowtexts = []
        for row in range(rows):
            index = 0
            rowtext = []
            aORD = ord("a")
            for i in range(self.core_length):
                for j in range(i+1, self.core_length):
                    edge = int(Condition_matrix[row][index])
                    if edge != self.ignore_edge:
                        rowtext.append(chr(aORD+i)+UIO.RELATIONTEXT[edge]+chr(aORD+j))
                    index += 1
            if rowtext:
                rowtexts.append(" AND ".join(rowtext))
        return 3*" " + 16*"-" + "\n" + "\nOR\n".join(rowtexts) + "\n"+ 3*" " + 16*"-"
    
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

    