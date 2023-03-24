import numpy as np
import time

print('Hello')

IGNORE_EDGE = 0

seqlength = 15
n = 30

filter = np.ones((2, seqlength))*IGNORE_EDGE
filter[0][0] = 1
filter[1][3] = 1

cor = np.ones((n, seqlength))
cor[-1] = 0
cor[-2][0] = 0
cor[-3][3] = 0
cor[-4][0] = 0
cor[-4][3] = 0
print(cor)
print(filter)

def evaluate_Conditions2(lk_correct_seqs, Condition_matrix):
    # step 1: encode right condition into condition-MATRIX and check against all correct seqs
    counter = 0

    def checkrowcondition(lkcorseq, rowcondition):
        for i in range(seqlength):
            if rowcondition[i] != IGNORE_EDGE:
                if rowcondition[i] != lkcorseq[i]:
                    return False
        return True

    for lkcorseq in lk_correct_seqs:
        if all([checkrowcondition(lkcorseq, rowcondition) for rowcondition in Condition_matrix]):
            counter += 1

    return counter

def evaluate_Conditions(lk_correct_seqs, Condition_matrix):
    # step 1: encode right condition into condition-MATRIX and check against all correct seqs
    counter = 0

    def checkrowcondition(lkcorseq, rowcondition):
        for i in range(seqlength):
            if rowcondition[i] != IGNORE_EDGE:
                if rowcondition[i] != lkcorseq[i]:
                    return False
        return True

    for lkcorseq in lk_correct_seqs:
        if any([checkrowcondition(lkcorseq, rowcondition) for rowcondition in Condition_matrix]):
            counter += 1

    return counter

def count_cores(coreRepresentations, Condition_matrix):
    # step 1: encode right condition into condition-MATRIX and check against all correct seqs
    IGNORE_EDGE = 0
    counter = 0
    coreedges = len(coreRepresentations[0])

    # Condition_matrix is not so straight to the point when one wants to check the conditions, so let's prune it a bit so it's easier to do the checking
    Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != IGNORE_EDGE] 
                  for conditionrow in Condition_matrix]

    print("Condition_matrix:", Condition_matrix)
    print("Conditions:", Conditions)
    
    def coreFitsConditions(correp):
        for rowcondition in Conditions:
            fits = True
            for edgeIndex, edgevalue in rowcondition:
                if correp[edgeIndex] != edgevalue:
                    fits = False
                    break
            if fits:
                return True
        return False
    
    # count how many fit 1 of the conditions
    for correp in coreRepresentations:
        if coreFitsConditions(correp):
            counter += 1

    return counter
            

t = time.time()
print("A:", evaluate_Conditions(cor, filter))
print("B:", count_cores(cor, filter))
print("time:", time.time()-t)