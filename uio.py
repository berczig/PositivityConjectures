"""
step 1: generate uio
step 2: for all uio check all permutations and get the l,k correct sequences
step 3: From each l,k correct sequence extract this data(called the core):
    - last k+1 intervals
    - maximum interval from first l-1 intervals
    - p critical pairs from first intervals
step 4: This data can be viewed as a graph and we classify all these graphs(from all uio) into the emerging categories.
step 5: Sum problem: find the graph G such that the number of graphs who have G as subgraph is exactly the coefficent c_{l,k}
        - using linear programming (Adam, Gurobi)
        - using RL

# maybe we don't have  to use ~ for the filter


TODO:
    - double check that I get the right number of categories:
        uio_old:
        l,k		l,k cor seq		categories
        4,2		6416			384
        5,2                     667
        6,2		1227180			793

        uio:
        l,k		l,k cor seq		categories
        4,2		6416			440
        5,2                     723
        6,2		1227180			849

"""

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

from itertools import permutations
from math import factorial as fac
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt 
import sys
import time 
import stanley_crossentropy




# COMBINATORIC FUNCTIONS
permutations_n = {}
def getPermutationsOfN(n):
    # calculate all permutations of 1,..,n once - used multiple times
    global permutations_n
    if n not in permutations_n:
        permutations_n[n] = list(permutations(range(n)))
    return permutations_n[n]

def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0

def C_n(n):
    """ 
    n   Catalan number / uios   total correct sequences     total n-2,2 correct sequences   total n-3,3 correct sequences   total n-2,2 correct sequences categories    total n-3,3 correct sequences categories
    3   5.0
    4   14.0																											
    5   42.0
    6   132.0												6416															440
    7   429.0                                                                                                               723
    8   1430.0												1227180															848
    9   4862.0
    """
    return binomial(2*n, n)/(n+1)

def generate_all_uios(n):
    A = []
    generate_uio_rec(A, [0], n, 1)
    print("Generated", len(A), "unit order intervals encodings")
    return A

def generate_uio_rec(A, uio, n, i):
    if i == n:
        A.append(uio)
        return
    for j in range(uio[i-1], i+1):
        generate_uio_rec(A, uio+[j], n, i+1)


###

def uio_to_graph(uio):
    G=nx.Graph()
    n = len(uio)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if uio[j] <= i:
                G.add_edge(i,j)
    nx.draw(G)
    plt.show()


###############################################################################################

class UIO:

    INCOMPARABLE = 100    # (i,j) is INCOMPARABLE if neither i < j nor j < i nor i = j -- connected in the incomparability graph -- intersecting intervals
    LESS = 101              # (i,j) is LE iff i < j     interval i is to the left of j 
    EQUAL = 102              # (i,j) is EQ iff i = j     interval i and interval j are same interval
    GREATER = 103              # (i,j) is LE iff i > j     interval i is to the right of j
    RELATIONTEXT = {LESS:"<", EQUAL:"=", GREATER:">"}

    def __init__(self, uio_encoding):
        self.n = len(uio_encoding)
        self.encoding = uio_encoding

        # decode encoding to get comparison matrix
        self.comparison_matrix = np.zeros((self.n,self.n)) + self.EQUAL
        for i in range(self.n):
            for j in range(i+1, self.n):
                if uio_encoding[j] <= i:
                    self.comparison_matrix[i,j] = self.INCOMPARABLE
                    self.comparison_matrix[j,i] = self.INCOMPARABLE
                else:
                    self.comparison_matrix[i, j] = self.LESS
                    self.comparison_matrix[j,i] = self.GREATER
        self.lkCorrectSequences = {} # {(l,k):[corseq1, corseq2,...], ... }

        # compute correct sequences
        self.computeCorrectSequences()
    
    ##### CORRECT SEQUENCES ####

    def iscorrect(self, seq):
        for i in range(1,len(seq)):
            # not to the left of previos interval
            if self.comparison_matrix[seq[i], seq[i-1]] == UIO.LESS:
                return False
            # intersects with some previous interval
            intersects = False
            for j in range(0, i):
                if self.comparison_matrix[seq[i], seq[j]] in [UIO.LESS, UIO.INCOMPARABLE]:
                    intersects = True
                    #break
            if not intersects:
                return False
        return True
    
    def is_lk_correct(self,seq,l,k):
        return self.iscorrect(seq[:l]) and self.iscorrect(seq[l:])
    
    def computeCorrectSequences(self):
        self.lkCorrectSequences[(self.n,0)] = [seq for seq in getPermutationsOfN(self.n) if self.iscorrect(seq)]
    
    def computelkCorrectSequences(self, l, k):
        self.lkCorrectSequences[(l,k)] = [seq for seq in getPermutationsOfN(self.n) if self.is_lk_correct(seq,l,k)]
        self.l = l
        self.k = k

    ##### THE CORE OF CORRECT SEQUENCE ####

    def getmaximalinterval(self, subseq):
        maximals = []
        for i in subseq:
            ismaximal = True
            for j in subseq:
                if self.comparison_matrix[j, i] == UIO.GREATER: # one very redundant comparison: i==j, but whatever
                    # i can't be maximal
                    ismaximal = False
                    break
            if ismaximal:
                maximals.append(i)
        return max(maximals)
    
    def getCore(self, seq, p): # step 3 for l,k
        core = []
        # take last critical pairs (at most p)
        pairs = 0 # count number of registered critical pairs
        for i in range(self.l-1, 0, -1):
            if self.comparison_matrix[seq[i], seq[i-1]] == UIO.INCOMPARABLE:
                pairs += 1
                core.append(seq[i-1])
                core.append(seq[i])
                if pairs >= p:
                    break
        core.insert(0, pairs)

        # maximal element in first l-1
        core.append(self.getmaximalinterval(seq[:self.l-1]))

        # last k+1 elements
        core += seq[-(self.k+1):]
        return core
    
    def computeCores(self, p):
        # Assumes that computelkCorrectSequences has been called
        self.cores = []
        for seq in self.lkCorrectSequences[(self.l, self.k)]:
            self.cores.append(self.getCore(seq, p))

    #### GRAPH REPRESENTATIONS(TUPLE OF EDGES) OF CORE ####

    def getCoreRepresentations(self):
        representations = []
        for core in self.cores:
            k = len(core)
            representations.append(tuple([self.comparison_matrix[core[i], core[j]] for i in range(1, k) for j in range(i+1, k)]))
        return representations
    
    #### COEFFICIENT ####

    def getCoefficient(self):
        # assumes that lk correct sequences allready  have been calculated
        return len(self.lkCorrectSequences[self.l, self.k]) - len(self.lkCorrectSequences[self.n, 0])

def getAllCoreRepresentationsOfAllUios(l,k,p):

    # Compute UIO length
    n = l+k

    # step 1
    print(111122222222)
    uios = [UIO(uio_encoding) for uio_encoding in generate_all_uios(n)]
    coreRepresentations = []
    print(122222222222222)

    g = 0
    for uio in uios:
        g += 1
        print("k:", g)
        # step 2
        uio.computelkCorrectSequences(l,k)

        # step 3
        uio.computeCores(p)

        # step 4
        coreRepresentations.append(uio.getCoreRepresentations())

    return uios, coreRepresentations

class ConditionEvaluator:

    def __init__(self, l, k, p, ignoreEdge):
        self.coreRepresentations = [] # the i'th entry contains all coreRepresentations of the i'th uio
        self.trueCoefficients = []
        self.ignoreEdge = ignoreEdge        
        self.l = l
        self.k = k
        self.p = p
        self.coreOccurrences = {} # category/core type: list(i'th entry is number of times this tpe appears in i'th uio)

        # Compute UIO length
        self.n = l+k
        t = time.time()

        print("Create ConditionEvaluator with n =", self.n, "l =", l, "k =", k, "p =", p)

        # step 1
        print("Generating uios and correct sequences...")
        self.uios = []
        uio_encodings = generate_all_uios(self.n)
        self.uios_n = len(uio_encodings)
        printvalue = self.uios_n//16
        print("printe every:", printvalue)
        for i, uio_encoding in enumerate(uio_encodings):
            if i%printvalue == 0:
                print(i+1, "/", self.uios_n)
            self.uios.append(UIO(uio_encoding))

        print("Computing uio l,k correct sequences and cores...")
        for i,uio in enumerate(self.uios):
            if i%printvalue == 0:
                print(i+1, "/", self.uios_n)
            # step 2
            uio.computelkCorrectSequences(l,k)

            # step 3
            uio.computeCores(p)

            # step 4
            self.coreRepresentations.append(uio.getCoreRepresentations())
            self.trueCoefficients.append(uio.getCoefficient())
        self.countCategories2()
        print("ConditionEvaluator ready! - time elapsed to create:", time.time()-t)

    def countComplyingCores(self, coreRepresentations, Conditions):
        if len(coreRepresentations) == 0:
            return 0
            
        counter = 0
        coreedges = len(coreRepresentations[0]) # assume same number of critical pairs

        # for the case were the number of critical pairs can vary: using condition_matrix create another conditions2 for the case with fewer edges, then given a correp always check the length and pick the right conditions
        
        def coreFitsConditions(correp): # ANDs conditions in row together
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
    
    def coreFitsConditions2(self, correp, Conditions): # ANDs conditions in row together
        for rowcondition in Conditions:
            fits = True
            for edgeIndex, edgevalue in rowcondition:
                if correp[edgeIndex] != edgevalue:
                    fits = False
                    break
            if fits:
                return True
        return False
    
    def evaluate2(self, Condition_matrix, verbose=False):
        # for each uio of length l+k, check how many of its cores comply  with 
        # the Condition_matrix and compare that amount with the true coefficient c_{l,k}
        # 

        if verbose:
            print("evaluate Condition_matrix:", Condition_matrix)
        score = 0 # bigger is better, negative

        # Condition_matrix is not so straight to the point when one wants to check the conditions, so let's prune it a bit so it's easier to do the checking
        Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != self.ignoreEdge] 
                    for conditionrow in Condition_matrix]

        counted = np.zeros(self.uios_n) # the i'th entry is the number of correps associated to the i'th uio that fit the Conditions
        for primeCoreRep in self.coreOccurrences:
            #print("primeCoreRep:", primeCoreRep)
            if self.coreFitsConditions2(primeCoreRep, Conditions) == True:
                #print("hey")
                counted += self.coreOccurrences[primeCoreRep]
        difference = counted - np.array(self.trueCoefficients)
        #print("sum:", sum(counted))
        #print("sum:", sum(difference))
        #print("sum:", sum(self.trueCoefficients))
        #print("difference:", difference)
        for x in difference:
            if x < 0:
                return -np.inf
        return -sum(difference)

    def evaluate(self, Condition_matrix, verbose=False):
        return self.evaluate2(Condition_matrix, verbose)
        # for each uio of length l+k, check how many of its cores comply  with 
        # the Condition_matrix and compare that amount with the true coefficient c_{l,k}
        # 

        if verbose:
            print("evaluate Condition_matrix:", Condition_matrix)
        score = 0 # bigger is better, negative

        # Condition_matrix is not so straight to the point when one wants to check the conditions, so let's prune it a bit so it's easier to do the checking
        Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != self.ignoreEdge] 
                    for conditionrow in Condition_matrix]

        for i, corereps in enumerate(self.coreRepresentations):
            amount = self.countComplyingCores(corereps, Conditions)
            difference = amount - self.trueCoefficients[i]

            if difference < 0: # too many conditions, didn't include enough cores
                return -np.inf
            score -= difference
        return score
    
    def convertConditionMatrixToText(self, Condition_matrix):
        rows, columns = Condition_matrix.shape
        rowtexts = []
        for row in range(rows):
            index = 0
            rowtext = []
            aORD = ord("a")
            for i in range(self.n):
                for j in range(i+1, self.n):
                    edge = Condition_matrix[row][index]

                    if edge != self.ignoreEdge:
                        rowtext.append(chr(aORD+i)+UIO.RELATIONTEXT[edge]+chr(aORD+j))
                    index += 1
            if rowtext:
                rowtexts.append(" AND ".join(rowtext))
        return " OR \n".join(rowtexts)
    
    def countCategories2(self):
        categories = {} # category:ID
        # knows the representative corereps
        # how many of them per uio
        ID = 0
        counter = {} # categoryID:dict(uioID:occurrences)
        for uioID, corereps in enumerate(self.coreRepresentations):
            for corerep in corereps:
                # determine category ID
                if corerep not in categories:
                    ID = len(categories)
                    categories[corerep] = ID
                else:
                    ID = categories[corerep]

                # count this observed category
                if ID not in counter:
                    counter[ID] = {uioID:1}
                else:
                    categoryOccurrencer = counter[ID]
                    if uioID not in categoryOccurrencer:
                        categoryOccurrencer[uioID] = 1
                    else:
                        categoryOccurrencer[uioID] += 1

        # Turn collected category-count data into a matrix
        for cat in categories:
            #print("cat:", cat)
            counted = np.zeros(self.uios_n)
            ID = categories[cat]
            for uioID in counter[ID]:
                counted[uioID] = counter[ID][uioID]
            self.coreOccurrences[cat] = counted

        columns = len(categories)
        print("Found",columns, "categories")
        print("size:", total_size(counter))
    
    def countCategories(self):
        categories = {} # category:ID
        categories_counters = [] # list of dicts(category counters)
        # knows the representative corereps
        # how many of them per uio
        ID = 0
        for corereps in self.coreRepresentations:
            counter = {}
            for i, corerep in enumerate(corereps):
                # determine category ID
                if corerep not in categories:
                    ID = len(categories)
                    categories[corerep] = ID
                else:
                    ID = categories[corerep]

                # count this observed category
                if ID not in counter:
                    counter[ID] = 1
                else:
                    counter[ID] += 1
            categories_counters.append(counter)

        # Turn collected category-count data into a matrix
        columns = len(categories)
        print("Found",columns, "categories")
        counts = []
        for i in range(self.uios_n):
            counter = categories_counters[i]
            row = [0 if j not in counter else counter[j] for j in range(columns)]
            counts.append(row)

        print("Computed count matrix of shape (", self.uios_n,",",columns, ")",sep="")
        print("size:", total_size(counts), counts[0])
        return counts
        # size of matrix categories representation:
        # 4,2 490700
        # 5,2 2637756
        # 6,2 9954224
          #    39537300

        # size of matrix categories representation:
        # 4,2 293756
        # 5,2 
        # 6,2 39537300
        
        #num = sum([val for val in counter.values()])
        #for key in counter:
        #    print(key, counter[key])
        #print("num:", num)


    # step 1: encode right condition into condition-MATRIX and check against all correct seqs
            
    # step2 : RL: deep neural cross entropy method
    # 
    # neural network policy structures:
    #   1: 45 or 46 , 60, 61, outputs for all the possible actions
    #   2: there is a 2. input telling you at what edge we are, just 4 outputs
    # 
    # Training happens like this:
    #   top 10 % become training data: conditions|->probability dist.
    #   train our current policy / neural network w.r.t. this classificaation. 


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

# Example of a representation of a core (a,b,c,d,e,f) of length 6 with a representation of length 6*5/2 = 15:
#    0   1   2   3   4       5   6   7   8       9   10  11      12  13      14  
#    a,b a,c a,d a,e a,f     b,c b,d b,e b,f     c,d c,e c,f     d,e d,f     e,f

def checkThmConditionMatrix():
    # Set UIO parameters
    tstart = time.time()
    CE = ConditionEvaluator(l=4, k=2, p=1, ignoreEdge=0)

    # The thm needs c<e and d<f  OR  a>e and b > f  that translates to 
    ThmConditionFilter = np.zeros((2,15))
    ThmConditionFilter[0][10] = UIO.LESS
    ThmConditionFilter[0][13] = UIO.LESS
    ThmConditionFilter[1][3] = UIO.GREATER
    ThmConditionFilter[1][8] = UIO.GREATER

    print(CE.convertConditionMatrixToText(ThmConditionFilter))

    tnow = time.time()
    print("score:", CE.evaluate(ThmConditionFilter))
    print("checking:", time.time()-tnow)
    print("all.", time.time()-tstart)

def inspectStatesFromFile(file, edges, edgetypes):
    print("Reading file ", file)
    print("UIO of length n. Inspecting states...")
    CE = ConditionEvaluator(l=4, k=2, p=1, ignoreEdge=UIO.INCOMPARABLE)
    with open(file) as f:
        lines = f.readlines()
        for edge in range(0, len(lines), edges):
            state = []
            for i in range(edges):
                if i < edges-1:
                    line = lines[edge+i][1:]
                else:
                    line = lines[edge+i][1:-2] # remove ] and \n
                vectoraction = eval(line.replace(" ", ",")) #convert black spaces to commas. Evaluate this string as python list
                state += vectoraction
            print(15*"-")
            condmat = stanley_crossentropy.convertStateToConditionMatrix(state)
            conditiontext = CE.convertConditionMatrixToText(condmat)
            print(conditiontext, "\nhas a score of ", CE.evaluate(condmat))
            
def testCountCategories():
    CE = ConditionEvaluator(l=6, k=2, p=1, ignoreEdge=0)
    CE.countCategories2()

if __name__ == "__main__":
    testCountCategories()
    #inspectStatesFromFile("best_species_txt_763.txt", 15, 7)
    #checkThmConditionMatrix()