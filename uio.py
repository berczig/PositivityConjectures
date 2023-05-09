"""
step 1: generate uio
step 2: for all uio check all permutations and get the l,k correct sequences
step 3: From each l,k correct sequence extract this data(called the core):
    - last k+1 intervals
    - maximum interval from first l-1 intervals
    - p critical pairs from first intervals
step 4: This data can be viewed as a graph and we classify all these graphs(from all uio) into the emerging coreTypes. 
    Then number of coreTypes is much smaller than the number of l,k correct sequences
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
    - run conditionmatrix with fewer/more rows(logical) so: 
            COND1 OR COND2
            AND
            COND3 OR COND4
    - change adams saveouts
    - save uio data                                          DONE
    - dynamic change of learning rate
    - fix this error: when all the score sof the graphs appear to be the same then the multinomial distribution function gives an error     DONE
    - fill out table of combinatorical stuff
    - make the parameters in stanley_cross nicer and more clear
    - np.seed is not enough to replicate same results
    - graph over loss while training
    - plot graph as networkx
    - save model?   
        weights                 DONE
        step                    DONE
        bestscores              DONE
        meansccore              DONE
        calculationtime         DONE
        numberofgraphs          DONE
        adams elites etc    ..
        stategraph scores   ..
            different format
        bestgraph in each step ..

    - save scores of graphs?
    - permutate core
    - incooperate aprior knowledge about the core a < b etc.
    - only use random portion of uios when doing 1 step like 300
    - display best graph
    - evolution animation
    -tuesday 02.05, 2 pm
    - check getmaximalinterval
    - calculate eschers and minimal interval 



other notes:
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

    


Info table:
    n   Catalan number / uios   total correct sequences     total n-2,2 correct sequences   total n-3,3 correct sequences   total n-2,2 correct sequences categories    total n-3,3 correct sequences categories
    3   5.0
    4   14.0																											
    5   42.0
    6   132.0												6416															440
    7   429.0                                                                                                               723                                         10238
    8   1430.0												1227180															848                                         40946
    9   4862.0                                                                              19121202                                                                    89443
"""
from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from memory_profiler import profile
try:
    from reprlib import repr
except ImportError:
    pass

from itertools import permutations
from math import factorial as fac
import numpy as np
import random
#import networkx as nx
import matplotlib.pyplot as plt 
import sys
import pickle
import time 
from extra import Loadable




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
        self.comparison_matrix = np.zeros((self.n,self.n)) + self.EQUAL # (i,j)'th index says how i is in relation to j
        for i in range(self.n):
            for j in range(i+1, self.n):
                if uio_encoding[j] <= i:
                    self.comparison_matrix[i,j] = self.INCOMPARABLE
                    self.comparison_matrix[j,i] = self.INCOMPARABLE
                else:
                    self.comparison_matrix[i, j] = self.LESS
                    self.comparison_matrix[j,i] = self.GREATER
        """
        for i in range(n):
            for j in range(1, i-1):
                if i <= """
        self.lkCorrectSequences = {} # {(l,k):[corseq1, corseq2,...], ... }
        self.lkCorrectSequences_n = {} # {(l,k):number of (l,k) correct sequences}

        self.eschers = {} # {n:[eschers of length n]}
        self.subeschers = {} # {(escher, length k):all k-subeschers}

        # compute correct sequences
        #self.computeCorrectSequences()
    
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
        self.lkCorrectSequences_n[(self.n,0)] = len(self.lkCorrectSequences[(self.n,0)])
    
    def computelkCorrectSequences(self, l, k):
        self.lkCorrectSequences[(l,k)] = [seq for seq in getPermutationsOfN(self.n) if self.is_lk_correct(seq,l,k)]
        self.lkCorrectSequences_n[(l,k)] = len(self.lkCorrectSequences[(l,k)])
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
        print(max(maximals) == np.argmax(maximals))
        return np.argmax(maximals)
    
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

    #################### ESCHERS ###################

    def isescher(self, seq):
        for i in range(len(seq)-1):
            if self.comparison_matrix[seq[i], seq[i+1]] == UIO.GREATER:
                return False
        return self.comparison_matrix[seq[-1], seq[0]] != UIO.GREATER

    def computeeschers(self, n):
        print("compute eschers for length", n)
        self.eschers[n] = [seq for seq in getPermutationsOfN(n) if self.isescher(seq)]
    
    def computevalidsubeschers(self, escher, k): # k > 0 
        subeschers = []
        #print("k:", k)
        h = len(escher)
        for m in range(h): # m can be 0
            #print("m:", m)
            #print("indices:", (m+k)%self.n, (m+1)%self.n)
            cond1 = self.comparison_matrix[escher[(m+k)%h], escher[(m+1)%h]] != UIO.GREATER # EQUAL also intersects
            #print("cond1:", cond1)
            #print("indices:", m, (m+k+1)%self.n)
            cond2 = self.comparison_matrix[escher[m], escher[(m+k+1)%h]] != UIO.GREATER
            #print("cond2:", cond2)
            if cond1 and cond2:
                lastindex = m+k+1
                if lastindex > self.n:
                    subeschers.append(escher[m+1:]+escher[:lastindex-h])
                    # subeschers.append(list(range(m+1,self.n))+list(range(lastindex-self.n)))
                else:
                    # subeschers.append(list(range(m+1,lastindex)))
                    subeschers.append(escher[m+1:lastindex])
            
        self.subeschers[(escher,k)] = subeschers

    def computevalidsubescherstartingpoints(self, escher, k): # k > 0 
        subeschers = []
        #print("k:", k)
        h = len(escher)
        for m in range(h): # m can be 0
            #print("m:", m)
            #print("indices:", (m+k)%self.n, (m+1)%self.n)
            cond1 = self.comparison_matrix[escher[(m+k)%h], escher[(m+1)%h]] != UIO.GREATER # EQUAL also intersects
            #print("cond1:", cond1)
            #print("indices:", m, (m+k+1)%self.n)
            cond2 = self.comparison_matrix[escher[m], escher[(m+k+1)%h]] != UIO.GREATER
            #print("cond2:", cond2)
            if cond1 and cond2:
                lastindex = m+k+1
                if lastindex > self.n:
                    subeschers.append(escher[m+1:]+escher[:lastindex-h])
                    # subeschers.append(list(range(m+1,self.n))+list(range(lastindex-self.n)))
                else:
                    # subeschers.append(list(range(m+1,lastindex)))
                    subeschers.append(escher[m+1:lastindex])
            
        self.subeschers[(escher,k)] = subeschers

    def getsubeschers(self, escher, k):
        if (escher, k) not in self.subeschers:
            self.computevalidsubeschers(escher, k)
        return self.subeschers[(escher, k)]

    def getInsertionPoints(self, u, v, lcm = 0): # u of length n > k
        n = len(u)
        k = len(v)
        self.comparison_matrix
        if lcm == 0:
            lcm = np.lcm(n,k) # implement
        points = []
        for i in range(lcm):
            if self.comparison_matrix[u[i%n], v[(i+1)%k]] != UIO.GREATER and self.comparison_matrix[v[i%k], u[(i+1)%n]] != UIO.GREATER:
                points.append(i)
        #print("found {} insertion points between {} and {}".format(len(points),u,v))
        return points

    def getEscherPairs(self, n,k):
        self.pairs = []
        for seq in getPermutationsOfN(n+k):
            if self.isescher(seq[:n]) and self.isescher(seq[n:]):
                self.pairs.append((seq[:n], seq[n:]))
        print("escher pairs:", len(self.pairs))
                
    def getEscherCore(self, u, v): # u is length n
        n = len(u)
        k = len(v)
        lcm = np.lcm(n, k)
        #print("type:", type(u), u)
        uu = u*(lcm//n)
        return (self.getInsertionPoints(u, v, lcm), self.getsubeschers(uu, k))

    def getEschersCores(self, n, k):
        cores = []
        for v in [n,k]:
            if v not in self.eschers:
                self.computeeschers(v)
        for u,v in self.pairs:
        #for u in self.eschers[n]:
            #for v in self.eschers[k]:
            cores.append(self.getEscherCore(u,v))
        return cores            

class UIODataExtractor:
    """
    UIODataExtractor Creates all uio and computes l,k and correct sequences and the cores and counts the core types
    """
    def __init__(self, l, k, p):
        self.l = l
        self.k = k
        self.p = p
        self.coreTypesRaw = [] # list of all coreTypes
        self.coreTypes = {} # coreType(int):occurrences(dict), key i in occurrences is how often that type appeared in i'th uio
        self.trueCoefficients = [] # i'th entry is the coefficient c_{l,k} of the i'th uio's CSF

        # Compute UIO length
        self.n = l+k

        print("Create UIODataExtractor with l =", l, "k =", k, "p =", p)
        t = time.time()

        # step 1 - Generate UIOs
        self.uios = []
        uio_encodings = generate_all_uios(self.n)
        self.uios_n = len(uio_encodings)
        printvalue = self.uios_n//16

        print("Generating uios and correct sequences...(print every", str(printvalue)+")")
        for i, uio_encoding in enumerate(uio_encodings):
            if i%printvalue == 0:
                print(i+1, "/", self.uios_n)
            self.uios.append(UIO(uio_encoding))
        print("Generated correct sequences:", sum([uio.lkCorrectSequences_n[(self.n, 0)] for uio in self.uios]))

        print("Computing uio l,k correct sequences and cores...")
        for i,uio in enumerate(self.uios):
            if i%printvalue == 0:
                print(i+1, "/", self.uios_n)
            # step 2 - compute l,k correct sequences
            uio.computelkCorrectSequences(l,k)

            # step 3 - compute the cores
            uio.computeCores(p)

            # step 4.1 - generate the coreTypes from the cores (The core is independent of the comparison matrix from its UIO) 
            self.coreTypesRaw.append(uio.getCoreRepresentations())
            self.trueCoefficients.append(uio.getCoefficient())
        self.trueCoefficients = np.array(self.trueCoefficients)
        print("Generated l,k correct sequences:", sum([uio.lkCorrectSequences_n[(self.l, self.k)] for uio in self.uios]))

        # step 4.2 - classify the coreTypes by counting how many different types there are
        self.countCategories()

        print("Created UIODataExtractor in", round(time.time()-t,3), "seconds")

    def countCategories(self):
        print("Categorizing core types...")
        categories = {} # category:ID
        # knows the representative corereps
        # how many of them per uio
        ID = 0
        counter = {} # categoryID:dict(uioID:occurrences)
        for uioID, corereps in enumerate(self.coreTypesRaw):
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
            #for uioID in counter[ID]:
                #counted[uioID] = counter[ID][uioID]
            self.coreTypes[cat] = counter[ID]

        columns = len(categories)
        print("Found",columns, "categories")
        #print("size:", total_size(counter))

class ConditionEvaluator(Loadable):

    def __init__(self, l, k, p, ignoreEdge, uiodataextractor:UIODataExtractor=None):
        self.l = l
        self.k = k
        self.p = p
        self.ignoreEdge = ignoreEdge        

        # Compute UIO length
        self.n = l+k

        if uiodataextractor != None:
            self.trueCoefficients = np.array(uiodataextractor.trueCoefficients)
            self.coreTypes = uiodataextractor.coreTypes 
            self.activeCoreTypes = self.coreTypes
            self.activeTrueCoefficients = self.trueCoefficients
            self.uios_n = uiodataextractor.uios_n
            self.activeuios_n = self.uios_n
            print("Created ConditionEvaluator Using UIODataExtractor, n =", self.n, "l =", l, "k =", k, "p =", p)
            print("Using", len(self.coreTypes), "core types / categories for the ConditionEvaluator")

    def load(self, filename):
        super().load(filename)
        self.trueCoefficients = np.array(self.trueCoefficients)         # I set these because I use an old save where this data isnt saved
        self.activeCoreTypes = self.coreTypes
        self.activeTrueCoefficients = self.trueCoefficients
        self.activeuios_n = self.uios_n
        print("Created ConditionEvaluator by loading file, n =", self.n, "l =", self.l, "k =", self.k, "p =", self.p)
        print("Using", len(self.coreTypes), "core types / categories for the ConditionEvaluator")
    
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
    

    def evaluate(self, Condition_matrix, verbose=False):
        # for each uio of length l+k, check how many of its cores comply  with 
        # the Condition_matrix and compare that amount with the true coefficient c_{l,k}
        if verbose:
            print("evaluate Condition_matrix:", Condition_matrix)
        score = 0 # bigger is better, negative

        # Condition_matrix is not so straight to the point when one wants to check the conditions, so let's prune it a bit so it's easier to do the checking
        Conditions = [[(i, edgecondition) for i, edgecondition in enumerate(conditionrow) if edgecondition != self.ignoreEdge] 
                    for conditionrow in Condition_matrix]

        counted = np.zeros(self.activeuios_n) # the i'th entry is the number of correps associated to the i'th uio that fit the Conditions
        for primeCoreRep in self.activeCoreTypes:
            if self.coreFitsConditions(primeCoreRep, Conditions) == True:
                dict_ = self.activeCoreTypes[primeCoreRep]
                for uioID in dict_:
                    a = dict_[uioID]
                    counted[uioID] += a
        difference = counted - self.activeTrueCoefficients
        for x in difference:
            if x < 0:
                return -np.inf
        return -sum(difference)
    
    def convertConditionMatrixToText(self, Condition_matrix):
        rows, columns = Condition_matrix.shape
        rowtexts = []
        for row in range(rows):
            index = 0
            rowtext = []
            aORD = ord("a")
            for i in range(self.n):
                for j in range(i+1, self.n):
                    edge = int(Condition_matrix[row][index])

                    if edge != self.ignoreEdge:
                        rowtext.append(chr(aORD+i)+UIO.RELATIONTEXT[edge]+chr(aORD+j))
                    index += 1
            if rowtext:
                rowtexts.append(" AND ".join(rowtext))
        return " OR \n".join(rowtexts)
    
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

## Standalone functions 
def calculateCombinatoricalTable():

    pass

def checkThmConditionMatrix():
    # Set UIO parameters
    tstart = time.time()
    #CE = ConditionEvaluator(l=4, k=2, p=1, ignoreEdge=0, uiodataextractor=UIODataExtractor(l=4,k=2,p=1))
    CE = ConditionEvaluator(l=4, k=2, p=1, ignoreEdge=100)
    CE.load("saves/coreTypes_l=4_k=2_p=1_ignore=100.bin")

    # The thm needs c<e and d<f  OR  a>e and b > f  that translates to 
    ThmConditionFilter = np.ones((2,15))*UIO.INCOMPARABLE
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
            import stanley_crossentropy # pickle problem mixed with circular problem
            condmat = stanley_crossentropy.convertStateToConditionMatrix(state)
            conditiontext = CE.convertConditionMatrixToText(condmat)
            print(conditiontext, "\nhas a score of ", CE.evaluate(condmat))
            
def testCountCategories():
    CE = ConditionEvaluator(l=6, k=2, p=1, ignoreEdge=0)
    CE.countCategories()

def testsave():
    t = time.time()
    l = 4
    k = 2
    p = 1
    ignore = UIO.INCOMPARABLE
    DE = UIODataExtractor(l,k,p)
    CE = ConditionEvaluator(l,k,p,ignore,DE)
    CE.save("coreTypes_l={}_k={}_p={}_ignore={}.bin".format(l,k,p, ignore))
    print("testsave elapsed:", time.time()-t)

def testload():
    l = 4
    k = 2
    p = 1
    CE = ConditionEvaluator(l,k,p,0)
    CE.load("coreTypes_l=4_k=2_p=1.bin")
    for key in CE.coreTypes:
        print(key, CE.coreTypes[key])

def eschertest():
    n = 6
    A = generate_all_uios(n)
    A = [[0, 0, 1, 1, 4, 4]]
    for uio_encod in A:
        uio = UIO(uio_encod)
        uio.computeeschers(n)
        eschers = uio.eschers[n]
        uio.computeeschers(2)
        uio.computeeschers(4)
        print("2 eschers:", len(uio.eschers[2]))
        print("4 eschers:", len(uio.eschers[4]))
        neschers = len(eschers)
        print(25*"#", "uio:", uio_encod, 25*"#")
        uio.computelkCorrectSequences(n, 0)
        print(" - has", neschers, "eschers and {} corrects sequences".format(uio.lkCorrectSequences_n[(n,0)]))
        for i, escher in enumerate(eschers):
            print("{}/{}".format(i+1, neschers), "escher:", escher)
            for k in range(1, n+1):
                uio.computevalidsubeschers(escher, k)
                print(k, "-subeschers:", uio.subeschers[(escher, k)], sep="")
        
        """for seq in getPermutationsOfN(n-1):
            if uio.is_lk_correct(seq, n, 0):
                corrects += 1
            if uio.isescher(seq):
                eschers += 1
        dif  =corrects-eschers
        if dif != 0:
            print("#"*200, dif)"""
        

def eschercoretest():
    n = 4
    k = 2
    N = n+k
    #A = generate_all_uios(N)
    A = [[0,0,1,1,4,4]]
    #A = [[0,0,0,0,0,0,0,0,0,0]]
    for uio_encod in A:
        uio = UIO(uio_encod)
        t = time.time()
        uio.getEscherPairs(n,k)
        print("elapsed time: {}".format(time.time()-t))
        cores = uio.getEschersCores(n,k)
        for core in cores:
            print(core)

if __name__ == "__main__":
    #testsave()
    #testload()
    #testCountCategories()
    #inspectStatesFromFile("best_species_txt_763.txt", 15, 7)
    #checkThmConditionMatrix()
    eschercoretest()