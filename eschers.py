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

#####

def compareJAoutput(file1, file2):
    Pcores = []
    for file in [file1, file2]:
        cores = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                cores.append(eval(line))
        Pcores.append(cores)
    J,A = Pcores
    for x in J:
        if x not in A:
            print("!!!",x)       

###############################################################################################

class UIO:

    INCOMPARABLE = 100    # (i,j) is INCOMPARABLE if neither i < j nor j < i nor i = j -- connected in the incomparability graph -- intersecting intervals
    LESS = 101              # (i,j) is LE iff i < j     interval i is to the left of j 
    EQUAL = 102              # (i,j) is EQ iff i = j     interval i and interval j are same interval
    GREATER = 103              # (i,j) is LE iff i > j     interval i is to the right of j
    RELATIONTEXT = {LESS:"<", EQUAL:"=", GREATER:">"}

    def __init__(self, uio_encoding):
        self.N = len(uio_encoding)
        self.encoding = uio_encoding

        # decode encoding to get comparison matrix
        self.comparison_matrix = np.zeros((self.N,self.N)) + self.EQUAL # (i,j)'th index says how i is in relation to j
        for i in range(self.N):
            for j in range(i+1, self.N):
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

        #self.eschers = {} # {n:[eschers of length n]}
        self.escherpairs = {} # {(n,k,l):[tripple escher pair]}
        self.subeschers = {} # {(escher, length k):all k-subeschers}


    #### COEFFICIENT ####

    def getCoeffientByEscher(self, n,k,l=0):
        if l == 0:
            return len(self.getEscherPairs(n,k)) - len(self.getEscherPairs(n+k))
        return 2*len(self.getEscherPairs(n+k+l)) + len(self.getEscherPairs(n,k,l)) - len(self.getEscherPairs(n+l,k)) - len(self.getEscherPairs(n+k,l)) - len(self.getEscherPairs(l+k,n))
    #################### ESCHERS ###################

    def setnlk(self, n_, k_, l_, verbose=False):
        global n,k,l, npk, lcm
        n = n_
        k = k_
        l = l_
        npk = n+k
        if verbose:
            print("setnlk", n_, k_, l_, npk)
        lcm = np.lcm(n,k)

    def isescher(self, seq):
        for i in range(len(seq)-1):
            if self.isarrow(seq, i, i+1) == False:
                return False
        return self.isarrow(seq, -1, 0)
    
    def computevalidsubeschers(self, escher, k): # k > 0 
        subeschers = []
        #print("k:", k)
        h = len(escher)
        for m in range(h): # m can be 0
            #print("m:", m)
            cond1 = self.comparison_matrix[escher[(m+k)%h], escher[(m+1)%h]] != UIO.GREATER # EQUAL also intersects
            #print("cond1:", cond1)
            cond2 = self.comparison_matrix[escher[m], escher[(m+k+1)%h]] != UIO.GREATER
            #print("cond2:", cond2)
            if cond1 and cond2:
                lastindex = m+k+1
                if lastindex > h:
                    subeschers.append(escher[m+1:]+escher[:lastindex%h])
                else:
                    # subeschers.append(list(range(m+1,lastindex)))
                    subeschers.append(escher[m+1:lastindex])
            
        self.subeschers[(escher,k)] = subeschers

    def isarrow(self, escher, i,j):
        try:
            w = self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER
        except:
            print("error, isarrow:", escher, i, j)
        return w # EQUAL also intersects

    def getpseudosubescherstartingpoints(self, escher, k): # k > 0, pretends the input is an escher and finds valid k-subescher 
        subeschersstartingpoint = [] # start of the box
        h = len(escher)
        #print("escher length:", h)
        for m in range(-1, h-1): # m can be 0, m is before the start of the box mBBBB#
            cond1 = self.isarrow(escher, (m+k)%h, (m+1)%h)
            cond2 = self.isarrow(escher, m%h, (m+k+1)%h)
            #if m == 1:
            #    print((m+k)%h, escher[(m+k)%h])
            #    print((m+1)%h, escher[(m+1)%h])
            #    print(m, escher[m])
            #    print((m+k+1)%h, escher[(m+k+1)%h])
            #print("m:", m)
            #print("cond1:", cond1)
            #print("cond2:", cond2)
            if cond1 and cond2:
                subeschersstartingpoint.append(m+1)
        return subeschersstartingpoint

    def getsubeschers(self, escher, k):
        if (escher, k) not in self.subeschers:
            self.computevalidsubeschers(escher, k)
        return self.subeschers[(escher, k)]

    def getInsertionPoints(self, u, v, lcm = 0): # u of length n > k
        n = len(u)
        k = len(v)
        if lcm == 0:
            lcm = np.lcm(n,k)
        points = []
        for i in range(lcm):
            if self.comparison_matrix[u[i%n], v[(i+1)%k]] != UIO.GREATER and self.comparison_matrix[v[i%k], u[(i+1)%n]] != UIO.GREATER:
                points.append(i)
        #print("found {} insertion points between {} and {}".format(len(points),u,v))
        return points

    def getFirstInsertionPoint(self, u,v,lcm=0): # the returned G can be bigger than len(u) and len(v)
        n = len(u)
        k = len(v)
        if lcm == 0:
            lcm = np.lcm(n,k)
        for i in range(lcm):
            if self.comparison_matrix[u[i%n], v[(i+1)%k]] != UIO.GREATER and self.comparison_matrix[v[i%k], u[(i+1)%n]] != UIO.GREATER:
                return i
        return -1

    def getEscherPairs(self, n,k=0,l=0): # speedup
        #print("getEscherPairs:", n,k,l)
        if (n,k,l) in self.escherpairs:
            return self.escherpairs[(n,k,l)]

        pairs = []
        if l == 0 and k == 0:
            pairs = [seq for seq in getPermutationsOfN(n) if self.isescher(seq)]
            self.escherpairs[(n,0,0)] = pairs
            return self.escherpairs[(n,0,0)]
        if l == 0:
            for seq in getPermutationsOfN(n+k):
                if self.isescher(seq[:n]) and self.isescher(seq[n:]):
                    pairs.append((seq[:n], seq[n:]))
            self.escherpairs[(n,k,0)] = pairs
            return self.escherpairs[(n,k,0)]
        for seq in getPermutationsOfN(n+k+l):
            if self.isescher(seq[:n]) and self.isescher(seq[n:n+k]) and self.isescher(seq[n+k:]):
                pairs.append((seq[:n], seq[n:n+k], seq[n+k:]))
        self.escherpairs[(n,k,l)] = pairs
        return self.escherpairs[(n,k,l)]

    def cyclicslice(self, tuple, start, end): # end exclusive
        #print("cyclic:", tuple, start, end)
        n = len(tuple)
        start = start%n
        end = end%n
        if start < end:
            return tuple[start:end]
        return tuple[start:]+tuple[:end]
                
    def getEscherCore(self, u, v): # u is length n
        n = len(u)
        k = len(v)
        lcm = np.lcm(n, k)
        #print("type:", type(u), u)
        uu = u*(lcm//n)
        #print("uu:", uu)
        insertions = self.getInsertionPoints(u,v,lcm)
        subeschers = []
        if len(insertions) > 0:
            insertion = insertions[0]
            extrav = self.cyclicslice(v,insertion+1,insertion+2)
            #print("extrav:", extrav)
            groundtrooper = self.getpseudosubescherstartingpoints(u, k)
            #if 0 in groundtrooper:
            #    print("#"*100)
            snakefan = self.getpseudosubescherstartingpoints(uu[:insertion+1]+extrav, k)
            #print("g:", groundtrooper, "s:", snakefan)
           # if 0 in groundtrooper:
           #     subeschers == []
           # else:
            subeschers = snakefan
        return (insertions, subeschers)
        #return (self.getInsertionPoints(u, v, lcm), self.getvalidsubescherstartingpoints(uu, k))

    def getsnake(self, u, v, insertionpoint, headlength):
        n = len(u)
        k = len(v)
        lcm = np.lcm(n,k)
        if n >= k:
            top = v
            bot = u
        else:
            top = u
            bot = v

        botbot = bot*(lcm//len(bot))
        snakehead = self.cyclicslice(top,insertionpoint+1,insertionpoint+1+headlength)
        return botbot[:insertionpoint+1]+snakehead

    def concat(self, first, second, insertionpoint): # assume insertionpoint < len(first)
        # v0     vL vL+1
        # u0 ... uL uL+1
        #print("concat:", first, insertionpoint, first[:insertionpoint%n+1], self.cyclicslice(second, insertionpoint+1, insertionpoint+k+1), first[insertionpoint%n+1:])
        # (insertionpoint+1)+(k-1)+1 = insertionpoint+k+1
        return first[:insertionpoint%n+1]+self.cyclicslice(second, insertionpoint+1, insertionpoint+k+1)+first[insertionpoint%n+1:]


    def getEscherRGCore(self, u, v):
        n = len(u)
        k = len(v)
        if k > n:
            return self.getEscherRGCore(v,u)
        lcm = np.lcm(n, k)
        
        insertions = self.getInsertionPoints(u,v,lcm)
        if len(insertions) == 0:
            return (-1, -1)
        G = insertions[0]
        
        snake = self.getsnake(u,v,G,1)
        #print("snake:", G, snake, u, v)

        for i in range(1, G): # box starts at i and ends at i+k-1 (inclusive
            if self.isarrow(snake, i-1, (i+k)%(G+2)) and self.isarrow(snake, (i+k-1)%(G+2), i):
                R = i+k-1
                return (R,G)

        return (-1, G)

    def getEscherTrippleCore(self, u,v,w):
        Ruv,Guv = self.getEscherRGCore(u,v)
        Ruw,Guw = self.getEscherRGCore(u,w)
        Rvw,Gvw = self.getEscherRGCore(v,w)

        Ruw_v, Guw_v = (-1,-1)
        if Guv != -1:
            uw = self.concat(u, w, Guw)
            Ruw_v, Guw_v = self.getEscherRGCore(uw, v)
        
        Rvw_u, Gvw_u = (-1, -1)
        if Guw != -1:
            vw = self.concat(v, w, Gvw)
            Rvw_u, Gvw_u = self.getEscherRGCore(vw, u)
        return (Ruv, Guv, Ruw, Guw, Rvw, Gvw, Ruw_v, Guw_v, Rvw_u, Gvw_u)
        #return (*R1, *self.getEscherRGCore(u,w), *self.getEscherRGCore(v,w))

    def TrippleRGcoreIsGood(self, core):
        return self.RGcoreIsGood(core[:2]) and (self.RGcoreIsGood(core[2:4]) or not self.RGcoreIsGood(core[6:8])) and (self.RGcoreIsGood(core[4:6]) or not self.RGcoreIsGood(core[8:]))
    
    def getTripplEscherCores(self, n, k, l, verbose=False, savefile=False):
        cores = []
        goods = 0
        pairs = self.getEscherPairs(n,k,l)
        for u,v,w in pairs:
            core = self.getEscherTrippleCore(u,v,w)
            isgood = self.TrippleRGcoreIsGood(core)
            if isgood:
                goods += 1
            if verbose:
                #print(u,v,w, isgood, core)
                if isgood:
                    print((list(w), list(v), u))
                # ([0, 1], [7, 6], (5, 3, 4, 2))
            cores.append(core)
        return cores, goods

    def getEscherCores(self, n, k, verbose=False):
        cores = []
        pairs = self.getEscherPairs(n,k,0)
        for u,v in pairs:
        #for u in self.eschers[n]:
            #for v in self.eschers[k]:
            core = self.getEscherCore(u,v)
            if verbose:
                print(u,v, self.coreIsGood(core, n, k, np.lcm(n,k)), core)
            cores.append(core)
        return cores          

    
    def getEscherRGCores(self, n, k, verbose=False):
        cores = []
        pairs = self.getEscherPairs(n,k,0)
        goods = 0
        for u,v in pairs:
            core = self.getEscherRGCore(u,v)
            isgood = self.RGcoreIsGood(core)
            if isgood:
                goods += 1
            if verbose:
                print(u,v, isgood, core)
            cores.append(core)
        return cores, goods        

    def RGcoreIsGood(self, core):
        #print(core)
        if core[1] == -1:
            return True
        if core[0] == -1 and n > core[1]:
            return False
        
        #if core[1] >= n:
            #print("exp")
        return (core[0] <= core[1]) or (n <= core[1])

    def coreIsGood(self, core, n, k, lcm):
        # G_i i'th insertion points, Y1 = n, Y2 = n+k, R = right endpoint of left most subescher ( indexing starting at 1)
        # I     no insertions 
        # II    R < G1 < Y1 = n
        # III   n = Y1 <= G1 and n+k = Y2 < G2
        # in 0 indexing:
        # I     no insertions
        # II    0 < R < G1 < n-1
        # III   n-1 <= G1 and n+k-1 < G2 
        insertions, escherstartpoints = core
        if len(insertions) == 0:
            return True
        if len(escherstartpoints) == 0:
            return False
        
        #print(len(escherstartpoints)) 
        #print("escherstartpoints:", escherstartpoints)
        for escherstartpoint in escherstartpoints:
            if escherstartpoint > 0:
                #print(escherstartpoint+k-1, "vs", insertions[0])
                return escherstartpoint+k-1 <= insertions[0]
        return False

    def coreIsGoodold(self, core, n, k, lcm):
        # G_i i'th insertion points, Y1 = n, Y2 = n+k, R = right endpoint of left most subescher ( indexing starting at 1)
        # I     no insertions 
        # II    R < G1+1 < Y1 = n
        # III   n = Y1 <= G1 and n+k = Y2 < G2
        # in 0 indexing:
        # I     no insertions
        # II    0 < R < G1+1, G <= n-1
        # III   n <= G1 and n+k-1 < G2 
        insertions, escherstartpoints = core
        if len(insertions) == 0:
            return True
        #
        # 
        # print("insertions:", len(insertions), escherstartpoints)
        if insertions[0] <= n-1: 
            for subescherstartpoint in escherstartpoints:
                if subescherstartpoint > 0 and (subescherstartpoint + k-1) < insertions[0]+1: # completly contained
                    return True
        else: # exceptional case "R doesn't play a role"
            if len(insertions) == 1:
                return True  # because Y2 < G2 = inf
            else:
                return n+k-1 < insertions[1]
            #insertions.append(insertions[0]+lcm)  # repeat
        #if len(insertions) == 2:
            #if escherstartpoints[0] + k > insertions[0] and insertions[1] > n+k-1:
                #return True 
        return False

    ###### MAP P_{n+k} -> P_{n,k} ################
    def phi(self, u):
        # compute L, the point before the valid k-subescher
        for L in range(0, n+k): # starts at 0, 0 indexing in the paper for "5. the proof"
            if self.isarrow(u, L%npk, (L+k+1)%npk) and self.isarrow(u, (L+k)%npk, (L+1)%npk):
                break

        # the k-subescher goes from L+1 to L+k
        # the n-subescher goes from L+k+1 to L+k+n
        v = self.cyclicslice(u, L+1, L+k+1)
        w = self.cyclicslice(u, L+k+1, L+k+n+1)

        # We found the subeschers, but we have to change the starting point
        # for both subescher the startpoint should be s.t. the the L'th position (L can be 0) is the end of the original subescher
        
        v = self.rewindescher(v, L+1)
        w = self.rewindescher(w, L+1)
        #print("L:", L)
        return (w, v) #(1, 0, 3, 6, 5, 4, 2)
        """
        # Find k-multiple in [L+1:L+k]
        for qk in range(L+1, L+k+1):
            if qk%k == 0:
                break

        # Find n-multiple in [L+k+1:L] = [L+k+1:L+k+n-1] U [0:L]
        # for qn in range(L+k+1, L+k+n+1):
        for qn in list(range(L+k+1, L+k+n)) + list(range(0, L+1)): 
            if qn%n == 0:
                break
        
        # k-escher goes from qk to qk+k-1 (inclusive)
        # n-escher goes from qk to qn+n-1 (inclusive)
        v = self.cyclicslice(u, qk, qk+k)
        w = self.cyclicslice(u, qn, qn+n)

        print("L:{}, qk:{}, qn:{}".format(L, qk, qn))
        return (v,w)
        """

    def psi(self, u, v):
        #u - n-escher
        #v - k-escher
        G = self.getFirstInsertionPoint(u,v)
        print("G:", G)
        if G == -1:
            return None
        w = self.concat(u,v, G)
        if G >= n:
            #print(12*"EXP")
            v_n = v[n%k]
            while w[0] != v_n:
                w = self.rewindescher(w, 1)
        return w
        #if G < n-1:
            #return self.concat(u,v,G)
        #return self.concat(v,u,G)

    def rewindescher(self, escher, steps): # moves the escher clockwise
        x = len(escher)
        return self.cyclicslice(escher, -steps, x-steps)
        

def eschercoretest():
    A = generate_all_uios(n+k)
    # 000012
    #A = [[0,0,1,1,4,4]]
    #A = [[0,0,0,1, 2]]
    Primer7 = [[0,0,1,2,3]]
    Nonbubbler = [[0,0,1,1,3]]
    wrongy = [[0,0,0,1,1,4]]
    A = [[0,0,0,2,3,4,4]]
    A = [[0,0,1,1,2,3,3,5,7]] # 5,4 breaker 1   12.05 11:17
    #A = [[0, 0, 1, 1, 2, 3, 5]]
    #A = [[0,0,1,1,2,3,3,5,6]] # 5,4 breaker 2   12.05.11:17
    #A = [[0, 0, 1, 1, 2, 3, 4, 6]]
    #A = wrongy2 = [[0,0,1,1,2,3,4,6]] # 5,3 breaker
    #A = [[0, 0, 1, 2, 2, 3, 5]]
    #A = [[0, 0, 1, 2, 3, 3]]
    #A = wrongy2
    #A = [[0,0,1,1,2,3,3,5,6]]
    #A = wrongy
    #A = [[0,0,1,1,3,3,3]]
    #A = [[0,0,0,0,0,0]]
    #A = [[0,0,0,0,0,0,0,0,0,0]]
    for uio_encod in A:
        uio = UIO(uio_encod)
        t = time.time()
        #print("elapsed time: {}".format(time.time()-t))
        cores, goods = uio.getEscherRGCores(n,k, verbose=True)
        cores_ = uio.getEscherCores(n,k, verbose=False)
        truecoef = uio.getCoeffientByEscher(n,k,0)
        print("conjecture:", goods,  "true:", truecoef,"eschers:", len(cores), uio_encod)
        if goods != truecoef:
            print("conjecture:", goods,  "true:", truecoef,"eschers:", len(cores), uio_encod)
            print("diff")
            #print("conjecture:", goods, "true:", truecoef,"eschers:", len(cores), uio_encod)
        
        #print("conjectured coeff:", goods, "true coeff:", truecoef)

# M_(n+k),l U M_(n+l),k U M_n,k,l injection to --> P_n x M_k,l
def injection():
    # M_a,b is complement of image(phi_a,b) aka (a,b) cant come from a a+b escher by breaking it (not the same as impossible to concat them to an escher, try to find an example?)
    M_npk_l = 4

def tripplemaptest():
    # n+k, l
    A = generate_all_uios(n+k+l)
    totaluio = len(A)
    options = [(n,k,l),(k,l,n),(l,n,k)]
    random.shuffle(A)
    #A= [[0,0,0,0,1,3,5,6,6,8]]
    for uioid, encod in enumerate(A):
        uio = UIO(encod)
        # n =4, k = 4, l =2 -- > 6,4 and 8,2
        seen = []
        images = []
        for i, (a, b, c) in enumerate(options):
            if (a+b, c) in seen:
                continue
            seen.append((a+b, c))
            #print("seen:", seen)
            uio.setnlk(a,b,0)
            image = []
            pairs = uio.getEscherPairs(a+b,c)
            for apbescher, cescher in pairs:
                aescher,bescher = uio.phi(apbescher)
                if i == 0:
                    print("(aescher, bescher, cescher)", (aescher, bescher, cescher))
                    image.append((aescher, bescher, cescher))
                elif i == 1:
                    image.append((bescher, cescher, aescher))
                elif i == 2:
                    image.append((cescher, aescher, bescher))
            images.append(image)
        
        ## check if images are disjoint
        #print("check if images are disjoint...")
        Set = []
        for img in images:
            Set += img
        x = len(Set)
        y = len(set(Set))
        print(x, "vs", y, encod, uioid, "/", totaluio)
        if x != y:
            sys.exit(0)

def maptest():
    #A = [[0,0,0,2,3,4,4]]
    #A = [[0,0,1,1,2,3,3,5,6]]
    #A= [[0,0,0,0,1,3,5,6,6,8]]
    #A = generate_all_uios(n+k)
    #random.shuffle(A)
    #A = [[0, 0, 1, 1, 2, 3, 5]]
    #A = [[0, 0, 1, 1, 2, 3, 5]]
    A = [[0,0,1,1,2,3,3,5,7]] # 5,4 breaker 1   12.05 11:17
    for encod in A:
        uio = UIO(encod)
        P_npk = uio.getEscherPairs(n+k,0,0)
        P_n_k = uio.getEscherPairs(n,k,0)
        print(encod)
        print("n={}+k={} eschers: {}".format(n,k,len(P_n_k)))
        print("n+k={} eschers: {}".format(n+k,len(P_npk)))
        phiimage = []
        for v in P_npk:
            nescher, kescher = uio.phi(v)
            #u,w = uio.phi(v)
            back = uio.psi(nescher, kescher)
            phiimage.append((nescher, kescher))
            print(v, "-->", (nescher, kescher), "-->", back)
            if v != back:
                print(20*"diff!", encod)
                #print(v, "-->", (nescher, kescher), "-->", back)
        im = len(set(phiimage))
        #print("phi's image contains {} elements and phi's domain contains {} elements.".format(im,  len(P_npk)), "phi injective:", im == len(P_npk))
            

def trippleescher(): # and check 4,4,2
    A = generate_all_uios(n+k+l)
    A = [[0, 0, 0, 1, 2, 2, 5, 6]]
    random.shuffle(A)
    A= [[0,0,0,0,1,3,5,6,6,8]]
    for encod in A:
        uio = UIO(encod)
        print("here")
        truecoeff =  uio.getCoeffientByEscher(n,k,l)
        #print("here2")
        cores, goods = uio.getTripplEscherCores(n,k,l, verbose=True)
        print("true:", truecoeff, "conjectured:", goods, encod)
        if goods != truecoeff:
            print(20*"diff")

if __name__ == "__main__":
    global n,k,l,lcm,expo
    n = 4
    expo = 0
    k = 4
    npk = n+k
    l = 2
    lcm = np.lcm(n,k)
    #eschercoretest()
    #trippleescher()
    #maptest()
    #compareJAoutput("Joutput.txt", "Aoutput.txt")
    tripplemaptest()