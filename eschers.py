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


#####
   

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

 #################### ESCHERS ###################

    def setnlk(self, n_, k_, l_, verbose=False):
        global n,k,l, npk, lcm
        n = n_
        k = k_
        l = l_
        npk = n+k
        if verbose:
            print("setnlk", "n:", n_, "k:", k_, "l:", l_, npk)
        lcm = np.lcm(n,k)

    def isescher(self, seq):
        for i in range(len(seq)-1):
            if self.isarrow(seq, i, i+1) == False:
                return False
        return self.isarrow(seq, -1, 0)

    def isarrow(self, escher, i,j, verbose=False): # 0 <= i,j < len(escher)
        if verbose:
            print("arrow", escher, i, j, self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER)
        return self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER # EQUAL also intersects

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

    def getFirstInsertionPoint(self, u,v,lcm=0): 
        # the returned G can be bigger than len(u) and len(v)
        # doesn't matter if u <= v or v <= u
        n = len(u)
        k = len(v)
        if lcm == 0:
            lcm = np.lcm(n,k)
        for i in range(lcm):
            if self.comparison_matrix[u[i%n], v[(i+1)%k]] != UIO.GREATER and self.comparison_matrix[v[i%k], u[(i+1)%n]] != UIO.GREATER:
                return i
        return -1

    def getEscherPairs(self, n,k=0,l=0, verbose=False): # speedup
        if verbose:
            print("getEscherPairs:", "n:", n,"k:", k,"l:", l)
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
                
    ###### MAP P_{n+k} -> P_{n,k} ################
    def phi(self, u, verbose=False): # n <= k
        # return n-escher, k-escher
        # compute L, the point before the valid k-subescher^
        if verbose:
            print("phi", u, "n:", n, "k:", k, "l:", l)
        for L in range(0, n+k): # starts at 0, 0 indexing in the paper for "5. the proof"
            if self.isarrow(u, L%npk, (L+n+1)%npk, verbose) and self.isarrow(u, (L+n)%npk, (L+1)%npk, verbose):
                break

        # the k-subescher goes from L+1 to L+n
        # the n-subescher goes from L+n+1 to L+n+k
        v = self.cyclicslice(u, L+1, L+n+1) # n-escher
        w = self.cyclicslice(u, L+n+1, L+n+k+1) # k-escher

        # We found the subeschers, but we have to change the starting point
        # for both subescher the startpoint should be s.t. the the L'th position (L can be 0) is the end of the original subescher
        
        v = self.rewindescher(v, L+1)
        w = self.rewindescher(w, L+1)
        if verbose:
            print("L:", L)
        return (v, w) #(1, 0, 3, 6, 5, 4, 2)
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

    def psi(self, u, v): # n <= k
        #u - n-escher
        #v - k-escher
        G = self.getFirstInsertionPoint(u,v)
        #print("G:", G)
        if G == -1:
            return None
        w = self.concat(u,v, G)
        if G >= k:
            #print(12*"EXP")
            v_n = v[k%n]
            while w[0] != v_n:
                w = self.rewindescher(w, 1)
        return w
        #if G < n-1:
            #return self.concat(u,v,G)
        #return self.concat(v,u,G)

    def rewindescher(self, escher, steps): # moves the escher clockwise
        x = len(escher)
        return self.cyclicslice(escher, -steps, x-steps)
        

def tripplemaptest():
    # n+k, l
    A = generate_all_uios(n+k+l)
    A = [[0, 0, 1, 1, 2, 4]]
    realn = n
    realk = k
    reall = l
    #A = [[0, 0, 0, 0, 1, 1, 4, 5, 5]] # 2,3,4 breaker, not injective
    totaluio = len(A)
    options = [(n,k, l),(k,l, n),(n,l, k)]
    random.shuffle(A)
    #A= [[0,0,0,0,1,3,5,6,6,8]]
    for uioid, encod in enumerate(A):
        uio = UIO(encod)
        # n =4, k = 4, l =2 -- > 6,4 and 8,2
        seen = []
        images = []
        print()
        for i, (a, b, c) in enumerate(options): # a <= b, c fit anywhere, i.e. could be b <= c or  c <= a
            #if (a+b, c) in seen:
                #continue
            #n_ = min(a,b)       # with convention n_ <= k_
            #k_ = max(a,b)
            seen.append((a+b, c))
            print("seen:", seen)
            fn = min(a+b, c)
            fk = max(a+b, c)
            print("a+b:", a,b, a+b)
            print("c:", c)
            print("fn:", fn)
            print("fk:", fk)
            uio.setnlk(fn, fk, 0, verbose=True) # to be able to use phi_(n+k),l
            image = []
            complement = getcomplement(uio) # complement of M_a+b,c
            uio.setnlk(a,b,0, verbose=True) # to be able to use phi_n,k
            #pairs = uio.getEscherPairs(a+b,c, verbose=True)
            for cescher, apbescher in complement:
                aescher,bescher = uio.phi(apbescher)
                #image.append((aescher, bescher, cescher))
                
                if i == 0:
                    #print("(aescher, bescher, cescher)", (aescher, bescher, cescher))
                    res = (aescher, bescher, cescher)
                elif i == 1:
                    res = (cescher, aescher, bescher)
                elif i == 2:
                    res = (aescher, cescher, bescher)
                image.append(res)
                print(i, apbescher, cescher, "-->", res)
            print("img:", len(image), (None if len(image) == 0 else image[0]))
            images.append(image)
        
        ## check if images are disjoint
        #print("check if images are disjoint...")
        """pairs = uio.getEscherPairs(n+k,l, verbose=True)
        nklimage = images[0]
        klnimage = images[1]
        for i in range(len(pairs)):
            print(pairs[i])
            npkescher, lescher = pairs[i]
            if nklimage[i] in klnimage:
                print("problem:", npkescher, lescher, "-->", nklimage[i], "but also in klnimage", )"""
        Set = []
        for img in images:
            Set += img
        x = len(Set)
        y = len(set(Set))
        print(x, "vs", y, encod, uioid, "/", totaluio)
        A1 = images[1]
        A2 = images[0]
        if x != y:

            a,b,c = options[2]
            uio.setnlk(min(a+b,c), max(a+b, c), 0, verbose=True) # to be able to use phi_(n+k),l
            image = []
            complement = getcomplement(uio, False) # complement of M_a+b,c
            uio.setnlk(a,b,0, verbose=True) # to be able to use phi_n,k
            #pairs = uio.getEscherPairs(a+b,c, verbose=True)
            for cescher, apbescher in complement:
                aescher,bescher = uio.phi(apbescher)
                #image.append((aescher, bescher, cescher))
                res = None
                if i == 0:
                    #print("(aescher, bescher, cescher)", (aescher, bescher, cescher))
                    res = (aescher, bescher, cescher)
                elif i == 1:
                    res = (cescher, aescher, bescher)
                elif i == 2:
                    res = (aescher, cescher, bescher)
                
                if res in A1:
                    print( apbescher,cescher, "-->", res, "in k+l,n")
                
                if res in A2:
                    print( apbescher,cescher, "-->", res, "in n+k,l")

            sys.exit(0)

def getcomplement(uio:UIO, verbose=False):
    P_npk = uio.getEscherPairs(n+k,0,0)
    P_n_k = uio.getEscherPairs(n,k,0, True)
    if verbose:
        print("get complement:", uio)
        print("n={}+k={} eschers: {}".format(n,k,len(P_n_k)))
        print("n+k={} eschers: {}".format(n+k,len(P_npk)))
    complement = []
    phiimage = []
    for v in P_npk:
        nescher, kescher = uio.phi(v)
        phiimage.append((nescher, kescher))
    for (u,v) in P_n_k:
        if (u,v) not in phiimage:
            complement.append((u,v))
    return complement

        
def setdifferrence(A,B): # A \ B
    return [x for x in A if x not in B]

def complementtest():
    A = [
        [0,0,1,1,2,3,3,5,7],  
        [0,0,1,1,2,3,3,5,6],
        [0,0,1,1,2,3,3,6,6],
        [0,0,1,1,2,3,3,6,7],
        [0,0,1,1,2,3,4,5,7]]
    for encod in A:
        print("uio:", encod)
        uio = UIO(encod)
        complement = getcomplement(uio)
        goods, cores = getgoods(uio)
        ncomp = len(complement)
        ngoods = len(goods)
        print("complement:", ncomp)
        case1 = 0 
        case2 = 0
        case3 = 0  
        for core in cores:
            if core[1] == -1:
                case1 += 1
            elif core[0] != -1:
                case2 += 1
            else:
                print("case3:", core)
                case3 += 1
        print("goods: {} = {} + {} + {}".format(ngoods, case1, case2,case3))
        falsegoods = setdifferrence(goods, complement)
        if len(falsegoods) == 0:
            print("- all cores that are goods are in the complement")
        else:
            print("- goods that shouldn't be good:")
            for x in falsegoods:
                print(x, uio.getEscherRGCoreExperimental(*x))

        missinggoods = setdifferrence(complement, goods)
        if len(missinggoods) == 0:
            print("- The complement is part of our goods")
        else:
            print("- goods we are missing:")
            for x in missinggoods:
                print(x, uio.getEscherRGCoreExperimental(*x))

def maptest():
    #A = [[0,0,0,2,3,4,4]]
    #A = [[0,0,1,1,2,3,3,5,6]]
    #A= [[0,0,0,0,1,3,5,6,6,8]]
    A = generate_all_uios(n+k)
    A = [[0,0,1]]
    random.shuffle(A)
    #A = [[0, 0, 1, 1, 2, 3, 5]]
    #A = [[0, 0, 1, 1, 2, 3, 5]]
    total = len(A)
    #A = [[0,0,1,1,2,3,3,5,7]] # 5,4 breaker 1   12.05 11:17
    for uioid, encod in enumerate(A):
        uio = UIO(encod)
        P_npk = uio.getEscherPairs(n+k,0,0)
        P_n_k = uio.getEscherPairs(n,k,0)
        #print(encod)
        #print("n={}+k={} eschers: {}".format(n,k,len(P_n_k)))
        #print("n+k={} eschers: {}".format(n+k,len(P_npk)))
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
        isinj = im == len(P_npk)
        print(uioid, "/", total, encod, "phi's image contains {} elements and phi's domain contains {} elements.".format(im,  len(P_npk)), "phi injective:", isinj)
        if isinj == False:
            sys.exit(0)
            

if __name__ == "__main__": # 2 case: n>=k, 3 case: n<=k<=L
    #maptest()
    tripplemaptest()
    #complementtest()

# 16.05 todo: check tipple injective, make proof for injective in double case
# proof that he coefficient is equal to the difference of thee eschers sets
# phi_n,k (paper notation) but where n < k?