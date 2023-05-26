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
import pandas as pd



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

def cyclicslice(tuple, start, end): # end exclusive
        #print("cyclic:", tuple, start, end)
        n = len(tuple)
        start = start%n
        end = end%n
        if start < end:
            return tuple[start:end]
        return tuple[start:]+tuple[:end]

def rewindescher(escher, steps): # moves the escher clockwise: (x,y,z) -> (z,x,y) is 1 step
        x = len(escher)
        return cyclicslice(escher, -steps, x-steps)

def setstartpoint(seq, startpoint):
    # startpoint is an element in seq
    while seq[0] != startpoint:
        seq = rewindescher(seq, 1)
    return seq


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
        self.repr = str(self.encoding)

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

    def __repr__(self):
        return self.repr

 #################### ESCHERS ###################

    #### COEFFICIENT ####

    def addupto(self, List, element, uptoindex):
        # adds element to List up to index uptoindex (exclusive)
        if len(List) == uptoindex:
            return List
        A = (uptoindex-len(List))*[element]
        return List + A

    def subuioencoding(self, seq):
        #print("seq:", seq)
        N = len(seq)
        encod = []
        for i in range(N):
            for j in range(i+1, N):
                #print(i, j, self.comparison_matrix[seq[i], seq[j]])
                if self.comparison_matrix[seq[i], seq[j]] != self.INCOMPARABLE: # not intersect
                    #print(i, "s up to (exclusive)", j)
                    encod = self.addupto(encod, i, j)
                    break
                elif j == N-1:
                    #print(i, "s up to (exclusive)", N, "final")
                    encod = self.addupto(encod, i, N)
        if self.comparison_matrix[seq[-1], seq[-2]] != self.INCOMPARABLE:
            encod.append(N-1)
        
        return encod

    def getCoeffientByEscher(self, n,k,l=0):
        if l == 0:
            return len(self.getEscherPairs((n,k))) - len(self.getEscherPairs((n+k,)))
        return 2*len(self.getEscherPairs((n+k+l,))) + len(self.getEscherPairs((n,k,l))) - len(self.getEscherPairs((n+l,k))) - len(self.getEscherPairs((n+k,l))) - len(self.getEscherPairs((l+k,n)))
    

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
    
    def concat(self, first, second, insertionpoint): # assume insertionpoint < len(first)
        # v0     vL vL+1
        # u0 ... uL uL+1
        #print("concat:", first, insertionpoint, first[:insertionpoint%n+1], self.cyclicslice(second, insertionpoint+1, insertionpoint+k+1), first[insertionpoint%n+1:])
        # (insertionpoint+1)+(k-1)+1 = insertionpoint+k+1
        f = len(first)
        s = len(second)
        return first[:insertionpoint%f+1]+cyclicslice(second, insertionpoint+1, insertionpoint+s+1)+first[insertionpoint%f+1:]

    def getEscherPairs(self, partition, verbose=False): # speedup
        if verbose:
            print("getEscherPairs:", "partition:", partition)

        if partition in self.escherpairs:
            return self.escherpairs[partition]

        pairs = []

        if len(partition) == 1:
            pairs = [seq for seq in getPermutationsOfN(partition[0]) if self.isescher(seq)]
        elif len(partition) == 2:
            n,k = partition
            for seq in getPermutationsOfN(n+k):
                if self.isescher(seq[:n]) and self.isescher(seq[n:]):
                    pairs.append((seq[:n], seq[n:]))
        else:
            n,k,l = partition
            for seq in getPermutationsOfN(n+k+l):
                if self.isescher(seq[:n]) and self.isescher(seq[n:n+k]) and self.isescher(seq[n+k:]):
                    pairs.append((seq[:n], seq[n:n+k], seq[n+k:]))

        self.escherpairs[partition] = pairs
        return self.escherpairs[partition]
    
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
        v = cyclicslice(u, L+1, L+n+1) # n-escher
        w = cyclicslice(u, L+n+1, L+n+k+1) # k-escher

        # We found the subeschers, but we have to change the starting point
        # for both subescher the startpoint should be s.t. the the L'th position (L can be 0) is the end of the original subescher
        
        v = rewindescher(v, L+1)
        w = rewindescher(w, L+1)
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
        v = cyclicslice(u, qk, qk+k)
        w = cyclicslice(u, qn, qn+n)

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
                w = rewindescher(w, 1)
        return w
        #if G < n-1:
            #return self.concat(u,v,G)
        #return self.concat(v,u,G)

    
class EscherBreaker:

    def __init__(self, uio:UIO, n, k): # n <= k
        self.uio = uio
        self.n = n
        self.k = k
        self.lcm = np.lcm(n,k)
        self.npk = n+k
        self.image = None
    
    """def getFirstInsertionPoint(self, u,v,lcm=0): 
        # the returned G can be bigger than len(u) and len(v)
        # doesn't matter if u <= v or v <= u
        for i in range(self.lcm):
            if self.comparison_matrix[u[i%self.n], v[(i+1)%self.k]] != UIO.GREATER and self.comparison_matrix[v[i%self.k], u[(i+1)%self.n]] != UIO.GREATER:
                return i
        return -1"""
    
    def getImage(self, verbose=False):
        P_npk = self.uio.getEscherPairs((self.n+self.k,))
        if verbose:
            print("get image:", self.uio)
            print("n+k={} eschers: {}".format(self.n+self.k,len(P_npk)))
        self.image = []
        for v in P_npk:
            nescher, kescher = self.map(v)
            self.image.append((nescher, kescher))
        return self.image
    
    def checkInjective(self):
        print("checkInjective")
        if self.image == None:
            self.getImage()
        seen = {}
        duplicates = []
        pairs = self.uio.getEscherPairs((self.n+self.k,))
        for (i, escher) in enumerate(self.image):
            if escher not in seen:
                seen[escher] = [i]
            else:
                duplicates.append(escher)
                seen[escher].append(i)

        for escher in duplicates:
            indices = seen[escher]
            print(15*"-")
            for i in indices:
                print(self.uio, pairs[i], "-->", self.map(pairs[i]))

    def checkLeftInverse(self):
        pairs = self.uio.getEscherPairs((self.n+self.k,))
        for u in pairs:
            b = self.map(u)
            c = self.inversemap(*b)
            if u != c:
                print(u, "-->", b, "-->", c, self.uio)
                sys.exit(0)

    def getcomplement(self, verbose=False):
        P_n_k = self.uio.getEscherPairs((self.n,self.k))
        if verbose:
            print("get complement:", self.uio)
            print("n={}+k={} eschers: {}".format(self.n,self.k,len(P_n_k)))
        complement = []
        phiimage = self.getImage(verbose)
        for (u,v) in P_n_k:
            if (u,v) not in phiimage:
                complement.append((u,v))
        return complement


class MyEscherBreaker(EscherBreaker):
    def __init__(self, uio, n, k, extra_v_offset=0, extra_w_offset=0):
        super().__init__(uio, n, k)
        self.extra_v_offset = extra_v_offset
        self.extra_w_offset = extra_w_offset

    def map(self, u, verbose = False):
        # return n-escher, k-escher

        # compute L, the point before the valid k-subescher^
        if verbose:
            print("phi", u, "n:", self.n, "k:", self.k)
        for L in range(0, self.n+self.k): # starts at 0, 0 indexing in the paper for "5. the proof"
            if self.uio.isarrow(u, L%self.npk, (L+self.n+1)%self.npk, verbose) and self.uio.isarrow(u, (L+self.n)%self.npk, (L+1)%self.npk, verbose):
                break

        # the k-subescher goes from L+1 to L+n
        # the n-subescher goes from L+n+1 to L+n+k
        v = cyclicslice(u, L+1, L+self.n+1) # n-escher
        w = cyclicslice(u, L+self.n+1, L+self.n+self.k+1) # k-escher

        # We found the subeschers, but we have to change the starting point
        # for both subescher the startpoint should be s.t. the the L'th position (L can be 0) is the end of the original subescher
        
        v = rewindescher(v, L+1+self.extra_v_offset) # v = rewindescher(v, L+1+2) vs v = rewindescher(v, L+1) for 1,2,3 on [0, 0, 0, 2, 2, 4]
        w = rewindescher(w, L+1+self.extra_w_offset)
        if verbose:
            print("L:", L)
        return (v, w) #(1, 0, 3, 6, 5, 4, 2)
    
    def inversemap(self, u, v): # n <= k
        #u - n-escher
        #v - k-escher
        G = self.uio.getFirstInsertionPoint(u,v)
        #print("G:", G)
        if G == -1:
            return None
        w = self.uio.concat(v,u, G)
        if G >= self.k:
            w = setstartpoint(w, u[self.k%self.n])
        return w
    
class PaperEscherBreaker(EscherBreaker):
    def map(self, u, verbose = False):
        # return n-escher, k-escher
        ZZ = False
        # compute L, the point before the valid k-subescher^
        if verbose:
            print("phi", u, "n:", self.n, "k:", self.k)
        for L in range(0, self.n+self.k): # starts at 0, 0 indexing in the paper for "5. the proof"
            if self.uio.isarrow(u, L%self.npk, (L+self.n+1)%self.npk, verbose) and self.uio.isarrow(u, (L+self.n)%self.npk, (L+1)%self.npk, verbose):
                break

        # the k-subescher goes from L+1 to L+n
        # the n-subescher goes from L+n+1 to L+n+k
        v = cyclicslice(u, L+1, L+self.n+1) # n-escher
        w = cyclicslice(u, L+self.n+1, L+self.n+self.k+1) # k-escher

        if ZZ:
            print("v1:", v)
            print("w1:", w)

        # We found the subeschers, but we have to change the starting point - like in the paper
        
        # Find n-multiple in [L+1:L+n]
        for qn in range(L+1, L+self.n+1):
            if qn%self.n == 0:
                break

        # Find k-multiple in [L+n+1:L] = [L+n+1:L+n+k-1] U [0:L]
        # for qn in range(L+n+1, L+n+k+1):
        #for qk in list(range(L+self.n+1, L+self.n+self.k)) + list(range(0, L+1)): 
        for qk in range(L+self.n+1, L+self.n+self.k+1):
            if qk%self.k == 0:
                break
        
        # k-escher goes from qk to qk+k-1 (inclusive)
        # n-escher goes from qk to qn+n-1 (inclusive)
        v = setstartpoint(v, u[qn%self.npk])
        w = setstartpoint(w, u[qk%self.npk])
        #v = cyclicslice(u, qn, qn+self.n)
        #w = cyclicslice(u, qk, qk+self.k)

        if ZZ:
            print("qn:", qn)
            print("qk:", qk)
            print("v2:", v)
            print("w2:", w)

        if verbose:
            print("L:", L)
        return (v, w) 
    
    def inversemap(self, u, v): # n <= k
        #u - n-escher
        #v - k-escher
        G = self.uio.getFirstInsertionPoint(u,v)
        #print("G:", G)
        if G == -1:
            return None
        w = self.uio.concat(v,u, G)
        if G >= self.k:
            w = setstartpoint(w, u[self.k%self.n])
        return w
    

class AugmentedEscherBreaker:
    def __init__(self, uio, n, k, l):
        self.uio = uio
        self.n =n 
        self.k = k
        self.l = l

    def do(self, p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62):
        """ # Always take k-subescher s.t. k is the smallest number, but still not injective
        n = 2
        k = 3
        l = 4
        graph1 = []
        phi_2p3_4 = MyEscherBreaker(self.uio, 4, 5,  p11, p12)
        phi_2_3 = MyEscherBreaker(self.uio, 2,3,  p21, p22)
        for (w, uv) in phi_2p3_4.getcomplement():
            u,v = phi_2_3.map(uv)
            graph1.append(((uv, w), (u,v,w)))
        
        graph2 = []
        phi_3p4_2 = MyEscherBreaker(self.uio, 2,7,  p31, p32)
        phi_3_4 = MyEscherBreaker(self.uio, 3,4,  p41, p42)
        for (u, vw) in phi_3p4_2.getcomplement():
            v,w = phi_3_4.map(vw)
            graph2.append(((vw, u), (u,v,w)))

        graph3 = []
        phi_4p2_3 = MyEscherBreaker(self.uio, 3, 6,  p51, p52)
        phi_4_2 = MyEscherBreaker(self.uio, 2,4,  p61, p62)
        for (v, uw) in phi_4p2_3.getcomplement():
            u,w = phi_4_2.map(uw)
            graph3.append(((uw, v), (u,v,w)))
        """

        """
        graph1 = []
        phi_2p3_4 = MyEscherBreaker(self.uio, 2+3, 4)
        phi_2_3 = MyEscherBreaker(self.uio, 2,3)
        for (uv, w) in phi_2p3_4.getcomplement():
            u,v = phi_2_3.map(uv)
            graph1.append(((uv, w), (u,v,w)))
        
        graph2 = []
        phi_3p4_2 = MyEscherBreaker(self.uio, 3+4, 2)
        phi_3_4 = MyEscherBreaker(self.uio, 3,4)
        for (vw, u) in phi_3p4_2.getcomplement():
            v,w = phi_3_4.map(vw)
            graph2.append(((vw, u), (u,v,w)))

        graph3 = []
        phi_4p2_3 = MyEscherBreaker(self.uio, 4+2, 3)
        phi_4_2 = MyEscherBreaker(self.uio, 4,2)
        for (uw, v) in phi_4p2_3.getcomplement():
            u,w = phi_4_2.map(uw)
            graph3.append(((uw, v), (u,v,w)))
        """
         # maybe good? 3 3 1 1 3 1 0 0 0 0 3 1

        embeddings = []
        """
        graph1 = []
        phi_1p2_3 = MyEscherBreaker(self.uio, 1+2, 3, p11, p12)
        phi_1_2 = MyEscherBreaker(self.uio, 1,2, p21, p22)
        for (uv, w) in phi_1p2_3.getcomplement():
            u,v = phi_1_2.map(uv)
            #graph1.append(((uv, w), (u,v,w)))
            embeddings.append((u,v,w))
        
        graph2 = []
        phi_2p3_1 = MyEscherBreaker(self.uio, 1, 2+3, p31, p32)
        phi_2_3 = MyEscherBreaker(self.uio, 2,3, p41, p42)
        for (u, vw) in phi_2p3_1.getcomplement():
            v,w = phi_2_3.map(vw)
            #graph2.append(((vw, u), (u,v,w)))
            embeddings.append((u,v,w))

        graph3 = []
        phi_3p1_2 = MyEscherBreaker(self.uio, 2, 3+1, p51, p52)
        phi_3_1 = MyEscherBreaker(self.uio, 1,3, p61, p62)
        for (v, uw) in phi_3p1_2.getcomplement():
            u, w = phi_3_1.map(uw)
            #graph3.append(((uw, v), (u,v,w)))
            embeddings.append((u,v,w))"""

        graph1 = []
        phi_npk_l = MyEscherBreaker(self.uio, self.n+self.k, self.l, p11, p12)
        phi_n_k = MyEscherBreaker(self.uio, self.n,self.k, p21, p22)
        for (uv, w) in phi_npk_l.getcomplement():
            u,v = phi_n_k.map(uv)
            #graph1.append(((uv, w), (u,v,w)))
            embeddings.append((u,v,w))
        
        graph2 = []
        phi_kpl_n = MyEscherBreaker(self.uio, self.k+self.l, self.n, p31, p32)
        phi_k_l = MyEscherBreaker(self.uio, self.k,self.l, p41, p42)
        for (vw, u) in phi_kpl_n.getcomplement():
            v,w = phi_k_l.map(vw)
            #graph2.append(((vw, u), (u,v,w)))
            embeddings.append((u,v,w))

        graph3 = []
        phi_lpn_k = MyEscherBreaker(self.uio, self.l+self.n, self.k, p51, p52)
        phi_l_n = MyEscherBreaker(self.uio, self.l,self.n, p61, p62)
        for (uw, v) in phi_lpn_k.getcomplement():
            w, u = phi_l_n.map(uw)
            #graph3.append(((uw, v), (u,v,w)))
            embeddings.append((u,v,w))

        #if len(graph1) > 0 and len(graph2) > 0 and len(graph3) > 0:
        #    print(graph1[0], graph2[0], graph3[0])
        #print(len(embeddings), len(set(embeddings)))
        return len(embeddings) == len(set(embeddings))

        return self.compareGraphs(graph1, graph2, graph3)

    def compareGraphs(self, G1, G2, G3):
        triplets = []
        for G in [G1,G2,G3]:
            for x in G:
                triplets.append(x[1])
        A = len(triplets)
        B = len(set(triplets))
        #print(self.uio, "triplets:", len(triplets), "vs", B)

        return A == B



def tripplemaptest():
    # n+k, l
    n = 1
    k = 2
    l= 3
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
            #pairs = uio.getEscherPairs((a+b,c), verbose=True)
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
        """pairs = uio.getEscherPairs((n+k,l), verbose=True)
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
            #pairs = uio.getEscherPairs((a+b,c), verbose=True)
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
    P_npk = uio.getEscherPairs((n+k,0,0))
    P_n_k = uio.getEscherPairs((n,k,0), True)
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
        P_npk = uio.getEscherPairs((n+k,0,0))
        P_n_k = uio.getEscherPairs((n,k,0))
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
        print(uioid+1, "/", total, encod, "phi's image contains {} elements and phi's domain contains {} elements.".format(im,  len(P_npk)), "phi injective:", isinj)
        if isinj == False:
            sys.exit(0)
            
def v2doubletest():
    k = 4
    n = 3
    A = generate_all_uios(n+k)
    #A = [[0, 0, 0, 0, 0, 1, 1, 4, 5]]
    #A = [[0,0,0,0,2,4,5]]
    random.shuffle(A)
    #A = [[0,0,0,0,0,0,0]]
    total = len(A)
    for uioid, encod in enumerate(A):
        uio = UIO(encod)
        phi = MyEscherBreaker(uio, n, k)
        imagesize = len(set(phi.getImage()))
        domainsize = len(uio.getEscherPairs((n+k,)))
        phi.checkLeftInverse()
        if imagesize != domainsize:
            phi.checkInjective()
        print(uioid+1, "/", total, "image:", imagesize, "domain:", domainsize, uio)
        if imagesize != domainsize:
            sys.exit(0)

def v3trippletest():
    n = 2
    k = 3
    l = 4
    A = generate_all_uios(n+k+l)
    #A = [[0, 0, 0, 2, 2, 4]]
    A = [[0, 0, 0, 0, 2, 2, 4, 4, 6]]
    #A = [[0, 0, 0, 0, 0, 1, 1, 4, 5]]
    random.shuffle(A)
    for uioid, encod in enumerate(A):
        uio = UIO(encod)
        AEB = AugmentedEscherBreaker(uio, n, k, l)
        if AEB.do() == False:
            return
    print("done it!")
    sys.exit(0)

def v3trippletestparameters():
    n = 1
    k = 2
    l = 3
    A = generate_all_uios(n+k+l)
    A = [[0, 0, 0, 1, 2, 3]]
    A = [[0,0,1,2,3,4]]
    
    random.shuffle(A)
    uios = []
    for encod in A:
        uio = UIO(encod)
        if len(uio.getEscherPairs((n,k,l))) > 0:
            uios.append(encod)
    print(len(uios), "uios with non-zero P_n_k_l. from in total", len(A), "uios")
    iteration = 0
    for uio in uios:
        if checkall(uio,n,k,l) == False:
            print(uio, "is out")
        else:
            print(uio, "is good")

def checkall(uio,n,k,l):
    #for i in range(4000):
    for p11 in range(n+k):
        for p12 in range(l):
            for p21 in range(n):
                for p22 in range(k):

                    for p31 in range(k+l):
                        for p32 in range(n):
                            for p41 in range(k):
                                for p42 in range(l):

                                    for p51 in range(l+n):
                                        for p52 in range(k):
                                            for p61 in range(l):
                                                for p62 in range(n):
                                                    if tripplecheckparameters(n,k,l,[uio], p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62) == True:
                                                        return True
        """
        p11 = random.randrange(n+k)
        p12 = random.randrange(l)
        p21 = random.randrange(n)
        p22 = random.randrange(k)

        p31 = random.randrange(k+l)
        p32 = random.randrange(n)
        p41 = random.randrange(k)
        p42 = random.randrange(l)

        p51 = random.randrange(l+n)
        p52 = random.randrange(k)
        p61 = random.randrange(l)
        p62 = random.randrange(n)"""
        #p11 = p12 = p31 = p32 = p51 = p52 = 0
        
    return False

def tripplecheckparameters(n,k,l,uios, p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62):
    global bestscore, count
    total = len(uios)
    #print(p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62)
    for uioid, encod in enumerate(uios):
        uio = UIO(encod)
        AEB = AugmentedEscherBreaker(uio, n, k, l)
        #print(uioid, "/", total)
        if AEB.do(p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62) == False:
            if bestscore == None or uioid > bestscore[0]:
                bestscore = (uioid, p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62)
            print("failed at", uioid, count, "params:", p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62)
            count += 1
            return False
    return True
    print(total, "done it!", p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62)
    #sys.exit(0)

def newconjecturetest():
    n = 1
    k = 2
    l = 3
    A = generate_all_uios(n+k+l)
    A = [[0,0,1,2,3,4]]
    for encod in A:
        uio  = UIO(encod)
        pairs = uio.getEscherPairs((n,k,l))
        # (210) (4 3) (5)

        counter = 0
        #for (u,v,w) in pairs:
        for (u,v,w) in [( (5,), (4,3), (2,1,0) )]:
            E_uv = uio.subuioencoding(sorted(u+v))
            E_vw = uio.subuioencoding(sorted(v+w))
            E_uw = uio.subuioencoding(sorted(u+w))
            print("encod:", E_uv)
            print("encod:", E_vw)
            print("encod:", E_uw)
            
            phi_nk = MyEscherBreaker(UIO(E_uv), n, k)
            phi_kl = MyEscherBreaker(UIO(E_vw), k, l)
            phi_nl = MyEscherBreaker(UIO(E_uw), n, l)
            M_nk = phi_nk.getcomplement()
            M_kl = phi_kl.getcomplement()
            M_nl = phi_nl.getcomplement()
            print(M_nk, len(M_nk))
            print(M_kl, len(M_kl))
            print(M_nl, len(M_nl))

            print(u,v,w)
            if (u,v) in M_nk and (u,w) in M_nl and (v,w) in M_kl:
                print(u,v,w)
                counter += 1
        print(uio, counter, uio.getCoeffientByEscher(n,k,l))



def trippletable():
    n = 1
    k = 2
    l = 3
    A = generate_all_uios(n+k+l)
    #A = [[0,0,1,1,1,2,2,2,3]]

    data = []
    total = len(A)
    for uioid,encod in enumerate(A):
        uio = UIO(encod)
        p_npk_l = MyEscherBreaker(uio, n+k, l)
        p_kpl_n = MyEscherBreaker(uio, k+l, n)
        p_lpn_k = MyEscherBreaker(uio, l+n, k)
        pairs = uio.getEscherPairs((n,k,l))
        pairs2 = uio.getEscherPairs((n+k+l,))

        data.append([encod, 
        len(p_npk_l.getcomplement()),
        len(p_kpl_n.getcomplement()),
        len(p_lpn_k.getcomplement()),
        len(pairs2), 
        len(pairs)])
        print(uioid+1, "/", total)
    df = pd.DataFrame(data, columns = ['UIO', 'M_nk_l', 'M_kl_n', 'M_ln_k', 'P_nkl', 'P_n_k_l'])
    print(df.to_string())

def subencodingtest():
    encod = [0,0,1,2,3]
    uio = UIO(encod)
    seq = (0,1,2,3,4) # (1,3,4)
    subencod = uio.subuioencoding(seq)
    print(subencod)


if __name__ == "__main__": # 2 case: n>=k, 3 case: n<=k<=L
    global bestscore, count
    bestscore = None
    count = 0
    #maptest()
    #tripplemaptest()
    #complementtest()
    #v2doubletest()
    #trippletable()
    #v3trippletestparameters()
    newconjecturetest()
    #subencodingtest()

# 16.05 todo: check tipple injective, make proof for injective in double case
# proof that he coefficient is equal to the difference of thee eschers sets
# phi_n,k (paper notation) but where n < k?

# print out table
# phi maps depending on UIO?
# injective if use phi_0,0 for Ms?
# find tough UIO for 1,2,3 - [0,0,1,2,3,4] (find subescher by syntax not by smallest one)
# no shift at all


"""
Put down what you know, easy to get lost in things one still doesn't understand
a draft asap
short introduction of the world, setup, some important facts, then describe what I did/present results/workflow
rephrase papers like the escher one
later on still time to mention proofs (like number of escher equals number of correct sequence)
guess the external censors will not look at all the details 100%
"""