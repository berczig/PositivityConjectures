from SPC.Restructure.cores.EscherCoreGeneratorAbstract import EscherCoreGeneratorAbstract
from SPC.Restructure.UIO import UIO
import numpy as np
import sys
class EscherCoreGeneratorQuadruple(EscherCoreGeneratorAbstract):

    def compareTwoCoreElements(self, a, b):
        if a < b:
            return UIO.LESS
        return UIO.GREATER
    

    @staticmethod
    def getCoreComparisions(partition):
        return {
            "a" : ["b", "c"],
            "b" : ["c"]}

    @staticmethod
    def getCoreLabels(partition):
        return ["a", "b", "c"]

    def generateCore(self, escherquadruple):
        
        a,b,c,d = escherquadruple
        return (a[0], b[0], c[0])

    def get_insertion_and_subescher_of_2_eschers(self, u,v):
        n,k = len(u), len(v)
        core = self.getInsertionsSubeshers(u,v)
        insertions, escherstartpoints = core

        if len(insertions) == 0:
            return [n-1, -1, -1, -1]
        else:
            points = [n-1]
            has_bigger_than_0_startpoint = False
            for escherstartpoint in escherstartpoints:
                if escherstartpoint > 0:
                    points.append(escherstartpoint)
                    points.append(escherstartpoint+k-1)
                    has_bigger_than_0_startpoint = True
                    break
            if not has_bigger_than_0_startpoint:
                points.append(-1)
                points.append(-2+k)
            points.append(insertions[0]) # here 2(n+k)>n+k, but any such number will be fine,  "insertion + 0.5 is good"
        return points


    def get_short_insertion_and_subescher_of_2_eschers(self, u,v): #core = (escherstartpoint, esherendpoint,insertion)
        n,k = len(u), len(v)
        core = self.getInsertionsSubeshers(u,v)
        insertions, escherstartpoints = core

        if len(insertions) == 0:
            return [-1, -1, -1]
        else:
            points = []
            has_bigger_than_0_startpoint = False
            for escherstartpoint in escherstartpoints:
                if escherstartpoint > 0:
                    points.append(escherstartpoint)
                    points.append(escherstartpoint+k-1)
                    has_bigger_than_0_startpoint = True
                    break
            if not has_bigger_than_0_startpoint:
                points.append(-1)
                points.append(k-2)
            points.append(insertions[0]) # here 2(n+k)>n+k, but any such number will be fine,  "insertion + 0.5 is good"
        return points
    
    def get_shortb_insertion_and_subescher_of_2_eschers(self, u,v): 
        #Let uu be the longer, vv the shorter escher
        # n,k = max(len(u),len(v)), min(len(u),len(v))
        # if len(u) == n:
        #     uu = u
        #     vv = v
        # else:
        #     uu = v
        #     vv = u
        # core = self.getInsertionsSubeshers(uu,vv)
        n,k = len(u), len(v)
        core = self.getInsertionsSubeshers(u,v)
        insertions, escherstartpoints = core
        if len(insertions) == 0:
            return [-1, -1, -1]
        else:
            points = []
            has_bigger_than_0_startpoint = False
            for escherstartpoint in escherstartpoints:
                if escherstartpoint > 0:
                    points.append(escherstartpoint)
                    points.append(escherstartpoint+k-1)
                    has_bigger_than_0_startpoint = True
                    break
            if not has_bigger_than_0_startpoint:
                points.append(-1)
                points.append(insertions[0]+1)
            points.append(insertions[0])
            # if len(escherstartpoints) == 0:
            #     points.append(-1)
            #     points.append(-1)
            # else:
            #     points.append(escherstartpoints[0])
            #     points.append(escherstartpoints[0]+k-1)
            # points.append(insertions[0])
        return points

    
    
    def generateCore_2partition(self, escherpair):
        n,k = self.partition  # maybe have to switch n and k
        u,v = escherpair
        core = self.getInsertionsSubeshers(u,v)
        insertions, escherstartpoints = core
        if len(insertions) == 0:
            return []
        #if escherstartpoints == []:
        #    print("Problem with core:",self.encoding,u,v,core)
        #print(u,v, "insertions:", insertions, "escherstartpoints:", escherstartpoints)
        #print(points)
        else:
            points = [0]
            has_bigger_than_0_startpoint = False
            for escherstartpoint in escherstartpoints:
                if escherstartpoint > 0:
                    points.append(escherstartpoint)
                    points.append(escherstartpoint+k-1)
                    has_bigger_than_0_startpoint = True
                    break
            if not has_bigger_than_0_startpoint:
                points.append(-1)
                points.append(-2+k)
            points.append(insertions[0]) # here 2(n+k)>n+k, but any such number will be fine

            #if len(insertions) > 1:
            #    points.append(insertions[1]-0.5) # here 2(n+k)>n+k, but any such number will be fine
            #else:
            #    points.append(999)

            points.append(n-1)
            #points.append(n+k-1)
        return points
    
    def concat(self, first, second, insertionpoint): # assume insertionpoint < len(first)
        # v0     vL vL+1
        # u0 ... uL uL+1
        #print("concat:", first, insertionpoint, first[:insertionpoint%n+1], self.cyclicslice(second, insertionpoint+1, insertionpoint+k+1), first[insertionpoint%n+1:])
        # (insertionpoint+1)+(k-1)+1 = insertionpoint+k+1
        #print("first:", first)
        #print("second:", second)
        #print("insertionpoint:", insertionpoint)
        f = len(first)
        k = len(second)
        return first[:insertionpoint%f+1]+self.cyclicslice(second, insertionpoint+1, insertionpoint+k+1)+first[insertionpoint%f+1:]
    
    def getInsertionsSubeshers(self, u, v): # u is length n, v is length k
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

    def getInsertionPoints(self, u, v, lcm = 0): # u of length n > k
        n = len(u)
        k = len(v)
        if lcm == 0:
            lcm = np.lcm(n,k)
        points = []
        for i in range(lcm):
            if self.uio.comparison_matrix[u[i%n], v[(i+1)%k]] != UIO.GREATER and self.uio.comparison_matrix[v[i%k], u[(i+1)%n]] != UIO.GREATER:
                points.append(i)
        #print("found {} insertion points between {} and {}".format(len(points),u,v))
        return points
    
    def cyclicslice(self, tuple, start, end): # end exclusive
        #print("tuple:", tuple, start, end)
        n = len(tuple)
        start = start%n
        end = end%n
        if start < end:
            return tuple[start:end]
        elif start == end:
            return tuple
        return tuple[start:]+tuple[:end]
    
    def getpseudosubescherstartingpoints(self, escher, k): # k > 0, pretends the input is an escher and finds valid k-subescher 
        subeschersstartingpoint = [] # start of the box
        h = len(escher)
        #print("escher length:", h)
        for m in range(-1, h-1): # m can be 0, m is before the start of the box mBBBB#
            cond1 = self.uio.isarrow(escher, (m+k)%h, (m+1)%h)
            cond2 = self.uio.isarrow(escher, m%h, (m+k+1)%h)
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
