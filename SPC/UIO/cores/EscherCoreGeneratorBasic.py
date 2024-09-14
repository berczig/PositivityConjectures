from SPC.UIO.cores.EscherCoreGeneratorAbstract import EscherCoreGeneratorAbstract
from SPC.UIO.UIO import UIO
import numpy as np

class EscherCoreGeneratorBasic(EscherCoreGeneratorAbstract):

    def compareTwoCoreElements(self, a, b):
        if a < b:
            return UIO.LESS
        return UIO.GREATER
    
        # if a < b:
        #     return UIO.LESS
        # elif a > b:
        #     return UIO.GREATER
        # return UIO.EQUAL
    
    @staticmethod
    def getCoreLabels(partition):
        return ["0", "subescher start interval", "subescher end interval", "1.insert"]#, "n-1"]
    
    @staticmethod
    def getCoreComparisions(partition):
        #return {"0":["subescher start interval", "subescher end interval", "1.insert"],
        #        "subescher start interval":["1.insert", "n-1"], 
        #        "subescher end interval":["1.insert", "n-1"],
        #        "1.insert":["n-1"]}
        return {"0": ["subescher start interval","1.insert"],
                "subescher end interval":["1.insert"],}
    

    def generateCore(self, escherpair):
        n,k = self.partition  # maybe have to switch n and k
        u,v = escherpair
        core = self.getInsertionsSubeshers(u,v)
        insertions, escherstartpoints = core
        if len(insertions) == 0:
            #return "GOOD"
            return [0,-1,-1,-0.5]
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
                points.append(k-2)
            points.append(insertions[0]+0.5) # here 2(n+k)>n+k, but any such number will be fine

            #if len(insertions) > 1:
            #    points.append(insertions[1]-0.5) # here 2(n+k)>n+k, but any such number will be fine
            #else:
            #    points.append(999)

            #points.append(n-1)
            #points.append(n+k-1)
        return points
    
    """ def getCoreRepresentation(self, core):
        k = len(core)
        if k == 0:
            return ()
        comparison_matrix = np.zeros((k,k)) + UIO.EQUAL # (i,j)'th index says how i is in relation to j
        for i in range(k):
            for j in range(i+1,k):
                if core[i] < core[j]:
                    comparison_matrix[i, j] = UIO.LESS
                    comparison_matrix[j,i] = UIO.GREATER
                elif core[i] > core[j]:
                    comparison_matrix[i, j] = UIO.GREATER
                    comparison_matrix[j,i] = UIO.LESS
        return tuple([comparison_matrix[i,j] for i in range(k) for j in range(i+1, k)]) """
    
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

    

