from SPC.UIO.cores.EscherCoreGeneratorAbstract import EscherCoreGeneratorAbstract
from SPC.UIO.UIO import UIO
import numpy as np
import sys
class EscherCoreGeneratorTrippleSymmetricNoEqual(EscherCoreGeneratorAbstract):

    def compareTwoCoreElements(self, a, b):
        if a < b:
            return UIO.LESS
        return UIO.GREATER
    

    @staticmethod
    def getCoreComparisions(partition):
        #return EscherCoreGeneratorTrippleSymmetric.getAllCoreComparisions(partition)
        #return {
        #    "0" : ["subescher vw start", "vw 1. insert", "subescher uv_w start", "uv_w 1. insert", "subescher uw_v start", "uw_v 1. insert"],
        #    "len(v)-1" : ["subescher vw start", "subescher vw end", "vw 1. insert", "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "subescher vw start" : ["vw 1. insert", "len(uv)_1", "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "subescher vw end" : ["vw 1. insert", "len(uv)_1", "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "vw 1. insert" : ["len(uv)_1", "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "len(uv)_1" : ["subescher uv_w start", "subescher uv_w end", "uv_w 1. insert", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "subescher uv_w start" : ["uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "subescher uv_w end" : ["uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "uv_w 1. insert" : ["len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "len(uw)-1" : ["subescher uw_v start", "subescher uw_v end", "uw_v 1. insert"],
        #    "subescher uw_v start" : ["uw_v 1. insert"],
        #    "subescher uw_v end" : ["uw_v 1. insert"],}
        # Reduced edges
        #return {
        #    "0" : ["subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start"],#, "uv 1. insert","uw 1. insert", "vw 1. insert", "uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #    "len(v)-1" : ["vw 1. insert"],# "uv_w 1. insert", "uw_v 1. insert"],
        #    "len(u)-1" : ["uv 1. insert", "uw 1. insert"], #"uv_w 1. insert", "uw_v 1. insert"],
        #    "len(uv)_1" : ["uv_w 1. insert"],# "uw_v 1. insert"],
        #    "len(uw)-1" : ["uw_v 1. insert"],
        #    "len(vw)-1" : ["vw_u 1. insert"],
        #    "subescher uv end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert"],
        #    "subescher uw end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert"],
        #    "subescher vw end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert"],
        #    "subescher uv_w end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #    "subescher uw_v end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #    "subescher vw_u end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],}
        # Edges after dropping len's
        # return {
        #     "0" : ["subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start"],#, "uv 1. insert","uw 1. insert", "vw 1. insert", "uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #     "subescher uv end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #     "subescher uw end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #     "subescher vw end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #     "subescher uv_w end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert","uw 1. insert","vw 1. insert"],
        #     "subescher uw_v end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert","uv 1. insert","vw 1. insert"],
        #     "subescher vw_u end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv 1. insert","uw 1. insert"]}
        return {
            "0" : ["subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start"],#, "uv 1. insert","uw 1. insert", "vw 1. insert", "uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
            "subescher uv end" : ["uv 1. insert","uw 1. insert","vw 1. insert"],
            "subescher uw end" : ["uv 1. insert","uw 1. insert","vw 1. insert"],
            "subescher vw end" : ["uv 1. insert","uw 1. insert","vw 1. insert"]}
        #Edges after dropping len's and changing the cores for pairs
        # return {
        #     "subescher uv end" : ["0","uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert"],
        #     "subescher uw end" : ["0","uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert"],
        #     "subescher vw end" : ["0","uv 1. insert","uw 1. insert","vw 1. insert", "uv_w 1. insert", "uw_v 1. insert"],
        #     "subescher uv_w end" : ["0","uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #     "subescher uw_v end" : ["0","uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],
        #     "subescher vw_u end" : ["0","uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert"],}    
    

    @staticmethod
    def getCoreLabels(partition):
        # original cores
        #return ["0", "len(u)-1", "subescher uv start", "subescher uv end", "uv 1. insert", "len(v)-1", "subescher vw start", "subescher vw end", "vw 1. insert", 
        #        "len(u)-1 (again)", "subescher uw start", "subescher uw end", "uw 1. insert", "len(uv)_1", "subescher uv_w start", 
        #        "subescher uv_w end", "uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
        #        "len(vw)-1", "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert"]
        # short cores (without len's)
        return ["0", "subescher uv start", "subescher uv end", "uv 1. insert", "subescher vw start", "subescher vw end", "vw 1. insert", 
                "subescher uw start", "subescher uw end", "uw 1. insert", "subescher uv_w start", 
                "subescher uv_w end", "uv_w 1. insert", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
                "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert"]
        # shortest cores (without len's and escher starts)
        # return ["0", "subescher uv end", "uv 1. insert", "subescher vw end", "vw 1. insert", 
        #         "subescher uw end", "uw 1. insert",  
        #         "subescher uv_w end", "uv_w 1. insert", "subescher uw_v end", "uw_v 1. insert",
        #         "subescher vw_u end", "vw_u 1. insert"]

    def getCoreLabelGroups(partition):
        return {
            "0":{
                "0":"0"
            },
            "subescher start":{
                "subescher uv start":"uv",
                "subescher vw start":"vw",
                "subescher uw start":"uw",
                "subescher uv_w start":"uv_w",
                "subescher uw_v start":"uw_v",
                "subescher vw_u start":"vw_u",
            },
            "subescher end":{
                "subescher uv end":"uv",
                "subescher vw end":"vw",
                "subescher uw end":"uw",
                "subescher uv_w end":"uv_w",
                "subescher uw_v end":"uw_v",
                "subescher vw_u end":"vw_u",
            },
            "1. insertion":{
                "uv 1. insert":"uv",
                "vw 1. insert":"vw",
                "uw 1. insert":"uw",
                "uv_w 1. insert":"uv_w",
                "uw_v 1. insert":"uw_v",
                "vw_u 1. insert":"vw_u",
            }
        }
    
    def getCoreLabelGroupColors(partition):
        return {
            "0":"skyblue",
            "subescher start":"hotpink",
            "subescher end":"silver",
            "1. insertion":"limegreen",
            }

    # def generateCore(self, eschertripple):
        
    #     def add_half(L):
    #         return L[:-1]+[L[-1]+0.5]
    #     u,v,w = eschertripple

    #     # get data of double pairs
    #     core_u_v = self.get_insertion_and_subescher_of_2_eschers(u,v)
    #     core_v_w = self.get_insertion_and_subescher_of_2_eschers(v,w)
    #     core_u_w = self.get_insertion_and_subescher_of_2_eschers(u,w)
        
    #     # If there is no insertion, then it is good
    #     #if core_u_v == [] or core_u_w == [] or core_v_w == []:
    #     #    return "GOOD"

    #     if core_u_v[-1] == -1:
    #         core_uv_w = [len(u)+len(v)-1, -1, -1, -1]
    #     else:
    #         uv = self.concat(u, v, insertionpoint=core_u_v[-1])
    #         core_uv_w = self.get_insertion_and_subescher_of_2_eschers(uv,w)

    #     if core_u_w[-1] == -1:
    #         core_uw_v = [len(u)+len(w)-1, -1, -1, -1]
    #     else:
    #         uw = self.concat(u, w, insertionpoint=core_u_w[-1])
    #         core_uw_v = self.get_insertion_and_subescher_of_2_eschers(uw,v)

    #     if core_v_w[-1] == -1:
    #         core_vw_u = [len(v) + len(w)-1, -1, -1, -1]
    #     else:
    #         vw = self.concat(v, w, insertionpoint=core_v_w[-1])
    #         core_vw_u = self.get_insertion_and_subescher_of_2_eschers(vw,u)


    #     #print([0] + add_half(core_v_w) + add_half(core_uv_w) + add_half(core_uw_v))
    #     #print([0] + core_v_w + core_uv_w + core_uw_v)
    #     return [0] + add_half(core_u_v) + add_half(core_v_w) + add_half(core_u_w) + add_half(core_uv_w) + add_half(core_uw_v) + add_half(core_vw_u)

    # def generateCore(self, eschertripple):
        
    #     def add_half(L):
    #         return L[:-1]+[L[-1]+0.5]
    #     u,v,w = eschertripple

    #     # get data of double pairs
    #     core_u_v = self.get_short_insertion_and_subescher_of_2_eschers(u,v)
    #     core_v_w = self.get_short_insertion_and_subescher_of_2_eschers(v,w)
    #     core_u_w = self.get_short_insertion_and_subescher_of_2_eschers(u,w)
        
    #     # If there is no insertion, then it is good
    #     #if core_u_v == [] or core_u_w == [] or core_v_w == []:
    #     #    return "GOOD"

    #     if core_u_v[-1] == -1:
    #         core_uv_w = [-1, -1, -1]
    #     else:
    #         uv = self.concat(u, v, insertionpoint=core_u_v[-1])
    #         core_uv_w = self.get_short_insertion_and_subescher_of_2_eschers(uv,w)

    #     if core_u_w[-1] == -1:
    #         core_uw_v = [-1, -1, -1]
    #     else:
    #         uw = self.concat(u, w, insertionpoint=core_u_w[-1])
    #         core_uw_v = self.get_short_insertion_and_subescher_of_2_eschers(uw,v)

    #     if core_v_w[-1] == -1:
    #         core_vw_u = [-1, -1, -1]
    #     else:
    #         vw = self.concat(v, w, insertionpoint=core_v_w[-1])
    #         core_vw_u = self.get_short_insertion_and_subescher_of_2_eschers(vw,u)


    #     #print([0] + add_half(core_v_w) + add_half(core_uv_w) + add_half(core_uw_v))
    #     #print([0] + core_v_w + core_uv_w + core_uw_v)
    #     return [0] + add_half(core_u_v) + add_half(core_v_w) + add_half(core_u_w) + add_half(core_uv_w) + add_half(core_uw_v) + add_half(core_vw_u)

    def generateCore(self, eschertripple):
        
        def add_half(L):
            return L[:-1]+[L[-1]+0.5]
        u,v,w = eschertripple

        # get data of double pairs
        core_u_v = self.get_shortb_insertion_and_subescher_of_2_eschers(u,v)
        core_v_w = self.get_shortb_insertion_and_subescher_of_2_eschers(v,w)
        core_u_w = self.get_shortb_insertion_and_subescher_of_2_eschers(u,w)
        
        # If there is no insertion, then it is good
        #if core_u_v == [] or core_u_w == [] or core_v_w == []:
        #    return "GOOD"

        if core_u_v[-1] == -1:
            core_uv_w = [-1, -1, -1]
        else:
            uv = self.concat(u, v, insertionpoint=core_u_v[-1])
            core_uv_w = self.get_shortb_insertion_and_subescher_of_2_eschers(uv,w)

        if core_u_w[-1] == -1:
            core_uw_v = [-1,-1, -1]
        else:
            uw = self.concat(u, w, insertionpoint=core_u_w[-1])
            core_uw_v = self.get_shortb_insertion_and_subescher_of_2_eschers(uw,v)

        if core_v_w[-1] == -1:
            core_vw_u = [-1, -1, -1]
        else:
            vw = self.concat(v, w, insertionpoint=core_v_w[-1])
            core_vw_u = self.get_shortb_insertion_and_subescher_of_2_eschers(vw,u)


        #print([0] + add_half(core_v_w) + add_half(core_uv_w) + add_half(core_uw_v))
        #print([0] + core_v_w + core_uv_w + core_uw_v)
        return [0] + add_half(core_u_v) + add_half(core_v_w) + add_half(core_u_w) + add_half(core_uv_w) + add_half(core_uw_v) + add_half(core_vw_u)

        """
        u,v, w = eschertripple

        # get data of double pairs
        core_u_v = self.get_insertion_and_subescher_of_2_eschers(u,v)
        core_v_w = self.get_insertion_and_subescher_of_2_eschers(v,w)
        core_u_w = self.get_insertion_and_subescher_of_2_eschers(u,w)

        # if one doesnt have an insertion, then let's say it's good
        if core_u_v == [] or core_v_w == [] or core_u_w == []:
            return []
        
        # Concatenate
        uv = self.concat(u, v, insertionpoint=core_u_v[-1])
        vw = self.concat(v, w, insertionpoint=core_v_w[-1])
        uw = self.concat(u, w, insertionpoint=core_u_w[-1])

        # get data of partial concatenated double pairs
        core_uv_w = self.get_insertion_and_subescher_of_2_eschers(uv,w)
        core_vw_u = self.get_insertion_and_subescher_of_2_eschers(vw,u)
        core_uw_v = self.get_insertion_and_subescher_of_2_eschers(uw,v)
        
        return [0] + core_u_v + core_v_w + core_u_w + core_uv_w + core_vw_u + core_uw_v
        """

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
    
    """ def getCoreRepresentation(self, core):
        k = len(core)

        if core == "GOOD":
            return "GOOD"

        if core == "BAD":
            return "BAD"
        
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
