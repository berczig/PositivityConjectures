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
        # 0 is compared to all subescher starts, subescher ends are compared to all 1. inserts
        return {
            "0" : ["subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start", 
                   "subescher uz start", "subescher vz start", "subescher wz start", "subescher uz_w start", "subescher vz_u start", "subescher wz_v start"],
            "subescher uv end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"],
            "subescher uw end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"],
            "subescher vw end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"],
            "subescher uz end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"],
            "subescher vz end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"],
            "subescher wz end" : ["uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"],
            "subescher uv_w end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uw_v end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher vw_u end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uv_z end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uz_v end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher vz_u end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uw_z end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uz_w end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher wz_u end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher vw_z end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher vz_w end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher wz_v end" : ["uv_w 1. insert", "uw_v 1. insert", "vw_u 1. insert", "uv_z 1. insert", "uz_v 1. insert", "vz_u 1. insert","uw_z 1. insert", "uz_w 1. insert", "wz_u 1. insert", "vw_z 1. insert", "vz_w 1. insert", "wz_v 1. insert","uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uvw_z end" : ["uvw_z 1. insert", "uvz_w 1. insert", "uwz_v 1. insert", "vwz_u 1. insert", "uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uvz_w end" : ["uvw_z 1. insert", "uvz_w 1. insert", "uwz_v 1. insert", "vwz_u 1. insert", "uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher uwz_v end" : ["uvw_z 1. insert", "uvz_w 1. insert", "uwz_v 1. insert", "vwz_u 1. insert", "uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],
            "subescher vwz_u end" : ["uvw_z 1. insert", "uvz_w 1. insert", "uwz_v 1. insert", "vwz_u 1. insert", "uv_wz 1. insert", "uw_vz 1. insert", "vw_uz 1. insert"],}


    def getCoreLabels(partition):
        # original cores
        #return ["0", "len(u)-1", "subescher uv start", "subescher uv end", "uv 1. insert", "len(v)-1", "subescher vw start", "subescher vw end", "vw 1. insert", 
        #        "len(u)-1 (again)", "subescher uw start", "subescher uw end", "uw 1. insert", "len(uv)_1", "subescher uv_w start", 
        #        "subescher uv_w end", "uv_w 1. insert","len(uw)-1", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
        #        "len(vw)-1", "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert"]
        # short cores (without len's)
        # return subescher start for all pairs, 1. insert for all pairs, subescher end for all pairs
        return ["0", "subescher uv start", "subescher uv end", "uv 1. insert", "subescher vw start", "subescher vw end", "vw 1. insert", 
                "subescher uw start", "subescher uw end", "uw 1. insert", "subescher uz start", "subescher uz end", "uz 1. insert",
                "subescher vz start", "subescher vz end", "vz 1. insert", "subescher wz start", "subescher wz end", "wz 1. insert",
                "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert", 
                "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
                "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert", 
                "subescher uv_z start", "subescher uv_z end", "uv_z 1. insert",
                "subescher uz_v start", "subescher uz_v end", "uz_v 1. insert",
                "subescher vz_u start", "subescher vz_u end", "vz_u 1. insert",
                "subescher uw_z start", "subescher uw_z end", "uw_z 1. insert", 
                "subescher uz_w start", "subescher uz_w end", "uz_w 1. insert",
                "subescher wz_u start", "subescher wz_u end", "wz_u 1. insert",
                "subescher vw_z start", "subescher vw_z end", "vw_z 1. insert",
                "subescher vz_w start", "subescher vz_w end", "vz_w 1. insert",
                "subescher wz_v start", "subescher wz_v end", "wz_v 1. insert",
                "subescher uvw_z start", "subescher uvw_z end", "uvw_z 1. insert", 
                "subescher uvz_w start", "subescher uvz_w end", "uvz_w 1. insert",
                "subescher uwz_v start", "subescher uwz_v end", "uwz_v 1. insert", 
                "subescher vwz_u start", "subescher vwz_u end", "vwz_u 1. insert",
                "subescher uv_wz start", "subescher uv_wz end", "uv_wz 1. insert", 
                "subescher uw_vz start", "subescher uw_vz end", "uw_vz 1. insert",
                "subescher vw_uz start", "subescher vw_uz end", "vw_uz 1. insert",]
    #def getCoreLabels(partition):
    #    return ["a", "b", "c"]

    def generateCore(self, escherquadruple):
        
        u,v,w,z = escherquadruple

        def add_half(L):
            return L[:-1]+[L[-1]+0.5]
        

        # get data of double pairs
        core_u_v = self.get_shortb_insertion_and_subescher_of_2_eschers(u,v)
        core_v_w = self.get_shortb_insertion_and_subescher_of_2_eschers(v,w)
        core_u_w = self.get_shortb_insertion_and_subescher_of_2_eschers(u,w)
        core_u_z = self.get_shortb_insertion_and_subescher_of_2_eschers(u,z)
        core_v_z = self.get_shortb_insertion_and_subescher_of_2_eschers(v,z)
        core_w_z = self.get_shortb_insertion_and_subescher_of_2_eschers(w,z)

        
        # Calculate the core of a triple pair and ab_cd

        if core_u_v[-1] == -1:
            core_uv_w = [-1, -1, -1]
            core_uv_z = [-1, -1, -1]
            core_uv_wz = [-1, -1, -1]
        else:
            uv = self.concat(u, v, insertionpoint=core_u_v[-1])
            core_uv_w = self.get_shortb_insertion_and_subescher_of_2_eschers(uv,w)
            core_uv_z = self.get_shortb_insertion_and_subescher_of_2_eschers(uv,z)
            if core_w_z[-1] == -1:
                core_uv_wz = [-1, -1, -1]
            else:
                core_uv_wz = self.get_shortb_insertion_and_subescher_of_2_eschers(uv,self.concat(w,z, insertionpoint=core_w_z[-1]))

        if core_u_w[-1] == -1:
            core_uw_v = [-1,-1, -1]
            core_uw_z = [-1,-1, -1]
            core_uw_vz = [-1,-1, -1]
        else:
            uw = self.concat(u, w, insertionpoint=core_u_w[-1])
            core_uw_v = self.get_shortb_insertion_and_subescher_of_2_eschers(uw,v)
            core_uw_z = self.get_shortb_insertion_and_subescher_of_2_eschers(uw,z)
            if core_v_z[-1] == -1:
                core_uw_vz = [-1, -1, -1]   
            else:
                core_uw_vz = self.get_shortb_insertion_and_subescher_of_2_eschers(uw,self.concat(v,z, insertionpoint=core_v_z[-1]))

        if core_v_w[-1] == -1:
            core_vw_u = [-1, -1, -1]
            core_vw_z = [-1, -1, -1]
            core_vw_uz = [-1, -1, -1]
        
        else:
            vw = self.concat(v, w, insertionpoint=core_v_w[-1])
            core_vw_u = self.get_shortb_insertion_and_subescher_of_2_eschers(vw,u)
            core_vw_z = self.get_shortb_insertion_and_subescher_of_2_eschers(vw,z)
            if core_u_z[-1] == -1:
                core_vw_uz = [-1, -1, -1]
            else:
                core_vw_uz = self.get_shortb_insertion_and_subescher_of_2_eschers(vw,self.concat(u,z, insertionpoint=core_u_z[-1]))
        
        if core_u_z[-1] == -1:
            core_uz_w = [-1, -1, -1]
            core_uz_v = [-1, -1, -1]

        else:
            uz = self.concat(u, z, insertionpoint=core_u_z[-1])
            core_uz_w = self.get_shortb_insertion_and_subescher_of_2_eschers(uz,w)
            core_uz_v = self.get_shortb_insertion_and_subescher_of_2_eschers(uz,v)

        if core_v_z[-1] == -1:
            core_vz_w = [-1, -1, -1]
            core_vz_u = [-1, -1, -1]
        else:
            vz = self.concat(v, z, insertionpoint=core_v_z[-1])
            core_vz_w = self.get_shortb_insertion_and_subescher_of_2_eschers(vz,w)
            core_vz_u = self.get_shortb_insertion_and_subescher_of_2_eschers(vz,u)

        if core_w_z[-1] == -1:
            core_wz_u = [-1, -1, -1]
            core_wz_v = [-1, -1, -1]
        else:
            wz = self.concat(w, z, insertionpoint=core_w_z[-1])
            core_wz_u = self.get_shortb_insertion_and_subescher_of_2_eschers(wz,u)
            core_wz_v = self.get_shortb_insertion_and_subescher_of_2_eschers(wz,v)

        #Calculate the core of a quadruple pair of type abc_d

        if core_uv_w[-1] == -1 and core_uw_v[-1] == -1 and core_vw_u[-1] == -1:
            core_uvw_z = [-1, -1, -1]
        else:
            if core_uv_w[-1] != -1: 
                uvw = self.concat(uv, w, insertionpoint=core_uv_w[-1])
                core_uvw_z = self.get_shortb_insertion_and_subescher_of_2_eschers(uvw,z)
            elif core_uw_v[-1] != -1:
                uwv = self.concat(uw, v, insertionpoint=core_uw_v[-1])
                core_uvw_z = self.get_shortb_insertion_and_subescher_of_2_eschers(uwv,z)
            elif core_vw_u[-1] != -1:
                vwu = self.concat(vw, u, insertionpoint=core_vw_u[-1])
                core_uvw_z = self.get_shortb_insertion_and_subescher_of_2_eschers(vwu,z)  

        if core_uv_z[-1] == -1 and core_uz_v[-1] == -1 and core_vz_u[-1] == -1:
            core_uvz_w = [-1, -1, -1]
        else:
            if core_uv_z[-1] != -1:
                uvz = self.concat(uv, z, insertionpoint=core_uv_z[-1])
                core_uvz_w = self.get_shortb_insertion_and_subescher_of_2_eschers(uvz,w)
            elif core_uz_v[-1] != -1:
                uzv = self.concat(uz, v, insertionpoint=core_uz_v[-1])
                core_uvz_w = self.get_shortb_insertion_and_subescher_of_2_eschers(uzv,w)
            elif core_vz_u[-1] != -1:
                vzu = self.concat(vz, u, insertionpoint=core_vz_u[-1])
                core_uvz_w = self.get_shortb_insertion_and_subescher_of_2_eschers(vzu,w)

        if core_uw_z[-1] == -1 and core_uz_w[-1] == -1 and core_wz_u[-1] == -1:
            core_uwz_v = [-1, -1, -1]   
        else:
            if core_uw_z[-1] != -1:
                uwz = self.concat(uw, z, insertionpoint=core_uw_z[-1])
                core_uwz_v = self.get_shortb_insertion_and_subescher_of_2_eschers(uwz,v)
            elif core_uz_w[-1] != -1:
                uzw = self.concat(uz, w, insertionpoint=core_uz_w[-1])
                core_uwz_v = self.get_shortb_insertion_and_subescher_of_2_eschers(uzw,v)
            elif core_wz_u[-1]  != -1:
                wzu = self.concat(wz, u, insertionpoint=core_wz_u[-1])
                core_uwz_v = self.get_shortb_insertion_and_subescher_of_2_eschers(wzu,v)

        if core_vw_z[-1] == -1 and core_vz_w[-1] == -1 and core_wz_v[-1] == -1:
            core_vwz_u = [-1, -1, -1]

        else:
            if core_vw_z[-1] != -1:
                vwz = self.concat(vw, z, insertionpoint=core_vw_z[-1])
                core_vwz_u = self.get_shortb_insertion_and_subescher_of_2_eschers(vwz,u)
            elif core_vz_w[-1] != -1:
                vzw = self.concat(vz, w, insertionpoint=core_vz_w[-1])
                core_vwz_u = self.get_shortb_insertion_and_subescher_of_2_eschers(vzw,u)
            elif core_wz_v[-1]  != -1:
                wzv = self.concat(wz, v, insertionpoint=core_wz_v[-1])
                core_vwz_u = self.get_shortb_insertion_and_subescher_of_2_eschers(wzv,u)


        #print([0] + add_half(core_v_w) + add_half(core_uv_w) + add_half(core_uw_v))
        #print([0] + core_v_w + core_uv_w + core_uw_v)
        return [0] + add_half(core_u_v) + add_half(core_v_w) + add_half(core_u_w) + add_half(core_u_z) + add_half(core_v_z) + add_half(core_w_z) +add_half(core_uv_w) + add_half(core_uw_v) +add_half(core_vw_u)+add_half(core_vw_z) + add_half(core_uv_z) + add_half(core_uw_z) + add_half(core_uz_w) + add_half(core_uz_v) + add_half(core_vz_w) + add_half(core_vz_u) + add_half(core_wz_u) + add_half(core_wz_v)+add_half(core_uv_wz) + add_half(core_uw_vz) + add_half(core_vw_uz) + add_half(core_uvw_z) + add_half(core_uvz_w) + add_half(core_uwz_v)+add_half(core_vwz_u)




        #return (a[0], b[0], c[0])

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
