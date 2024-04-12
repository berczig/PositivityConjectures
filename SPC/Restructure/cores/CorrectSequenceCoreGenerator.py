from SPC.Restructure.CoreGenerator import CoreGenerator
from SPC.Restructure.UIO import UIO

class CorrectSequenceCoreGenerator(CoreGenerator):

    p = 1

    def generateCore(self, seq): # step 3 for l,k
        l,k = self.partition
        core = []
        # take last critical pairs (at most p)
        pairs = 0 # count number of registered critical pairs
        for i in range(l-1, 0, -1):
            if self.uio.intervalsAreIntersecting(seq[i], seq[i-1]):
                pairs += 1
                core.append(seq[i-1])
                core.append(seq[i])
                if pairs >= self.p:
                    break
        #assert pairs==p, "pairs not {p} but {pairs}"
        core.insert(0, pairs)

        # maximal element in first l-1
        core.append(self.getmaximalinterval(seq[:l-1]))

        # last k+1 elements
        core += seq[-(k+1):]
        return core
    
    def getmaximalinterval(self, subseq):
        maximals = []
        for i in subseq:
            ismaximal = True
            for j in subseq:
                if self.uio.intervalIsToTheRight(j, i): # one very redundant comparison: i==j, but whatever
                    # i can't be maximal
                    ismaximal = False
                    break
            if ismaximal:
                maximals.append(i)
        return max(maximals)
    