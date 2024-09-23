from SPC.UIO.cores.CorrectSequenceCoreGenerator import CorrectSequenceCoreGenerator
from SPC.UIO.cores.CorrectSequenceCoreGeneratorAbstract import CorrectSequenceCoreGeneratorAbstract
from SPC.UIO.cores.EscherCoreGeneratorBasic import EscherCoreGeneratorBasic
from SPC.UIO.cores.CoreGenerator import CoreGenerator
from SPC.UIO.cores.EscherCoreGeneratorTripple import EscherCoreGeneratorTripple
from SPC.UIO.cores.EscherCoreGeneratorAbstract import EscherCoreGeneratorAbstract
from SPC.UIO.UIO import UIO
from SPC.misc.misc import *

from functools import lru_cache


class UIODataExtractor:
    """
    The UIODataExtractor is focused on a specific UIO. It can generate and keep track of all possible λ-eschers and λ-correct sequences of the UIO and generates the cores using CoreGenerator.
    Note: λ-correct sequences are returned as 1 sequence were as λ-eschers are returned as tuples: λ-escher = (escher_1, escher_2)
    """
    
    def __init__(self, uio:UIO, core_generator_class:CoreGenerator):
        self.uio = uio
        self.core_generator_class = core_generator_class


    def getCorrectSequences(self, partition):
        P = getPermutationsOfN(self.uio.N)
        if len(partition) == 1:
            for seq in P:
                if self.uio.iscorrect(seq):
                    yield seq
        elif len(partition) == 2:
            a = partition[0]
            for seq in P:
                if self.uio.iscorrect(seq[:a]) and self.uio.iscorrect(seq[a:]):
                    yield seq
         
    def getCores(self, partition):
        GEN = self.core_generator_class(self.uio, partition)
        if isinstance(GEN, EscherCoreGeneratorAbstract):
            for escher in self.getEschers(partition):
                yield GEN.generateCore(escher)
        elif isinstance(GEN, CorrectSequenceCoreGeneratorAbstract):
            for corseq in self.getCorrectSequences(partition):
                yield GEN.generateCore(corseq)
        else:
            assert False==True, "{} is not a subclass of EscherCoreGeneratorAbstract or CorrectSequenceCoreGeneratorAbstract".format(str(self.core_generator_class))

    def getCoreRepresentations(self, partition):
        GEN = self.core_generator_class(self.uio, partition)
        for core in self.getCores(partition):
            yield GEN.getCoreRepresentation(core)


    def getEschers(self, partition):
        P = getKPermutationsOfN(self.uio.N, sum(partition))
        if len(partition) == 1:
            for seq in P:
                if self.uio.isescher(seq):
                    yield seq
        elif len(partition) == 2:
            a = partition[0]
            for seq in P:
                if self.uio.isescher(seq[:a]) and self.uio.isescher(seq[a:]):
                    yield (seq[:a], seq[a:])
        elif len(partition) == 3:
            a,b,c  = partition              
            for seq in P:
                if self.uio.isescher(seq[:a]) and self.uio.isescher(seq[a:a+b]) and self.uio.isescher(seq[a+b:]):
                    yield (seq[:a], seq[a:a+b], seq[a+b:])     
        elif len(partition) == 4:
            a,b,c,d  = partition              
            for seq in P:
                if self.uio.isescher(seq[:a]) and self.uio.isescher(seq[a:a+b]) and self.uio.isescher(seq[a+b:a+b+c]) and self.uio.isescher(seq[a+b+c:a+b+c+d]):
                    yield (seq[:a], seq[a:a+b], seq[a+b:a+b+c], seq[a+b+c:])     

    def getCoefficient(self, partition):
        if len(partition) == 1:
            return count(self.getEschers(partition))

        elif len(partition) == 2:
            return self.countEschers(partition) - self.countEschers((sum(partition),))
        
        elif len(partition) == 3:
            n,k,l = partition
            """ return 2*len(self.getEschers((n+k+l,))) +\
                  len(self.getEschers(partition)) -\
                      len(self.getEschers((n+l,k))) -\
                          len(self.getEschers((n+k,l))) -\
                              len(self.getEschers((l+k,n))) """
            return 2*self.countEschers((n+k+l,)) +\
                  self.countEschers(partition) -\
                      self.countEschers((n+l,k)) -\
                          self.countEschers((n+k,l)) -\
                              self.countEschers((l+k,n))
        elif len(partition) == 4:
            a,b,c,d = partition
            return \
            self.countEschers((a,b,c,d)) -\
            \
            self.countEschers((a+b,c,d)) -\
            self.countEschers((a+c,b,d)) -\
            self.countEschers((a+d,b,c)) -\
            self.countEschers((b+c,a,d)) -\
            self.countEschers((b+d,a,c)) -\
            self.countEschers((c+d,a,b)) +\
            \
            self.countEschers((a+b,c+d)) +\
            self.countEschers((a+c, b+d)) +\
            self.countEschers((a+d, b+c)) +\
            \
            2*self.countEschers((a+b+c, d)) +\
            2*self.countEschers((a+b+d, c)) +\
            2*self.countEschers((a+c+d,b)) +\
            2*self.countEschers((b+c+d,a)) -\
            \
            6*self.countEschers((a+b+c+d,))
        

    @lru_cache(maxsize=None) # while calculating the coefficient the same partition can appear multiple times
    def countEschers(self, partition):
        return count(self.getEschers(partition))
        
    def __repr__(self) -> str:
        return "EXTRACTOR OF ["+str(self.uio.encoding)+"]"

