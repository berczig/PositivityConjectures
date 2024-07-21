from SPC.Restructure.cores.CorrectSequenceCoreGenerator import CorrectSequenceCoreGenerator
from SPC.Restructure.cores.CorrectSequenceCoreGeneratorAbstract import CorrectSequenceCoreGeneratorAbstract
from SPC.Restructure.cores.EscherCoreGeneratorBasic import EscherCoreGeneratorBasic
from SPC.Restructure.cores.CoreGenerator import CoreGenerator
from SPC.Restructure.cores.EscherCoreGeneratorTripple import EscherCoreGeneratorTripple
from SPC.Restructure.cores.EscherCoreGeneratorAbstract import EscherCoreGeneratorAbstract
from SPC.Restructure.UIO import UIO
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


    @lru_cache(maxsize=None)
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
        elif isinstance(GEN, CorrectSequenceCoreGenerator):
            for corseq in self.getCorrectSequences(partition):
                yield GEN.generateCore(corseq)
        else:
            assert False==True, "{} is not a subclass of EscherCoreGeneratorAbstract or CorrectSequenceCoreGenerator".format(str(self.core_generator_class))

    def getCoreRepresentations(self, partition):
        GEN = self.core_generator_class(self.uio, partition)
        for core in self.getCores(partition):
            yield GEN.getCoreRepresentation(core)


    def getEschers(self, partition):
        P = getPermutationsOfN(self.uio.N)
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

    def getCoefficient(self, partition):
        if len(partition) == 1:
            return count(self.getCorrectSequences(partition))

        elif len(partition) == 2:
            return count(self.getCorrectSequences(partition)) - count(self.getCorrectSequences((self.uio.N,)))
        
        elif len(partition) == 3:
            n,k,l = partition
            """ return 2*len(self.getEschers((n+k+l,))) +\
                  len(self.getEschers(partition)) -\
                      len(self.getEschers((n+l,k))) -\
                          len(self.getEschers((n+k,l))) -\
                              len(self.getEschers((l+k,n))) """
            return 2*count(self.getEschers((n+k+l,))) +\
                  count(self.getEschers(partition)) -\
                      count(self.getEschers((n+l,k))) -\
                          count(self.getEschers((n+k,l))) -\
                              count(self.getEschers((l+k,n)))
        
    def __repr__(self) -> str:
        return "EXTRACTOR OF ["+str(self.uio.encoding)+"]"

