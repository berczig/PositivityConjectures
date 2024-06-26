from SPC.Restructure.cores.CorrectSequenceCoreGenerator import CorrectSequenceCoreGenerator
from SPC.Restructure.cores.EscherCoreGenerator import EscherCoreGenerator
from SPC.Restructure.cores.EscherTrippleCoreGenerator import EscherTrippleCoreGenerator
from SPC.Restructure.UIO import UIO
from SPC.misc.misc import *

from functools import lru_cache


class UIODataExtractor:
    """
    The UIODataExtractor is focused on a specific UIO. It can generate and keep track of all possible λ-eschers and λ-correct sequences of the UIO and generates the cores using CoreGenerator.
    Note: λ-correct sequences are returned as 1 sequence were as λ-eschers are returned as tuples: λ-escher = (escher_1, escher_2)
    """
    
    def __init__(self, uio:UIO):
        self.uio = uio


    @lru_cache(maxsize=None)
    def getCorrectSequences(self, partition):
        if len(partition) == 1:
            return [seq for seq in getPermutationsOfN(self.uio.N) if self.uio.iscorrect(seq)]
        elif len(partition) == 2:
            a = partition[0]
            return [seq for seq in getPermutationsOfN(self.uio.N) if self.uio.iscorrect(seq[:a]) 
                    and self.uio.iscorrect(seq[a:]) ]
        
    @lru_cache(maxsize=None)
    def getCorrectSequenceCores(self, partition):
        if len(partition) == 2:
            GEN = CorrectSequenceCoreGenerator(self.uio, partition)
            return [GEN.generateCore(corseq) for corseq in self.getCorrectSequences(partition)]
        
    @lru_cache(maxsize=None)
    def getCorrectSequenceCoreRepresentations(self, partition):
        if len(partition) == 2:
            return [self.uio.toPosetData(core) for core in self.getCorrectSequenceCores(partition)]



    @lru_cache(maxsize=None)
    def getEschers(self, partition):
        if len(partition) == 1:
            return [seq for seq in getPermutationsOfN(self.uio.N) if self.uio.isescher(seq)]
        elif len(partition) == 2:
            a = partition[0]
            return [(seq[:a], seq[a:]) for seq in getPermutationsOfN(self.uio.N) if self.uio.isescher(seq[:a]) 
                    and self.uio.isescher(seq[a:]) ]
        elif len(partition) == 3:
            a,b,c  = partition                   
            return [(seq[:a], seq[a:a+b], seq[a+b:]) for seq in getPermutationsOfN(self.uio.N) if self.uio.isescher(seq[:a]) 
                    and self.uio.isescher(seq[a:a+b]) and self.uio.isescher(seq[a+b:]) ]

    @lru_cache(maxsize=None)
    def getEscherCores(self, partition):
        if len(partition) == 2:
            GEN = EscherCoreGenerator(self.uio, partition)
            return [GEN.generateCore(escherpair) for escherpair in self.getEschers(partition)] 
        elif len(partition) == 3:
            GEN = EscherTrippleCoreGenerator(self.uio, partition)
            return [GEN.generateCore(escherpair) for escherpair in self.getEschers(partition)] 
        
    @lru_cache(maxsize=None)
    def getEscherCoreRepresentations(self, partition):
        if len(partition) == 2:
            return [EscherCoreGenerator.toEscherCoreRepresentation(core) for core in self.getEscherCores(partition)]
        if len(partition) == 3:
            return [EscherTrippleCoreGenerator.toEscherCoreRepresentation(core) for core in self.getEscherCores(partition)]



    def getCoefficient(self, partition):
        if len(partition) == 1:
            return len(self.getCorrectSequences(partition))

        elif len(partition) == 2:
            return len(self.getCorrectSequences(partition)) - len(self.getCorrectSequences((self.uio.N,)))
        
        elif len(partition) == 3:
            n,k,l = partition
            return 2*len(self.getEschers((n+k+l,))) +\
                  len(self.getEschers(partition)) -\
                      len(self.getEschers((n+l,k))) -\
                          len(self.getEschers((n+k,l))) -\
                              len(self.getEschers((l+k,n)))
        
    def __repr__(self) -> str:
        return "EXTRACTOR OF ["+str(self.uio.encoding)+"]"

