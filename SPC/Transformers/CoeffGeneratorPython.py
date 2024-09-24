from functools import lru_cache
from itertools import permutations
import os
import numpy as np
import SPC
from pathlib import Path
import time

#### Parameters Start ########
uio_partition = (4,2,1)
escher_core_generatpr = "Escher"
uio_max_size = 7
#### Parameters End   ########


#### Calculate constants ########
uio_size = sum(uio_partition)
#### Calculate constants ########


def getKPermutationsOfN(n, k):
    return permutations(range(n), k)

def partitionsOfN(n):
    def partitions(n, I=1):
        yield (n,)
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, i):
                yield p+(i,) 
    return partitions(n)

#def getAllPartitionsOfN(n):


def count(iterable):
    return sum((1 for _ in iterable))

def getUnusedFilepath(filepath):
    folder, filename = os.path.split(filepath)
    base_filename, extension = os.path.splitext(filename)
    newfilename = filename
    i = 1
    while newfilename in os.listdir(folder):
        newfilename = f"{base_filename}_{i:03d}{extension}"
        i += 1
    return os.path.join(folder, newfilename)



class UIO:

    INCOMPARABLE = 100    # (i,j) is INCOMPARABLE if neither i < j nor j < i nor i = j -- connected in the incomparability graph -- intersecting intervals
    LESS = 101              # (i,j) is LE iff i < j     interval i is to the left of j 
    GREATER = 102              # (i,j) is LE iff i > j     interval i is to the right of j
    EQUAL = 103              # (i,j) is EQ iff i = j     interval i and interval j are same interval
    RELATIONTEXT = {LESS:"<", EQUAL:"=", GREATER:">"}
    RELATIONTEXT2 = {LESS:"LE", EQUAL:"EQ", GREATER:"GR"}

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

    def __repr__(self):
        return self.repr
    

    ### POSET STRUCTURE ###
    

    def isescher(self, seq):
        for i in range(len(seq)-1):
            if self.isarrow(seq, i, i+1) == False:
                return False
        return self.isarrow(seq, -1, 0)
    
    def iscorrect(self, seq):
        for i in range(1,len(seq)):
            # 1) arrow condition
            if not self.isarrow(seq, i-1, i):
                return False
            # 2) intersects with some previous interval
            intersects = False
            for j in range(0, i):
                if self.comparison_matrix[seq[i], seq[j]] in [UIO.LESS, UIO.INCOMPARABLE]:
                    intersects = True
                    #break
            if not intersects:
                return False
        return True

    def isarrow(self, escher, i,j, verbose=False): # 0 <= i,j < len(escher)
        if verbose:
            print("arrow", escher, i, j, self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER)
        return self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER # EQUAL also intersects
    
    def intervalsAreIntersecting(self, i, j):
        return self.comparison_matrix[i, j] == UIO.INCOMPARABLE
    
    def intervalIsToTheRight(self, i, j):
        return self.comparison_matrix[i, j] == UIO.GREATER
    
    def toPosetData(self, seq):
        k = len(seq)
        return tuple([self.comparison_matrix[seq[i], seq[j]] for i in range(0, k) for j in range(i+1, k)])

    ### SUB-UIO ###


    def getsubuioencoding(self, seq):

        def addupto(List, element, uptoindex):
            # adds element to List up to index uptoindex (exclusive)
            if len(List) == uptoindex:
                return List
            A = (uptoindex-len(List))*[element]
            return List + A
    
        N = len(seq)
        encod = []
        for i in range(N):
            for j in range(i+1, N):
                #print(i, j, self.comparison_matrix[seq[i], seq[j]])
                if self.comparison_matrix[seq[i], seq[j]] != self.INCOMPARABLE: # not intersect
                    #print(i, "s up to (exclusive)", j)
                    encod = addupto(encod, i, j)
                    break
                elif j == N-1:
                    #print(i, "s up to (exclusive)", N, "final")
                    encod = addupto(encod, i, N)
        if self.comparison_matrix[seq[-1], seq[-2]] != self.INCOMPARABLE:
            encod.append(N-1)
        
        return encod


class EscherDoublePartitionGenerator:
    pass



class UIODataExtractor:
    """
    The UIODataExtractor is focused on a specific UIO. It can generate and keep track of all possible λ-eschers and λ-correct sequences of the UIO and generates the cores using CoreGenerator.
    Note: λ-correct sequences are returned as 1 sequence were as λ-eschers are returned as tuples: λ-escher = (escher_1, escher_2)
    """
    
    def __init__(self, uio:UIO, core_generator_class):
        self.uio = uio
        self.core_generator_class = core_generator_class
         
    def getCores(self, partition):
        if self.core_generator_class == "EscherDoublePartitionGenerator":
            GEN = EscherDoublePartitionGenerator(self.uio, partition)
            for escher in self.getEschers(partition):
                yield GEN.generateCore(escher)

    def getCoreRepresentations(self, partition):
        if self.core_generator_class == "EscherDoublePartitionGenerator":
            GEN = EscherDoublePartitionGenerator(self.uio, partition)
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
            n,k = partition
            return self.countEschers(partition) - self.countEschers((n+k,))
        
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





class GlobalUIODataPreparer():

    def __init__(self, n):
        #super().__init__(["coreRepresentationsCategorizer", "coefficients", "partition"])
        self.n = n
        self.uios_initialized = False
        self.coreRepresentationsCategorizer = {} # {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2,...}, ...}
        self.coefficients = [] # list of all coefficients (for the partition given to getTrainingData)

    def initUIOs(self, core_generator_type):
        self.uios_initialized = True
        encodings = self.generate_all_uio_encodings(self.n)
        #encodings = [encodings[2379]]
        self.extractors = [UIODataExtractor(UIO(enc), core_generator_type) for enc in encodings]
        print("Initialized {} UIOs".format(len(encodings)))
        #for i, enc in enumerate(encodings):
            #print(i, enc)

    def getUIOs(self,i):
        return self.extractors[i].uio.encoding
    
    def getAllUIOEncodings(self):
        return self.generate_all_uio_encodings(self.n)


    def computeTrainingData(self, partition:tuple, core_generator_type:str) -> tuple:
        """
        :return: (X,y) the training data. Where X is a dict with the keys being coreRepresentations, and 
        the values being a dict describing the occurences in each UIO. 
        
        In contrast to normal supervised learning input data, the data isn't ordered from first to last 
        input instance as a list, but it is the data of all input instances at once in a dictionary, 
        with keys not being the input instances (UIOs), but  some data (core representations) collected 
        among all UIOs, where we count the occurences coming from each individual UIO.
        
        y is a list with the coefficient of each UIO.
        """
        print("computing training data...")

        # uios initialized?
        if self.uios_initialized == False:
            self.initUIOs(core_generator_type)

        # count correps and compute coefficients
        self.partition = partition # remember what partition was used to calculate the most recent training data
        #self.countCoreRepresentations(partition)

        n = len(self.extractors)
        n_10 = max(1, n//10)
        print(f"Calculating coefficients... (print every {n_10} iterations)")
        self.coefficients = []
        for uioID, extractor in enumerate(self.extractors):
            if uioID % n_10 == 0:
                print(" > current UIO: {}/{}".format(uioID+1, n))
            self.coefficients.append(extractor.getCoefficient(partition))
    
    def loadTrainingData(self, filepath:str) -> tuple:
        self.load(filepath)

    def getTrainingData(self):
        return self.coreRepresentationsCategorizer, self.coefficients
    
    def saveTrainingData(self, filepath:str) -> None:
        self.save(filepath)
    

    def generate_uio_encoding_rec(self,A, uio, n, i):
        if i == n:
            A.append(uio)
            return
        for j in range(uio[i-1], i+1):
            self.generate_uio_encoding_rec(A, uio + [j], n, i + 1)

    def generate_all_uio_encodings(self,n):
        A = []
        self.generate_uio_encoding_rec(A, [0], n, 1)
        print("Generated", len(A), "unit order intervals encodings")
        return A


    # def generate_all_uio_encodings(self, n):

    #     def generate_uio_encoding_rec(A, uio, n, i):
    #         if i == n:
    #             A.append(uio)
    #             return
    #         for j in range(uio[i-1], i+1):
    #             generate_uio_encoding_rec(A, uio+[j], n, i+1)

    #     A = []
    #     generate_uio_encoding_rec(A, [0], n, 1)
    #     #print("this is the one 2379:", A[2379])
    #     print("Generated", len(A), "unit order intervals encodings")
    #     return A
    
    def getInputdataAsCountsMatrix(self):
        countmatrix = [[0 for __ in range(len(self.coreRepresentationsCategorizer))] for _ in range(len(self.coefficients))]
        for corerepID, corerep in enumerate(self.coreRepresentationsCategorizer):
            UIOcounts = self.coreRepresentationsCategorizer[corerep]
            for UIOID in UIOcounts:
                countmatrix[UIOID][corerepID] += UIOcounts[UIOID]
        return countmatrix
    
    def countCoreRepresentations(self, partition):
        core_rep_categories = {} # coreRepresentation:ID
        counter = {} # corerepID:dict(uioID1:occurrences1, uioID2:occurrences2)
        corerep_generators = [extractor.getCoreRepresentations(partition) for extractor in self.extractors]

        ID = 0
        total_corereps = 0
        n = len(corerep_generators)
        n_10 = max(1, n//10)
        print(f"Categorizing core representations... (print every {n_10} iterations)")
        for uioID, corerep_generator in enumerate(corerep_generators):
            if uioID % n_10 == 0:
                print(" > current UIO: {}/{}".format(uioID+1, n))
            for corerep in corerep_generator:
                total_corereps += 1
                #print("b", type(corerep), corerep)
                # determine corerep ID
                if corerep not in core_rep_categories:
                    ID = len(core_rep_categories)
                    core_rep_categories[corerep] = ID
                else:
                    ID = core_rep_categories[corerep]

                # count this observed category
                if ID not in counter:
                    counter[ID] = {uioID:1}
                else:
                    categoryOccurrencer = counter[ID]
                    if uioID not in categoryOccurrencer:
                        categoryOccurrencer[uioID] = 1
                    else:
                        categoryOccurrencer[uioID] += 1

        print("Found {} core representations in total".format(total_corereps))
        
        # change keys from an ID of the coreRepresentation to the coreRepresentation itself
        for cat in core_rep_categories:
            #print("cat:", cat)
            self.coreRepresentationsCategorizer[cat] = counter[core_rep_categories[cat]]
        #for key in self.coreRepresentationsCategorizer:
            #print("core:", key)
        print("Found",len(self.coreRepresentationsCategorizer), "distinct core representations / categories")


    
def getUIOData(uio_max_size, partition_):
    DataPreparer = GlobalUIODataPreparer(uio_max_size)
    DataPreparer.computeTrainingData(partition=partition_, core_generator_type="EMPTY")
    return DataPreparer.generate_all_uio_encodings(uio_max_size), DataPreparer.coefficients

def getAllUIODataUpTo(uio_max_size):
    for uio_size in range(1, uio_max_size+1):
        for partitionsize in range(1, uio_size+1):
            for partition in partitionsOfN(partitionsize):
                printUIOData(getUIOData(uio_size, partition), partition)


def printUIOData(data, partition):
    encod, coeffs = data
    n = len(encod)
    for i in range(n):
        print(f"{i+1}th UIO {encod[i]} has {partition}-coeff {coeffs[i]}")



if __name__ == "__main__":
    start_time = time.time()
    getUIOData(uio_max_size, uio_partition)
    print("Time:", time.time()-start_time)
    #printUIOData(getUIOData(uio_max_size, partition))



# no subclasses
# no importlib
# only 1 file