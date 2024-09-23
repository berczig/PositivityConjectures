from itertools import permutations
import numpy as np

#### Parameters Start ########
uio_partition = (3,1)
escher_core_generatpr = "Escher"
uio_max_size = 4
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

    def isarrow(self, escher, i,j, verbose=False): # 0 <= i,j < len(escher)
        if verbose:
            print("arrow", escher, i, j, self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER)
        return self.comparison_matrix[escher[i], escher[j]] != UIO.GREATER # EQUAL also intersects
    

class UIODataExtractor:
    """
    The UIODataExtractor is focused on a specific UIO. It can generate and keep track of all possible λ-eschers and λ-correct sequences of the UIO and generates the cores using CoreGenerator.
    Note: λ-correct sequences are returned as 1 sequence were as λ-eschers are returned as tuples: λ-escher = (escher_1, escher_2)
    """
    
    def __init__(self, uio:UIO, core_generator_class):
        self.uio = uio
        self.core_generator_class = core_generator_class

    def countEschers(self, partition):
        count = 0
        P = getKPermutationsOfN(self.uio.N, sum(partition))
        if len(partition) == 1:
            for seq in P:
                if self.uio.isescher(seq):
                    count += 1
        elif len(partition) == 2:
            a = partition[0]
            for seq in P:
                if self.uio.isescher(seq[:a]) and self.uio.isescher(seq[a:]):
                    count += 1
        elif len(partition) == 3:
            a,b,c  = partition              
            for seq in P:
                if self.uio.isescher(seq[:a]) and self.uio.isescher(seq[a:a+b]) and self.uio.isescher(seq[a+b:]):
                    count += 1     
        elif len(partition) == 4:
            a,b,c,d  = partition              
            for seq in P:
                if self.uio.isescher(seq[:a]) and self.uio.isescher(seq[a:a+b]) and self.uio.isescher(seq[a+b:a+b+c]) and self.uio.isescher(seq[a+b+c:a+b+c+d]):
                    count += 1
        return count     

    def getCoefficient(self, partition):
        if len(partition) == 1:
            return self.countEschers(partition)

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

        n = len(self.extractors)
        n_10 = max(1, n//10)
        print(f"Calculating coefficients... (print every {n_10} iterations)")
        self.coefficients = []
        for uioID, extractor in enumerate(self.extractors):
            if uioID % n_10 == 0:
                print(" > current UIO: {}/{}".format(uioID+1, n))
            self.coefficients.append(extractor.getCoefficient(partition))
    
    def generate_uio_encoding_rec(self, A, uio, n, i):
        if i == n:
            A.append(uio)
            return
        for j in range(uio[i-1], i+1):
            self.generate_uio_encoding_rec(A, uio+[j], n, i+1)
                
    def generate_all_uio_encodings(self, n):
        A = []
        self.generate_uio_encoding_rec(A, [0], n, 1)
        #print("this is the one 2379:", A[2379])
        print("Generated", len(A), "unit order intervals encodings")
        return A
    
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
    getAllUIODataUpTo(uio_max_size)
    #printUIOData(getUIOData(uio_max_size, partition))



# no subclasses
# no importlib
# only 1 file
# no lre cache
# no yield
# no function inside function