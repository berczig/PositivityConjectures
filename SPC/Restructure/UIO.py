import numpy as np

class UIO:

    INCOMPARABLE = 100    # (i,j) is INCOMPARABLE if neither i < j nor j < i nor i = j -- connected in the incomparability graph -- intersecting intervals
    LESS = 101              # (i,j) is LE iff i < j     interval i is to the left of j 
    GREATER = 102              # (i,j) is LE iff i > j     interval i is to the right of j
    EQUAL = 103              # (i,j) is EQ iff i = j     interval i and interval j are same interval
    RELATIONTEXT = {LESS:"<", EQUAL:"=", GREATER:">"}

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
    
    def getsubUIO(self, seq):
        return SubUIO(self.getsubuioencoding(seq), seq)

class SubUIO(UIO):

    def __init__(self, uio_encoding, subseq): # assumes subseq to be ordered smallest to biggest
        super().__init__(uio_encoding)
        self.rename = {}
        for ID, globalID in enumerate(subseq):
            self.rename[globalID] = ID

    def to_internal_indexing(self, seq):
        return tuple([self.rename[w] for w in seq])