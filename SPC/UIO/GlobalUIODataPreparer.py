from SPC.UIO.UIODataExtractor import UIODataExtractor
from SPC.UIO.UIO import UIO
from SPC.UIO.cores.CoreGenerator import CoreGenerator
from SPC.misc.extra import PartiallyLoadable
import importlib

class GlobalUIODataPreparer(PartiallyLoadable):

    def __init__(self, n):
        super().__init__(["coreRepresentationsCategorizer", "coefficients", "partition"])
        self.n = n
        self.coreRepresentationsCategorizer = {} # {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2,...}, ...}
        self.coefficients = [] # list of all coefficients (for the partition given to getTrainingData)

    def initUIOs(self, core_generator_type):
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
        if not hasattr(self, "extractors"):
            class_ = getattr(importlib.import_module("SPC.UIO.cores."+core_generator_type), core_generator_type)
            self.initUIOs(class_)
        # count correps and compute coefficients
        self.partition = partition # remember what partition was used to calculate the most recent training data
        self.countCoreRepresentations(partition)

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
    
    def generate_all_uio_encodings(self, n):

        def generate_uio_encoding_rec(A, uio, n, i):
            if i == n:
                A.append(uio)
                return
            for j in range(uio[i-1], i+1):
                generate_uio_encoding_rec(A, uio+[j], n, i+1)

        A = []
        generate_uio_encoding_rec(A, [0], n, 1)
        #print("this is the one 2379:", A[2379])
        print("Generated", len(A), "unit interval order encodings")
        return A
    
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

if __name__ == "__main__":
    Preparer = GlobalUIODataPreparer(6,CoreGenerator)
    print("here")
    X,y = Preparer.computeTrainingData((4,2))
    #X,y = Preparer.loadTrainingData("SPC/Saves,Tests/Preparersavetest.bin")
    #for key in X:
        #print(key, Preparer.coreRepresentationsCategorizer[key])
    print(y)
    Preparer.saveTrainingData("SPC/Saves,Tests/Preparersavetest.bin")
