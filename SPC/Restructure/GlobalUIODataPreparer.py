from SPC.Restructure.UIODataExtractor import UIODataExtractor
from SPC.Restructure.UIO import UIO
from SPC.misc.extra import PartiallyLoadable

class GlobalUIODataPreparer(PartiallyLoadable):

    def __init__(self, n):
        super().__init__(["coreRepresentationsCategorizer", "coefficients"])
        self.n = n
        self.coreRepresentationsCategorizer = {} # {coreRepresentation1:{UIOID1:occurences_in_UIOID1, UIOID2:occurences_in_UIOID2}}
        self.coefficients = [] # list of all coefficients (for the partition given to getTrainingData)

    def initUIOs(self):
        encodings = self.generate_all_uio_encodings(self.n)
        self.extractors = [UIODataExtractor(UIO(enc)) for enc in encodings]

    def computeTrainingData(self, partition:tuple) -> tuple:
        """
        :return: (X,y) the training data. Where X is a dict with the keys being coreRepresentations, and 
        the values being a dict describing the occurences in each UIO. 
        
        In contrast to normal supervised learning input data, the data isn't ordered from first to last 
        input instance as a list, but it is the data of all input instances at once in a dictionary, 
        with keys not being the input instances (UIOs), but  some data (core representations) collected 
        among all UIOs, where we count the occurences coming from each individual UIO.
        
        y is a list with the coefficient of each UIO.
        """
        # uios initialized?
        if not hasattr(self, "extractors"):
            self.initUIOs()
        # count correps and compute coefficients
        self.countCoreRepresentations(partition)
        self.coefficients = [extractor.getCoefficient(partition) for extractor in self.extractors]
    
    def loadTrainingData(self, filepath:str) -> tuple:
        self.load(filepath)

    def getTrainingData(self):
        return self.coreRepresentationsCategorizer, self.coefficients
    
    def saveTrainingData(self, filepath:str) -> None:
        self.save(filepath)
    
    def getAllCorrectSequenceCoreRepresentations(self, partition):
        return [extractor.getCorrectSequenceCoreRepresentations(partition) for extractor in self.extractors]

    def generate_all_uio_encodings(self, n):

        def generate_uio_encoding_rec(A, uio, n, i):
            if i == n:
                A.append(uio)
                return
            for j in range(uio[i-1], i+1):
                generate_uio_encoding_rec(A, uio+[j], n, i+1)

        A = []
        generate_uio_encoding_rec(A, [0], n, 1)
        print("Generated", len(A), "unit order intervals encodings")
        return A
    
    def countCoreRepresentations(self, partition):
        print("Categorizing core representations...")
        core_rep_categories = {} # coreRepresentation:ID
        counter = {} # corerepID:dict(uioID1:occurrences1, uioID2:occurrences2)
        all_corereps = self.getAllCorrectSequenceCoreRepresentations(partition)
        print("Found", len(all_corereps), "core representations")

        ID = 0
        for uioID, corereps in enumerate(all_corereps):
            for corerep in corereps:

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

        # change keys from an ID of the coreRepresentation to the coreRepresentation itself
        for cat in core_rep_categories:
            self.coreRepresentationsCategorizer[cat] = counter[core_rep_categories[cat]]

        print("Found",len(self.coreRepresentationsCategorizer), "distinct core representations / categories")

if __name__ == "__main__":
    Preparer = GlobalUIODataPreparer(6)
    print("here")
    # X,y = Preparer.computeTrainingData((4,2))
    X,y = Preparer.loadTrainingData("SPC/Saves,Tests/Preparersavetest.bin")
    #for key in X:
        #print(key, Preparer.coreRepresentationsCategorizer[key])
    print(y)
    Preparer.saveTrainingData("SPC/Saves,Tests/Preparersavetest.bin")
