from SPC.Restructure.UIO import UIO
class CoreGenerator:

    def __init__(self, uio:UIO, partition):
        self.uio = uio
        self.partition = partition

    def generateCore(self, seq):
        """
        abstract function
        """
        pass

    def getCoreLabels(partition):
        """
        abstract function
        """
        pass