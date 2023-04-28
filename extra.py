import pickle

class Loadable:
    """
    pickles, saves, loads all attributes of the superclass
    """

    def save(self, filename):
        # saves all currently present attributes of the instance
        with open(filename, 'wb') as f:
            pickle.dump(self,f)

    def load(self, filename):
        # loads dump and sets all attributes of dump as attributes of the current instance
        with open(filename, 'rb') as f:
            loaded_instance = pickle.load(f)
            for var in vars(loaded_instance):
                setattr(self, var, getattr(loaded_instance, var))

class PartiallyLoadable:
    """
    pickles, saves, loads some attributes of the superclass
    """
    def __init__(self, saveable_variables):
        self.saveable_variables = saveable_variables
        self._savehelper = Loadable()

    def save(self, filename):
        for var in self.saveable_variables:
            setattr(self._savehelper, var, getattr(self, var))
        self._savehelper.save(filename)

    def load(self, filename):
        self._savehelper.load(filename)
        for var in self.saveable_variables:
            setattr(self, var, getattr(self._savehelper, var))
