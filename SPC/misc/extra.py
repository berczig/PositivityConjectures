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
    def __init__(self, saveable_variables, default_values = None):
        self.saveable_variables = saveable_variables
        self.default_values = {} if default_values == None else default_values
        self._savehelper = Loadable()

    def save(self, filename):
        for var in self.saveable_variables:
            setattr(self._savehelper, var, getattr(self, var))
        self._savehelper.save(filename)

    def load(self, filename):
        self._savehelper.load(filename)
        for var in self.saveable_variables:
            if hasattr(self._savehelper, var):
                setattr(self, var, getattr(self._savehelper, var))
            else:
                value = self.default_values.get(var, None)
                print("\n"+5*"#"+"\n"+f"THE FILE \"{filename}\" IS OUTDATED!\nIT IS MISSING THE ATTRIBUTE \"{var}\""+"\n"+f"SETTING {var} = {value}"+"\n"+5*"#"+"\n")
                setattr(self, var, value)
