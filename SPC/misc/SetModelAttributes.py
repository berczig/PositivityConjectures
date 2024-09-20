import numpy as np
from SPC.UIO.ModelLogger import ModelLogger
from SPC.UIO.GlobalUIODataPreparer import GlobalUIODataPreparer
from pathlib import Path
import os
import SPC


coeffsums = {4:16,
            5:42,
            6:286,
            7:1824,
            8:8344,
            9:124470}

def load_model(model_file_path) -> ModelLogger:
    modelLogger = ModelLogger()
    if model_file_path != "":
        print("loading model...")
        modelLogger.load(model_file_path)
    return modelLogger

def load_all_models(folder_path, file_extension):
    models = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_extension):
            file_path = os.path.join(folder_path, file_name)
            models.append((file_path, load_model(file_path)))
    return models


def changeValue(model:ModelLogger):
    pass
    #print(model.uio_size, model.coeffsum)
    model.coeffsum = coeffsums[model.uio_size]
    #print(model.uio_size, np.sum(model.coefficients))
    # updated uio_size
    #model.uio_size = sum(model.partition)

def updateModels(updatefunction, extension, relativepath):
    path = os.path.join(Path(SPC.__file__).parent, relativepath)
    print("path:", path)
    models = load_all_models(path, extension)

    for file_path, model in models:
        print(file_path)
        
        #changeValue in model
        changeValue(model)

        # overwrite model with its updated version
        model.save(file_path)


updateModels(changeValue, ".my_meta", "Saves,Tests/models")
#updateModels(changeValue, ".bin", "Saves,Tests/Trainingdata")