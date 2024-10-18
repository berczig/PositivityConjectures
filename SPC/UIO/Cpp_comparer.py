from SPC.misc.misc import refpath
from SPC.UIO.GlobalUIODataPreparer import GlobalUIODataPreparer
import os
import json

def load_model(model_file_path) -> GlobalUIODataPreparer:
    modelLogger = GlobalUIODataPreparer(0)
    if model_file_path != "":
        print("loading model...")
        modelLogger.load(model_file_path)
    return modelLogger

def getCPPData(path):
    jsonpath = os.path.join(refpath, path)

    # Opening JSON file
    with open(jsonpath) as f:
        data = json.load(f)
        #print(data)
        alldata = {}
        for dataset in data["data"]:
            corerep_cate = {}
            for x in dataset["core_representations"]:
                corerep_cate[tuple(x[0])] = dict(x[1])

            oneconfig_data = {"corereps":corerep_cate, "coefficients":dataset["coefficients"]}

            key = (tuple(dataset["partition"]),dataset["uio_size"])
            
            alldata[key] = oneconfig_data
    return alldata

def getAllPythonData(folder_path):
    P_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".bin"):
            file_path = os.path.join(folder_path, file_name)
            P = getPythonData(file_path)
            P_data[P[0]] = P[1]
    return P_data

def getPythonData(model_file_path):
    model = load_model(model_file_path)
    key = (model.partition,model.uio_size)
    data = {"coefficients":model.coefficients,
        "corereps":model.coreRepresentationsCategorizer}
    return (key,data)

def compareData(C_data, P_data):
    for key in C_data:
        print("key:", key)
        print("key in Python?:", key in P_data)
        if  key in P_data:
            C_entry = C_data[key]
            P_entry = P_data[key]
            print("coefficients same?:", C_entry["coefficients"]==P_entry["coefficients"])
            print("corereps keys same?:", 
                  sorted(list(C_entry["corereps"].keys()))==sorted(list(P_entry["corereps"].keys())))
            for k in C_entry["corereps"]:
                if k not in P_entry["corereps"]:
                    print("In C but not in P corereps:", k)
            print("corereps same?:", C_entry["corereps"]==P_entry["corereps"])
            #print("C corereps:", C_entry["corereps"].keys(), "\n")
            #print("P corereps:", P_entry["corereps"].keys(), "\n")
            #print("C:", C_entry["corereps"], "\n")
            #print("P:", P_entry["corereps"], "\n")
            print()

def saveData(data, folder):
    for key in data:
        print("save data", key)
        part, uio_size = key
        modelLogger = GlobalUIODataPreparer(uio_size)
        modelLogger.partition = part
        modelLogger.coefficients = data[key]["coefficients"]
        modelLogger.coreRepresentationsCategorizer = data[key]["corereps"]
        modelLogger.save(os.path.join(folder, "partition_{}_n{}.bin".format(part, uio_size)))
#C = getCPPData("SPC/Saves,Tests/C_Trainingdata/cpp_data_all_partitions_of_5.json")
#saveData(C, os.path.join(refpath, "SPC/Saves,Tests/C_Trainingdata"))

#P = getAllPythonData(os.path.join(refpath, "SPC/Saves,Tests/Trainingdata"))
#compareData(C, P)