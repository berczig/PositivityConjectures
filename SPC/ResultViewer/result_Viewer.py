from nicegui import ui
import os
import numpy as np
import plotly.graph_objects as go
from SPC.UIO.ModelLogger import ModelLogger
from SPC.UIO.UIO import UIO
from SPC.UIO.cores.CoreGenerator import CoreGenerator
from SPC.misc.extra import PartiallyLoadable
from SPC.UIO.GlobalUIODataPreparer import GlobalUIODataPreparer
from SPC.UIO.FilterEvaluator import FilterEvaluator
from functools import lru_cache
from pathlib import Path
import asyncio
import time
import SPC
from datetime import datetime
from pathlib import Path
import importlib
import networkx as nx
from matplotlib import pyplot as plt

last_sort = ""
last_toggle = False

def load_trainingdata(trainingdata_file_path):
    preparer = GlobalUIODataPreparer(0)
    if trainingdata_file_path != "":
        print("loading model...")
        preparer.loadTrainingData(trainingdata_file_path)
    return preparer

def load_model(model_file_path):
    modelLogger = ModelLogger()
    if model_file_path != "":
        print("loading model...")
        modelLogger.load_model_logger(model_file_path)
    return modelLogger


# Function to read JSON files and extract the "name" field and file modification time (date)
@lru_cache(maxsize=None)
def load_json_files_cache(folder_path):
    return load_json_files(folder_path)

@lru_cache(maxsize=None)
def load_json_files_cache2(folder_path):
    return load_json_files2(folder_path)

def load_json_files(folder_path):
    print("load_json_files")
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".keras"):
            file_path = os.path.join(folder_path, file_name)
            print("file_name:", file_name)
            modelLogger = load_model(file_path)
            results.append((file_name, modelLogger))
    return results

def load_json_files2(folder_path):
    print("load_json_files2")
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".bin"):
            file_path = os.path.join(folder_path, file_name)
            print("file_name:", file_name)
            trainingdata = load_trainingdata(file_path)
            results.append((file_name, trainingdata))
    return results

def prepVE(V, edges):
    # Remove vertices without edges and translate edge from ID to labels and turn GR to LE
    edges_new = []
    V_new = []
    for (i,j,edgetype) in edges:
        if V[i] not in V_new:
            V_new.append(V[i])
        if V[j] not in V_new:
            V_new.append(V[j])
        if edgetype == UIO.GREATER:
            edges_new.append((V[j], V[i], UIO.LESS))
        else:
            edges_new.append((V[i], V[j], edgetype))
    return V_new, edges_new


def plot_directed_graph(V, E, V_groups, colors):
    """
    Plots multiple directed graphs with the same vertices but different edges in a single figure.
    
    Parameters:
    - V: List of vertices (nodes).
    - edges_list: A list of lists, where each inner list is a set of directed edges.
    - V_groups classifing nodes into groups to color them
    - colors: color of each group
    """
    
    # Remove vertices without edges and translate edge from ID to labels
    V, E = prepVE(V, E)

    edges = [x[:2] for x in E]
    edges_labels = dict([((a,b), UIO.RELATIONTEXT2[edgetype]) for (a,b,edgetype) in E if edgetype == UIO.EQUAL])
    
    # get label for each vertex
    vert_labels = {}
    for vertex in V:
        for group in V_groups:
            if vertex in V_groups[group]:
                vert_labels[vertex] = V_groups[group][vertex]
                break

    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(edges)

    pos = nx.circular_layout(G)

    for group in V_groups:
        NL = [vertex_name for vertex_name in V_groups[group].keys() if vertex_name in V]
        nx.draw_networkx_nodes(G, pos, nodelist=NL, node_color=colors[group], label=group, node_size=700)
        #nx.draw(G, pos, with_labels=False, node_size=700, node_color='skyblue', edgelist = edges,
                    #edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=25, font_size=16, font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist = edges,node_size=700,
                    edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=30)
    nx.draw_networkx_labels(G, pos,labels=vert_labels,font_color='black', font_size=16)
    nx.draw_networkx_edge_labels(G, pos,edge_labels=edges_labels,font_color='red', font_size=16)
    
    plt.legend()
    plt.tight_layout()


# Function to display the details of a selected result
def display_details(result, container, plotgraphs=True):
    filename, model = result
    model : ModelLogger
    container.clear()  # Clear previous details
    figs = []

    with container:
        #ui.label(f"Name: {result['name']}").style('font-size: 18px; font-weight: bold;')
        # Display other fields if needed
        
        with ui.column():

            with ui.row():
                # Info Text
                with ui.column():
                    ui.label(f"Model: {filename}").style('font-size: 24px; font-weight: bold;')
                    ui.label(f"Uio length: {model.uio_size}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Partition: {model.partition}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Final Graph score: {model.bestscore_history[-1]}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Final Total Residual Absolute Sum: { np.sum(np.abs(model.residuals))}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Final Perfect UIO coef. predictions: {np.sum(model.residuals == 0)}/{len(model.residuals)}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Iterations: {model.step}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"core_generator_type: {model.core_generator_type}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Graphs per RL session: {model.RL_n_graphs}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Condition rows per graph: {model.condition_rows}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Core length: {model.core_length}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Core representation length: {model.core_representation_length}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Edge Penalty: {model.edgePenalty}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Training algorithm: {model.ml_training_algorithm_type}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Model type: {model.ml_model_type}").style('font-size: 18px; font-weight: bold;')
                    d = datetime.fromtimestamp(model.last_modified)
                    ui.label(f"Last modified: {d.day}.{d.month}.{d.year} {d.hour}:{d.minute}:{d.second}").style('font-size: 18px; font-weight: bold;')

                    if model.some_wrong_uios != None:
                        text = "<br>".join([f"{encod} has res = {res}" for (encod, res) in model.some_wrong_uios])
                        ui.html(f"Non-zero UIOs: <br> {text}").style('font-size: 18px; font-weight: bold;')
                        #ui.label(f"Non-zero UIOs: {text}").style('font-size: 18px; font-weight: bold;')

                    ui.button("Export data", on_click= lambda:export_data(result, figs)).style('font-size: 22px; font-weight: bold;')


                # PLots
                x = list(range(1, len(model.bestscore_history)+1))
                if quickswitch_checkbox.value == False and plotgraphs:
                    with ui.pyplot(figsize=(8, 6)) as nicefig:
                        y = model.bestscore_history
                        fig = nicefig.fig
                        plt.plot(x, y, '-')

                        ax = nicefig.fig.gca()
                        ax.set_xlabel("iteration step")
                        ax.set_ylabel("Penalized Score")
                        ax.set_title("Best Penalized Score")
                        figs.append(("score", fig))

                    if model.residual_score_history != None:
                        with ui.pyplot(figsize=(8, 6)) as nicefig:
                            y = model.residual_score_history
                            fig = nicefig.fig
                            plt.plot(x, y, '-')

                            ax = nicefig.fig.gca()
                            ax.set_xlabel("iteration step")
                            ax.set_ylabel("Residual Absolute Sum")
                            ax.set_title("Residual Absolute Sum")
                            figs.append(("residual", fig))

                    if model.perfect_coef_history != None:
                        with ui.pyplot(figsize=(8, 6)) as nicefig:
                            y = model.perfect_coef_history
                            fig = nicefig.fig
                            plt.plot(x, y, '-')

                            ax = nicefig.fig.gca()
                            ax.set_xlabel("iteration step")
                            ax.set_ylabel("Perfect Coef. Predictions")
                            ax.set_title("Perfect Coef. Predictions")
                            figs.append(("perfect_coef", fig))

                    if model.graphsize_history != None:
                        with ui.pyplot(figsize=(8, 6)) as nicefig:
                            y = model.graphsize_history
                            fig = nicefig.fig
                            plt.plot(x, y, '-')

                            ax = nicefig.fig.gca()
                            ax.set_xlabel("iteration step")
                            ax.set_ylabel("best graph #edges")
                            ax.set_title("Number of edges in best graph")
                            figs.append(("graph_size", fig))

                        

            # networkx Graphs
            if quickswitch_checkbox.value == False and plotgraphs:
                with ui.column():
                    ui.label("Best Graph:").style('font-size: 24px; font-weight: bold;')
                    with ui.column():
                        with ui.column().classes('border bg-blue-250 p-4'):
                            ui.label("Graph Conditions:").style('font-size: 22px; font-weight: bold;')
                            conditions = [text for i, text in enumerate(model.current_bestgraph.split("\n")) if i%2 == 1]
                            for condition in conditions:
                                ui.label(f"{condition}").style('font: Courier; font-size: 18px; font-weight: bold;')
                        if model.graph_edges != None and model.graph_vertices != None:
                            with ui.column().classes('border bg-blue-250 p-4'):
                                for graphID, E in enumerate(model.graph_edges):
                                    with ui.pyplot(figsize=(10, 10)) as nicefig:
                                        fig = nicefig.fig
                                        ax = nicefig.fig.gca()
                                        ax.set_title(f"Graph {graphID+1}/{len(model.graph_edges)}")
                                        figs.append((f"graph {graphID+1}", fig))

                                        core_generator_class_ = getattr(importlib.import_module("SPC.UIO.cores."+model.core_generator_type), model.core_generator_type)
                                        core_generator_class_ : CoreGenerator
                                        labelgroups = core_generator_class_.getCoreLabelGroups(model.partition)
                                        
                                        plot_directed_graph(V = model.graph_vertices, E = E, V_groups= labelgroups, colors=core_generator_class_.getCoreLabelGroupColors(model.partition))


            
            """with ui.column().classes('border bg-gray-250 p-4'):
                ui.label(f"Best Penalized Score:").style('font-size: 24px; font-weight: bold;')
                fig = go.Figure(go.Scatter(
                    x=list(range(len(model.bestscore_history))), 
                    y=model.bestscore_history))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                ui.plotly(fig)

            with ui.column().classes('border bg-gray-250 p-4'):
                ui.label(f"Residual Absolute Sum:").style('font-size: 24px; font-weight: bold;')
                fig = go.Figure(go.Scatter(
                    x=list(range(len(model.residual_score_history))), 
                    y=model.residual_score_history))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                ui.plotly(fig)

            with ui.column().classes('border bg-gray-250 p-4'):
                ui.label(f"Perfect Coef. Predictions:").style('font-size: 24px; font-weight: bold;')
                fig = go.Figure(go.Scatter(
                    x=list(range(len(model.perfect_coef_history))), 
                    y=model.perfect_coef_history))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                ui.plotly(fig)"""



# Function to display the details of a selected result
def display_details2(result):
    filename, data = result
    data : GlobalUIODataPreparer
    details_container2.clear()  # Clear previous details
    figs = []

    with details_container2:
        #ui.label(f"Name: {result['name']}").style('font-size: 18px; font-weight: bold;')
        # Display other fields if needed
        
        with ui.column():

            with ui.row():
                # Info Text
                with ui.column():
                    ui.label(f"Training Data: {filename}").style('font-size: 24px; font-weight: bold;')
                    ui.label(f"UIO length: {data.uio_size}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Partition: {data.partition}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"UIOs: {len(data.coefficients)}").style('font-size: 18px; font-weight: bold;')        
                    ui.label(f"Coeff. Sum: {np.sum(data.coefficients)}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Distinct Core Representations: {len(data.coreRepresentationsCategorizer)}").style('font-size: 18px; font-weight: bold;')





async def export_data(result, figs):
    # notification
    n = ui.notification(timeout=None)
    n.message = "Exporting..."
    n.spinner = True

    # create folder
    filename, model = result
    path = os.path.join(folder_input.value, os.path.splitext(filename)[0])
    Path(path).mkdir(parents=True, exist_ok=True)

    print(os.path.join(path, "score_plot.png"))
    for name, fig in figs:
        fig.savefig(os.path.join(path, name+".png"), bbox_inches='tight')   # save the figure to file
    
    # noti 2
    n.message = 'Finished exporting!'
    n.spinner = False
    await asyncio.sleep(1.8)
    n.dismiss()


# Function to display results in a scrollable grid
def display_results(results):
    global option
    result_container.clear()  # Clear previous results
    # Create a scrollable grid of results (4 squares per row)
    with result_container:
        with ui.row():
            for index, result in enumerate(results):
                filename, model = result
                with ui.card().on('click', lambda _, r=result: display_details(r, details_container)):  # Make each card clickable
                    if Sortoption == "by name":
                        ui.label(f"{filename}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by partition":
                        ui.label(f"{model.partition}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by iteration":
                        ui.label(f"{model.step}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by score":
                        ui.label(f"{model.bestscore_history[-1]}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by date":
                        d = datetime.fromtimestamp(model.last_modified)
                        ui.label(f"{d.day}.{d.month}.{d.year}").style('text-align: center; font-weight: bold;')
                    #ui.label(f"{model.step}").style('text-align: center; font-weight: bold;')

# Function to display results in a scrollable grid
def display_results2(results):
    global option
    result_container2.clear()  # Clear previous results
    # Create a scrollable grid of results (4 squares per row)
    with result_container2:
        with ui.row():
            for index, result in enumerate(results):
                filename, data = result
                with ui.card().on('click', lambda _, r=result: display_details2(r)):  # Make each card clickable
                    ui.label(f"{filename}").style('text-align: center; font-weight: bold;')
                    ui.label(f"{data.partition}").style('text-align: center; font-weight: bold;')
                    #ui.label(f"{model.step}").style('text-align: center; font-weight: bold;')

# Sorting functions
def sort_by_name(results, reverse=True):
    return sorted(results, key=lambda x: x[0].lower(), reverse=reverse)  # Sort alphabetically by name

def sort_by_partition(results, reverse=True):
    return sorted(results, key=lambda x: x[1].partition, reverse=True)  # Sort by date (newest first)

def sort_by_iteration(results, reverse=True):
    return sorted(results, key=lambda x: x[1].step, reverse=True)

def sort_by_score(results, reverse=True):
    return sorted(results, key=lambda x: x[1].bestscore_history[-1], reverse=True)

def sort_by_date(results, reverse=True):
    return sorted(results, key=lambda x: x[1].last_modified, reverse=True)


# Select the models with the best residual score
# Any distinct attributes tuple (dictated by distinctAttributes) gets its own row in the table
def selectModels(distinctAttributes):
    models = load_json_files_cache(folder_input.value)

    data = {} # {attr-tuple:(model, filename)}   
    for filename, modellogger in models:
        modellogger:ModelLogger

        # We want to use the same model in a row, so models with bigger uio sizes will be 
        # filtered away
        if modellogger.uio_size > sum(modellogger.partition):
            continue

        attributes = tuple([getattr(modellogger, attr) for attr in distinctAttributes])

        if attributes in data:
            # better than previous?
            if modellogger.residual_score_history[-1] < data[attributes][0].residual_score_history[-1]:
                data[attributes] = (modellogger, filename)
        else:
            data[attributes] = (modellogger, filename)
    return data

def getModelsByPartitionLength():
    models = load_json_files_cache(folder_input.value)

    data = {} # {partition-length:(modelnames, bestmodel)}   
    prevbest = None
    for filename, modellogger in models:
        modellogger:ModelLogger
        if modellogger.current_bestgraph_matrix is None:
            continue
        plength = len(modellogger.partition)
        if plength in data:
            data[plength][0].append(filename)
            # better than previous?
            if prevbest == None or modellogger.residual_score_history[-1] < prevbest:
                data[plength][1] = filename
                prevbest = modellogger.residual_score_history[-1]
        else:
            data[plength] = [[filename], filename]
    return data

def getTrainingdatasets():
    trainingdatasets = load_json_files_cache2(folder_input2.value)
    tds_ordered = {} # {partition: {uio_size:trainingdataset}}
    for filename, tds in trainingdatasets:
        tds:GlobalUIODataPreparer
        if tds.partition in tds_ordered:
            tds_ordered[tds.partition][tds.uio_size] = tds
        else:
            tds_ordered[tds.partition] = {tds.uio_size:tds}
    return tds_ordered


class ResultViewerTable(PartiallyLoadable):

    def __init__(self):
        super().__init__(["table"])
        self.path = os.path.join(Path(SPC.__file__).parent, "ResultViewer", "resultviewer_table.pickle")
        self.table = {} # {(model, partition):{uio_size:(resscore, perfectpredict, coeffsum, numberOfUIOs)} }
        self.smallest_uio_size = None
        self.biggest_uio_size = None
        self.tables = {}
        self.tableTitles = {"res":"Residual scores", "perfect":"Perfect coef. predictions"}

    def cleanUp(self):
        for tt in self.tables:
            table = self.tables[tt]
            table.remove(table)

    def createTable(self, scoretype):
        with tableresultscontainer:
            columns=[{'name': 'partition', 'label': 'Partition', 'field': 'partition'}]
            for uio_size in range(self.smallest_uio_size, self.biggest_uio_size+1):
                columns.append({'name': f'{uio_size}uio_size', 'label': f'{uio_size} uio_size', 'field': f'{uio_size}uio_size'})

            tabledata = []

            for model, partition in sorted(self.table, key=lambda x: (len(x[1]), x[1])):
                model:ModelLogger
                rows = model.condition_rows
                scores = self.table[(model, partition)]

                item = {"partition":f"{partition} ({rows} rows)"}
                for uio_size in scores:
                    resscore, perfectpredict, coeffsum, numberOfUIOs = scores[uio_size]
                    if scoretype == "perfect":
                        item[f"{uio_size}uio_size"] = f"{resscore:.0f}/{coeffsum}"
                    elif scoretype == "res":
                        item[f"{uio_size}uio_size"] = f"{perfectpredict:.0f}/{numberOfUIOs}"
                tabledata.append(item)

            
            table = ui.table(rows=tabledata, columns=columns, title=self.tableTitles[scoretype])
            self.tables[scoretype] = table




    def save(self):
        self.save(self.path)

    def load(self):
        if os.path.isfile(self.path):
            self.load(self.path)

    def addEntry(self, model, partition, uio_size, resscore, perfectpredict, coeffsum, numberOfUIOs):
        #print("addEntry", "uio size:", uio_size, "coeffsum:", coeffsum)
        key = (model, partition)
        if key not in self.table:
            self.table[key] = {}
        self.table[key][uio_size] = (resscore, perfectpredict, coeffsum, numberOfUIOs)

        if self.smallest_uio_size is None: 
            self.smallest_uio_size = uio_size
            self.biggest_uio_size = uio_size
        else:
            self.smallest_uio_size = min(uio_size, self.smallest_uio_size)
            self.biggest_uio_size = max(uio_size, self.biggest_uio_size)

    def __repr__(self):
        return repr(self.table)


def EvaluateModelOnTrainingData():
    print("EvaluateModelOnTrainingData:")
    global rvTable
    if rvTable is not None:
        rvTable.cleanUp()
    rvTable = ResultViewerTable()
    all_modles = dict(load_json_files_cache(folder_input.value))
    #models = selectModels(["partition", "condition_rows"])
    trainingdatasets = getTrainingdatasets()
    for modelname in [partition_2l_select.value, partition_3l_select.value, partition_4l_select.value]:
        if modelname is None:
            continue

        model = all_modles[modelname]
        model:ModelLogger
        modelpart = model.partition
        for part in trainingdatasets:
            if len(part) == len(modelpart):
                tdsp = trainingdatasets[part]
                for uio_size in tdsp:

                    tds = tdsp[uio_size]
                    tds:GlobalUIODataPreparer
                    FE = FilterEvaluator(coreRepresentationsCategorizer=tds.coreRepresentationsCategorizer,
                        true_coefficients=tds.coefficients, 
                        ignore_edge=FilterEvaluator.DEFAULT_IGNORE_VALUE,
                        model_logger=model, verbose=False)
                    
                    if not model.current_bestgraph_matrix is None:
                        residuals = FE.evaluate(filter=model.current_bestgraph_matrix, return_residuals=True)
                        resscore = np.sum(np.abs(residuals))
                        perfectpredict = np.sum(residuals==0)
                        rvTable.addEntry(model, part, uio_size, resscore, perfectpredict, np.sum(tds.coefficients), len(tds.coefficients))
                    else:
                        print(f"model {modelname} has no attribute \"current_bestgraph_matrix\"")
    print("rvTable:", rvTable)




# Function to sort and display results based on the selected option
Sortoption = "by name"
def sort_and_display_results(folder_path, option):
    global Sortoption
    Sortoption = option
    results = load_json_files_cache(folder_path)
    if option == 'by name':
        results = sort_by_name(results)
    elif option == 'by partition':
        results = sort_by_partition(results)
    elif option == 'by iteration':
        results = sort_by_iteration(results)
    elif option == 'by score':
        results = sort_by_score(results)
    elif option == 'by date':
        results = sort_by_date(results)
    display_results(results)


def updateTables():
    global rvTable
    rvTable.createTable("res")
    rvTable.createTable("perfect")

def setSelect():
    data = getModelsByPartitionLength()

    if 2 in data:
        partition_2l_select.set_options(data[2][0], value=data[2][1])
    if 3 in data:
        partition_3l_select.set_options(data[3][0], value=data[3][1])
    if 4 in data:
        partition_4l_select.set_options(data[4][0], value=data[4][1])

def updateModelSelect(model_file_name):
    EvaluateModelOnTrainingData()
    updateTables()

    all_modles = dict(load_json_files_cache(folder_input.value))
    display_details((model_file_name, all_modles[model_file_name]), details_container_results, plotgraphs=False)

def refresh_page():
    display_results(load_json_files(folder_input.value))
    display_results2(load_json_files2(folder_input2.value))
    setSelect()

# Load ResultViewer Data
global rvTable
rvTable = None
#rvTable.load()

# Main UI layout

with ui.tabs().classes('w-full') as tabs:
    Mtab = ui.tab('Models')
    Dtab = ui.tab('Data')
    Rtab = ui.tab('Results')

with ui.tab_panels(tabs, value=Mtab).classes('w-full'):
    with ui.tab_panel(Mtab):
        quickswitch_checkbox = ui.checkbox('quick model switch (no graphs)', value=False)
        folder_input = ui.input(label='Model Input Path', placeholder='Enter or select a folder...',
                                value=os.path.join(Path(SPC.__file__).parent, "Saves,Tests", "models")).classes("w-96")
        ui.label('Models:').style('font-size: 24px; margin-bottom: 10px;')
        with ui.column().classes("w-full"):

            result_container = ui.scroll_area().classes('w-512')  # Scrollable area to display results

            # Sorting dropdown and button
            with ui.row():
                sorting_option = ui.select(["by name", 'by partition', "by iteration", "by score", "by date"], label='Sort by', value="by name")
                ui.button('Sort', on_click=lambda: sort_and_display_results(folder_input.value, sorting_option.value))

            ui.button('Load Results', on_click=lambda: refresh_page())

        
            details_container = ui.column().classes('w-128 h-fill border bg-gray-100 p-4')


    with ui.tab_panel(Dtab):
        folder_input2 = ui.input(label='Training Data Input Path', placeholder='Enter or select a folder...',
                                value=os.path.join(Path(SPC.__file__).parent, "Saves,Tests", "Trainingdata")).classes("w-96")
        ui.label("Training data:")
        result_container2 = ui.scroll_area().classes('w-512')  # Scrollable area to display results
        details_container2 = ui.column().classes('w-128 h-fill border bg-gray-100 p-4')


    with ui.tab_panel(Rtab):
        #with ui.row().classes("w-full"):
        partition_2l_select = ui.select(options=[], with_input=True, on_change=lambda: updateModelSelect(partition_2l_select.value), label="2-partition model")
        partition_3l_select = ui.select(options=[], with_input=True, on_change=lambda: updateModelSelect(partition_3l_select.value), label="3-partition model")
        partition_4l_select = ui.select(options=[], with_input=True, on_change=lambda: updateModelSelect(partition_4l_select.value), label="4-partition model")

        with ui.row().classes("w-full"):
            tableresultscontainer = ui.element("div")
            details_container_results = ui.column().classes('w-128 h-fill border bg-gray-100 p-4')

        #display_table(*get_table_data("res"), "Residual scores")
        #display_table(*get_table_data("perfect"), "Perfect coef. predictions")


setSelect()

display_results(load_json_files_cache(folder_input.value))
display_results2(load_json_files_cache2(folder_input2.value))




# Run the app
ui.page_title('SPC Model Viewer')
ui.run()
