from nicegui import ui
import os
import numpy as np
import plotly.graph_objects as go
from SPC.Restructure.ModelLogger import ModelLogger
from SPC.Restructure.UIO import UIO
from functools import lru_cache
from pathlib import Path
import asyncio
import networkx as nx
from matplotlib import pyplot as plt

# "step", "all_scores", "bestscore_history", "meanscore_history", "bestfilter_history", "calculationtime_history", "partition", "core_generator_type", "core_length", "core_representation_length"
ModelLogger_attributes = ["partition", "step"]
last_sort = ""
last_toggle = False

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

def prepVE(V, edges):
    # Remove vertices without edges and translate edge from ID to labels
    edges_new = []
    V_new = []
    for (i,j,edgetype) in edges:
        edges_new.append((V[i], V[j], edgetype))
        if V[i] not in V_new:
            V_new.append(V[i])
        if V[j] not in V_new:
            V_new.append(V[j])
    return V_new, edges_new


def plot_directed_graph(V, E):
    """
    Plots multiple directed graphs with the same vertices but different edges in a single figure.
    
    Parameters:
    - fig: A pre-defined matplotlib figure.
    - V: List of vertices (nodes).
    - edges_list: A list of lists, where each inner list is a set of directed edges.
    """

    # Remove vertices without edges and translate edge from ID to labels
    V, E = prepVE(V, E)

    edges = [x[:2] for x in E]
    edges_labels = dict([((a,b), UIO.RELATIONTEXT2[edgetype]) for (a,b,edgetype) in E])

    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(edges)

    pos = nx.circular_layout(G)

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edgelist = edges,
                edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=25, font_size=16, font_color='black')
    nx.draw_networkx_edge_labels(G, pos,edge_labels=edges_labels,font_color='red', font_size=16)
    
    plt.tight_layout()


# Function to display the details of a selected result
def display_details(result):
    filename, model = result
    model : ModelLogger
    details_container.clear()  # Clear previous details
    figs = []

    with details_container:
        #ui.label(f"Name: {result['name']}").style('font-size: 18px; font-weight: bold;')
        # Display other fields if needed
        
        with ui.column():

            with ui.row():
                # Info Text
                with ui.column():
                    ui.label(f"Model: {filename}").style('font-size: 24px; font-weight: bold;')
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
                    ui.label(f"Training algorithm: {model.ml_training_algorithm_type}").style('font-size: 18px; font-weight: bold;')
                    ui.label(f"Model type: {model.ml_model_type}").style('font-size: 18px; font-weight: bold;')

                    ui.button("Export data", on_click= lambda:export_data(result, figs)).style('font-size: 22px; font-weight: bold;')

                # PLots
                with ui.pyplot(figsize=(8, 6)) as nicefig:
                    x = list(range(len(model.bestscore_history)))
                    y = model.bestscore_history
                    fig = nicefig.fig
                    plt.plot(x, y, '-')

                    ax = nicefig.fig.gca()
                    ax.set_xlabel("iteration step")
                    ax.set_ylabel("Penalized Score")
                    ax.set_title("Best Penalized Score")
                    figs.append(("score", fig))

                if model.residual_score_history != None and model.perfect_coef_history != None:
                    with ui.pyplot(figsize=(8, 6)) as nicefig:
                        x = list(range(len(model.bestscore_history)))
                        y = model.residual_score_history
                        fig = nicefig.fig
                        plt.plot(x, y, '-')

                        ax = nicefig.fig.gca()
                        ax.set_xlabel("iteration step")
                        ax.set_ylabel("Residual Absolute Sum")
                        ax.set_title("Residual Absolute Sum")
                        figs.append(("residual", fig))

                    with ui.pyplot(figsize=(8, 6)) as nicefig:
                        x = list(range(len(model.bestscore_history)))
                        y = model.perfect_coef_history
                        fig = nicefig.fig
                        plt.plot(x, y, '-')

                        ax = nicefig.fig.gca()
                        ax.set_xlabel("iteration step")
                        ax.set_ylabel("Perfect Coef. Predictions")
                        ax.set_title("Perfect Coef. Predictions")
                        figs.append(("perfect_coef", fig))

            # networkx Graphs
            with ui.column():
                ui.label("Best Graph:").style('font-size: 24px; font-weight: bold;')
                with ui.row():
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
                                    plot_directed_graph(V = model.graph_vertices, E = E)


            
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
                with ui.card().on('click', lambda _, r=result: display_details(r)):  # Make each card clickable
                    if Sortoption == "by name":
                        ui.label(f"{filename}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by partition":
                        ui.label(f"{model.partition}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by iteration":
                        ui.label(f"{model.step}").style('text-align: center; font-weight: bold;')
                    elif Sortoption == "by score":
                        ui.label(f"{model.bestscore_history[-1]}").style('text-align: center; font-weight: bold;')
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
    display_results(results)



# Main UI layout
folder_input = ui.input(label='Model Input Path', placeholder='Enter or select a folder...',
                                value=os.path.join(os.getcwd(), "SPC", "Saves,Tests", "models"))
ui.label('Models:').style('font-size: 24px; margin-bottom: 10px;')

with ui.row().classes('w-full gap-0'):  # Create a row layout for the grid and detail section to be side by side
    with ui.column().classes('w-full gap-0'):

        result_container = ui.scroll_area().classes('w-512')  # Scrollable area to display results

        # Sorting dropdown and button
        with ui.row():
            sorting_option = ui.select(["by name", 'by partition', "by iteration", "by score"], label='Sort by', value="by name")
            ui.button('Sort', on_click=lambda: sort_and_display_results(folder_input.value, sorting_option.value))

        ui.button('Load Results', on_click=lambda: display_results(load_json_files(folder_input.value)))

    # Create a section on the right to display details of a selected result
    with ui.column().classes('w-full gap-0'):
        details_container = ui.column().classes('w-128 h-fill border bg-gray-100 p-4')

    # Initial display of results
    display_results(load_json_files_cache(folder_input.value))


# Run the app
ui.page_title('SPC Model Viewer')
ui.run()
