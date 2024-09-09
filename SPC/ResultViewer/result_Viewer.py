from nicegui import ui
import os
import numpy as np
import plotly.graph_objects as go
from SPC.Restructure.ModelLogger import ModelLogger
from functools import lru_cache
from pathlib import Path

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

# Function to display the details of a selected result
def display_details(result):
    filename, model = result
    details_container.clear()  # Clear previous details
    with details_container:
        #ui.label(f"Name: {result['name']}").style('font-size: 18px; font-weight: bold;')
        # Display other fields if needed
        with ui.row():
            with ui.column():
                ui.label(f"Model: {filename}").style('font-size: 24px; font-weight: bold;')
                ui.label(f"Partition: {model.partition}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Final Graph score: {model.bestscore_history[-1]}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Total Residual Absolute Sum: { np.sum(np.abs(model.residuals))}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Perfect UIO coef. predictions: {np.sum(model.residuals == 0)}/{len(model.residuals)}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Iterations: {model.step}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"core_generator_type: {model.core_generator_type}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Graphs per RL session: {model.RL_n_graphs}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Condition rows per graph: {model.condition_rows}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Core length: {model.core_length}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Core representation length: {model.core_representation_length}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Training algorithm: {model.ml_training_algorithm_type}").style('font-size: 18px; font-weight: bold;')
                ui.label(f"Model type: {model.ml_model_type}").style('font-size: 18px; font-weight: bold;')
            
            with ui.column().classes('border bg-gray-250 p-4'):
                ui.label(f"Best Score:").style('font-size: 24px; font-weight: bold;')
                fig = go.Figure(go.Scatter(
                    x=list(range(len(model.bestscore_history))), 
                    y=model.bestscore_history))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                ui.plotly(fig)

            with ui.column().classes('border bg-blue-250 p-4'):
                ui.label("Best Graph:").style('font-size: 24px; font-weight: bold;')
                conditions = [text for i, text in enumerate(model.current_bestgraph.split("\n")) if i%2 == 1]
                for condition in conditions:
                    ui.label(f"{condition}").style('font: Courier; font-size: 18px; font-weight: bold;')


        ui.button("Export data", on_click= lambda:export_data(result, fig))

def export_data(result, fig):
    filename, model = result
    path = os.path.join(folder_input.value, os.path.splitext(filename)[0])
    Path(path).mkdir(parents=True, exist_ok=True)

    print(os.path.join(path, "score_plot.png"))
    #fig.show()
    #fig.write_image(os.path.join(path, "score_plot.png"))


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
ui.run()
