# Original version created by prog4321, Aug 2024 ==============================

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pyscript import document, display
import asyncio
from asyncio import sleep
from pyodide.http import open_url
import matplotlib.pyplot as plt
from pretty_html_table import build_table
from datetime import datetime

import numpy as np
import pandas as pd
from best_route_qlearner_model import BestRouteQLearner

sleep_interval = 0.001

show_dev_msg = False
# show_dev_msg = True

data_folder_path = './data/'
nodes_file = 'nodes.csv'
routes_file = 'routes.csv'
interchanges_file = 'interchanges.csv'

node_id_col = 0
node_name_col = 1

start_node_id_col = 0
end_node_id_col = 2
cost_col = 4

loading_text = document.getElementById("loading_text")

period_container = document.getElementById("period_container")
peak_dropdown = document.getElementById('peak_dropdown')

start_node_container = document.getElementById("start_node_container")
start_node_dropdown = document.getElementById('start_node_dropdown')

end_node_container = document.getElementById("end_node_container")
end_node_dropdown = document.getElementById('end_node_dropdown')

prediction_btn_container = document.getElementById("prediction_btn_container")
map_container = document.getElementById("map_container")

status_box = document.getElementById("status_box")
best_route_box = document.getElementById("best_route_box")
cost_box = document.getElementById("cost_box")
duration_box = document.getElementById('duration_box')
perf_log_box = document.getElementById('perf_log_box')
pred1_graph_box = document.getElementById('pred1_graph_box')
pred2_graph_box = document.getElementById('pred2_graph_box')
pred3_graph_box = document.getElementById('pred3_graph_box')
flag = document.getElementById('flag')

def extract_sorted_nodes():
    
    df_sorted_nodes = pd.read_csv(open_url(data_folder_path + nodes_file))
    sorted_nodes = df_sorted_nodes.iloc[:, [node_id_col, node_name_col]].values
    sorted_nodes = sorted_nodes[sorted_nodes[:,1].argsort()]
    
    return sorted_nodes

def populate_dropdown(sorted_nodes):
    
    # POPULATE THE DROPDOWN LISTS FOR THE START AND END STATIONS
    for node in sorted_nodes:
        
        node_name_id = node[1] + ' (' + node[0] + ')'
        node_id = node[0]
        
        entry_start = document.createElement('option')
        entry_start.textContent, entry_start.value = node_name_id, node_id
        start_node_dropdown.appendChild(entry_start)
        
        entry_end = document.createElement('option')
        entry_end.textContent, entry_end.value = node_name_id, node_id
        end_node_dropdown.appendChild(entry_end)

def make_containers_visible():
    
    period_container.style.visibility = 'visible'
    start_node_container.style.visibility = 'visible'
    end_node_container.style.visibility = 'visible'
    prediction_btn_container.style.visibility = 'visible'
    map_container.style.visibility = 'visible'
    flag.click()

populate_dropdown(extract_sorted_nodes())
make_containers_visible()

async def run_prediction(event):

    duration_box.innerText = ""
    perf_log_box.innerText = ""
    status_box.innerText = ""
    best_route_box.innerText = ""
    cost_box.innerText = ""
    pred1_graph_box.innerText = ""
    pred2_graph_box.innerText = ""
    pred3_graph_box.innerText = ""

    if show_dev_msg == True:
        start_time = datetime.now()
        display(f"PROGRAM TIMINGS:", target='duration_box', append=True)
        display(f"Start: {start_time.strftime('%I:%M:%S %p')}",
            target='duration_box', append=True)

    display('Initialising the model', target="status_box", append=True)
    
    df_nodes = pd.read_csv(open_url(data_folder_path + nodes_file))
    df_routes = pd.read_csv(open_url(data_folder_path + routes_file))
    df_interchanges = pd.read_csv(open_url(data_folder_path + interchanges_file))
    
    nodes = df_nodes.iloc[:, [node_id_col, node_name_col]].values
    routes = df_routes.iloc[:, [start_node_id_col, end_node_id_col, cost_col]].values
    
    interchanges_start_col = 1
    interchanges_col_count = df_interchanges.shape[1]
    interchanges = df_interchanges.iloc[:, interchanges_start_col:interchanges_col_count].values
        
    if peak_dropdown.value == 'peak_hr':
        is_peak_hour = True
        period_text = 'peak hour'
    else:
        is_peak_hour = False
        period_text = 'off-peak'
    
    # CREATE AN INSTANCE OF THE BestRouteQLearner CLASS AND PREDICT THE BEST PATH ============
    brq = BestRouteQLearner(alpha=1.0,
                            gamma=0.9,
                            epochs=30_000,
                            reward_coef=20,
                            is_peak_hour=is_peak_hour,
                            show_dev_msg=show_dev_msg)
    
    await brq.fit(nodes=nodes,
            routes=routes,
            interchanges=interchanges,
            node_id_alias='Station Code',
            node_name_alias='Station Name',
            cost_alias='Duration',
            cost_unit='min')
    
    start_node = start_node_dropdown.value
    end_node = end_node_dropdown.value
    
    is_valid_route, df_best_route = \
        await brq.predict(start_node=start_node, end_node=end_node)
    # =======================================================================================



    # DISPLAY RESULTS =======================================================================
    if is_valid_route == True:
    
        status_box.innerText = 'Best route from ' + start_node + \
                            ' (' + brq.get_node_name(start_node) + ') to ' + \
                            end_node + ' (' + brq.get_node_name(end_node) + '):'

        df_best_route = build_table(df_best_route, 'grey_light',
            font_size='10pt', font_family="'Sono', sans-serif")
        # df_best_route = build_table(df_best_route, 'grey_light')
        best_route_box.innerHTML = df_best_route
        
        cost_box.innerText = 'Total route will take ' + \
                        str(brq.total_cost) + ' ' + brq.cost_unit + \
                        ' during ' + period_text + ' period.'
    
    else:
        status_box.innerText = 'No valid route was found for Start Node ' + \
                            start_node + ' to End Node ' + end_node + '.'
    
    if show_dev_msg == True:
        end_time = datetime.now()
        duration = end_time - start_time
        duration = duration.total_seconds()
        display(f"End: {end_time.strftime('%I:%M:%S %p')}",
            target='duration_box', append=True)
        display(f"Duration: {duration:.2f}s",
            target='duration_box', append=True)
# =======================================================================================