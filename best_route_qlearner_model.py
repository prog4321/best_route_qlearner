# Original version created by prog4321, Aug 2024 ========================================

# APPLYING Q-LEARNING TO FIND THE BEST ROUTES IN SINGAPORE'S RAIL NETWORK (MRT)
# =============================================================================

# REFERENCES:
# 1. http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
# 2. https://amunategui.github.io/reinforcement-learning/index.html
# 3. 'Artificial Intelligence Programming with Python' by Perry Xiao (Wiley)

# Note: Please first read through Ref #1 above. Refs #2 and #3 both employ Ref #1's
# concepts and Python code quite heavily.

# Most examples on Q-learning that I have come across online are either very closely based
# on Ref #1, or apply Q-learning in a simulated/gaming environment (for e.g. OpenAI's
# Gym/Gymnasium environments).

# I wanted to see if I could apply Q-learning in a real-life environment. I have chosen
# to use Singapore's fairly extensive rail network (known as the MRT, or Mass Rapid Transit)
# as the environment for this project. I wanted to see if I could build on Ref #1's approach
# to find the best (shortest) route from a start node to an end node, but now factoring in
# the different travelling times between train stations on the MRT - the route that
# traverses the least number of stations may not necessarily be the route that is the
# shortest in terms of travelling duration.

# Here, I treat the travelling time between 2 train stations as a cost incurred. The longer
# the travelling time, the higher the cost. I normalise this cost through dividing it by the
# maximum cost, which I define as the maximum of the travelling times between any 2 stations,
# plus a chosen waiting time (i.e. wait cost) for the period (peak hour or off-peak). I then
# use 1 to deduct this normalised cost, in order to get a reward value. So the lesser the cost,
# the greater the reward. I further multiply this reward value with what I call a
# reward coefficient. After running a grid search-like algorithm (included below for reference)
# to find the best performing values for the alpha and gamma hyperparameters, there is also a
# range of values for this reward coefficient that work well for the chosen alpha and gamma values.
# I have put in a set of default values for initialising the BestRouteQLearner object that will
# work well with the Singapore MRT dataset - these default values may need to be adjusted
# accordingly for optimal usage on other datasets.

# The core MRT data can be found in the 'mrt_data.xlsx' file inside the Data folder of this
# project. It includes 3 worksheets:

# 1. Nodes
# This worksheet stores the Station Codes and Station Names. Each Station Code has to be unique
# (just like in a table in a relational database). NB: The train interchanges must share the same
# Station Name (for e.g. 'Bishan' for station codes NS17 and CC15). As the program evaluates the
# routes and detects a repeated instance of the Station Name, it will factor in a wait cost.
# For e.g. the commuter is on a route that makes a transit at Bishan interchange, going from the
# NS17 Bishan train platform to the CC15 Bishan train platform. The cost value indicated in the
# Routes worksheet (detailed shortly) will indicate the time needed to *walk* from the NS17
# platform to the CC15 platform. If the commuter is to board the train at the CC15 platform as
# part of the route, naturally there is also a waiting time (wait cost) of having to wait for the
# next train at the CC15 platform. This wait cost will be the chosen wait cost for the peak hour
# and off-peak periods, and can be set when the BestRouteQLearner object is initialised.

# 2. Routes
# Here, we indicate the Start Station Codes and the End Station Codes, as well as the travelling
# time (i.e. cost) between the two respective stations. In the case of interchanges, the cost
# to be indicated here will be the walking time between the 2 platforms of that interchange.
# (The Station Names in this workwheet simply use the vlookup function in Excel to reflect the
# Station Names that were entered in the Nodes worksheet.)

# 3. Interchanges
# Here, we list down the Station Codes that belong to each interchange. The program will
# check the route for any transits at the interchanges (as mentioned above) and add a
# wait cost to the route if applicable.

# These 3 worksheets are then saved as CSV files in the same Data folder. To ensure
# data integrity, it would naturally be best to have the data stored in a well-designed
# relational database, but for simplicity here I just use CSV files. (Furthermore, the
# data for a rail system does not change very often.) There will be code run in the 
# main.py file to extract the data from these 3 CSV files. Specifically, for the 'fit'
# method of the BestRouteQLearner object, we require numpy arrays in the following 
# format:

# 1. Nodes
# Node ID, Node Name
# Essentially the Station Code and Station Name

# 2. Routes
# Start Node ID, End Node ID, Cost
# I.e. the start Station Code, the end Station Code, and the travelling/waiting time
# between the 2 stations. NB: the program assumes the time taken to go from
# Station A to Station B will be the same as the time taken to go from Station B to
# Station A.

# 3. Interchanges
# All the Node IDs for each interchange are listed in subsequent columns,
# starting at the 2nd column, as in the Interchanges worksheet.

# Amend the code in main.py accordingly to cater for different input formats, as
# long as it is able to extract the required numpy arrays in the format mentioned above.

# For the front-end, for simplicity of demonstation I have chosen to use Pyscript,
# along with some simple Javascript for functionality and CSS for formatting.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
from pyscript import document, display
import asyncio
from asyncio import sleep

status_box = document.getElementById('status_box')
duration_box = document.getElementById('duration_box')
perf_log_box = document.getElementById('perf_log_box')

prediction_num_col = 0
route_id_col = 1
valid_route_col = 2
node_count_col = 3
total_cost_col = 4

sleep_interval = 0.001

# =======================================================================================
class BestRouteQLearner(object):

    def __init__(self,
                alpha=0.9,
                gamma=0.9,
                epochs=30_000,
                reward_coef=7,
                is_peak_hour=False,
                peak_hour_cost=2,
                off_peak_cost=5,
                show_dev_msg=False):

        self.alpha = alpha
        self.gamma = gamma
        self.epochs = epochs
        self.reward_coef = reward_coef

        if is_peak_hour == True:
            self.wait_cost = peak_hour_cost
        else:
            self.wait_cost = off_peak_cost

        self.show_dev_msg = show_dev_msg
        self.prediction_count = 0

    def get_available_actions(self, state):

        current_state_row = self.R[state,:]
        available_actions = np.where(current_state_row >= 0)[0]
        if len(available_actions) > 0:
            return True, available_actions
        else:
            return False, None

    def select_next_action(self, available_actions):

        next_action = np.random.choice(available_actions, 1)[0]
        return next_action

    def update_Q_table(self, state, action, end_node):

        next_state = action
        next_state_max_value = np.max(self.Q[next_state, :])

        reward = self.R[state, action] + self.calc_reward(state, action, end_node)

        self.Q[state, action] = self.Q[state, action] + \
                            self.alpha * (reward + (self.gamma * next_state_max_value) - \
                            self.Q[state, action])

        if np.max(self.Q) > 0:
            return(np.sum(self.Q/np.max(self.Q)*100))
        else:
            return(0)
    
    def get_nett_cost(self, cost, state, action, end_node):

        if self.is_interchange_transfer(state, action) == True:
            if action != end_node:
                cost += self.wait_cost
        return cost

    def get_cost(self, state, action, end_node):
        
        index = np.where((self.routes == [state, action]).all(axis=1))[0]
        if len(index) > 0:
            cost = self.costs[index][0][0]
            return self.get_nett_cost(cost, state, action, end_node)

        index = np.where((self.routes == [action, state]).all(axis=1))[0]
        if len(index) > 0:
            cost = self.costs[index][0][0]
            return self.get_nett_cost(cost, state, action, end_node)

    def calc_reward(self, state, action, end_node):

        reward = self.reward_coef * (1 - (self.get_cost(state, action, end_node) / self.cost_max))
        return reward
    
    def is_interchange_transfer(self, state, action):

        if (state in self.interchanges) and (action in self.interchanges):
            if self.get_node_name(self.encoder.inverse_transform([state])[0]) == \
                self.get_node_name(self.encoder.inverse_transform([action])[0]):
                return True

        return False

    def get_node_name(self, node_id):

        node_name = self.orig_nodes[np.flatnonzero(self.orig_nodes[:,0]==node_id),1][0]
        return node_name
    
    def get_alt_interchange(self, node_id):

        interchange = [i for i in self.interchanges if node_id in i]
        if len(interchange) > 0:
            interchange = [i for i in interchange[0] if pd.notnull(i)]
            interchange.remove(node_id)
            if len(interchange) > 1:
                interchange = np.random.choice(interchange, 1)[0]
            else:
                interchange = interchange[0]
        else:
            interchange = node_id

        return interchange

    def generate_consolidated_logs(self, total_predictions):
        self.consolidated_route_log = []
        self.consolidated_cost_log = []

        self.perf_log = np.zeros((total_predictions, 5))
        self.perf_log_ctr = 0
    
    def update_consolidated_logs(self, route_id, is_valid_route, best_route, cost_log):

        if is_valid_route == True:
            self.consolidated_route_log.append(best_route)
            self.consolidated_cost_log.append(cost_log)

            self.perf_log[self.perf_log_ctr, prediction_num_col] = self.prediction_count
            self.perf_log[self.perf_log_ctr, route_id_col] = route_id
            self.perf_log[self.perf_log_ctr, valid_route_col] = True
            self.perf_log[self.perf_log_ctr, node_count_col] = len(cost_log)
            self.perf_log[self.perf_log_ctr, total_cost_col] = sum(cost_log)
        else:
            self.consolidated_route_log.append([])
            self.consolidated_cost_log.append([])

            self.perf_log[self.perf_log_ctr, prediction_num_col] = self.prediction_count
            self.perf_log[self.perf_log_ctr, route_id_col] = route_id
            self.perf_log[self.perf_log_ctr, valid_route_col] = False
        
        self.perf_log_ctr += 1

    async def update_prediction_count(self, total_predictions):
        self.prediction_count += 1
        await sleep(sleep_interval)
        display(f'Running prediction {self.prediction_count} of {total_predictions}',
            target="status_box", append=True)

    async def generate_ref_tables(self, end_node):

        self.generate_R_table(end_node)

        await self.generate_Q_table(end_node)

    def generate_R_table(self, end_node):

        self.R = np.ones((self.table_size, self.table_size))
        self.R *= -1
        path_score = 0
        goal_score = 100

        for node in self.routes:
            if node[1] == end_node:
                self.R[node[0], node[1]] = goal_score
                self.R[node[1], node[0]] = path_score
            elif node[0] == end_node:
                self.R[node[0], node[1]] = path_score
                self.R[node[1], node[0]] = goal_score
            else:
                self.R[node[0], node[1]] = path_score
                self.R[node[1], node[0]] = path_score

    async def generate_Q_table(self, end_node):

        self.Q = np.zeros((self.table_size, self.table_size))

        self.score_log = []

        ctr = 0
        init_text = status_box.innerHTML

        for i in range(self.epochs):

            gen_text = '- applying Q-learning algorithm'
            suffix = '.'
            if (i % 1000 == 0):
                await sleep(sleep_interval)
                status_box.innerHTML = init_text + gen_text + (ctr * suffix)
                ctr += 1
                if ctr == 4:
                    ctr = 0

            state = np.random.randint(0, self.table_size)
            has_available_actions, available_actions = self.get_available_actions(state)
            
            if has_available_actions == True:
                action = self.select_next_action(available_actions)
                score = self.update_Q_table(state, action, end_node)
                self.score_log.append(score)
        
        if self.show_dev_msg == True:
            # DISPLAY CONVERGENCE GRAPH
            fig, ax = plt.subplots(figsize=(4.4, 3.2))
            ax.tick_params(axis='both', which='major', labelsize=6)
            plt.plot(self.score_log)
            plt.xlabel('Training Epochs', fontsize=7)
            plt.ylabel('Sum of normalised Q values', fontsize=7)
            plt.title(f'Prediction {self.prediction_count}: Convergence graph of sum of normalised Q values',
                fontsize=8)
            target_element = 'pred' + str(self.prediction_count) + '_graph_box'
            display(fig, target=target_element, append=True)

    async def get_best_route(self, start_node, end_node):

        await self.generate_ref_tables(end_node)

        state = start_node
        best_route = [state]
        cost_log = []
        is_valid_route = True

        if start_node == end_node:
            cost_log.append(0)

        while state != end_node:

            state_row = self.Q[state,:]
            available_index = np.where(state_row > 0)[0]
            available_index = np.array([i for i in available_index if i not in best_route])

            if len(available_index) == 1:
                next_state = available_index[0]
            elif len(available_index) > 0:
                available_state_values = np.array([i for i in state_row[available_index]])
                max_index = \
                    np.flatnonzero(available_state_values == np.max(available_state_values))
                if len(max_index) == 1:
                    next_state = available_index[max_index][0]
                elif len(max_index) > 1:
                    max_index = np.random.choice(max_index, 1)[0]
                    next_state = available_index[max_index]
            else:
                is_valid_route = False
                break

            best_route.append(next_state)

            if state == start_node:
                if self.is_interchange_transfer(state, next_state) == True:
                    cost_log.append(0)
                else:
                    cost_log.append(self.wait_cost)

            cost = self.get_cost(state, next_state, end_node)
            cost_log.append(cost)

            state = next_state
        # End of while loop

        if is_valid_route == True:
            return True, best_route, cost_log
        else:
            return False, None, None

    async def fit(self,
            nodes,
            routes,
            interchanges,
            node_id_alias=None,
            node_name_alias=None,
            cost_alias=None,
            cost_unit=None):
        
        await sleep(sleep_interval)
        display('Fitting the model', target="status_box", append=True)

        self.encoder = LabelEncoder()

        df_nodes = pd.DataFrame(nodes)
        df_nodes = df_nodes.set_axis(['node_id', 'node_name'], axis=1)
        self.df_nodes = df_nodes
        self.orig_nodes = df_nodes.values

        self.nodes = nodes[:, 0]
        self.nodes = self.encoder.fit_transform(self.nodes)
        self.nodes = self.nodes.reshape(-1, 1)

        route_start_nodes = routes[:,0]
        route_start_nodes = self.encoder.transform(route_start_nodes)
        route_start_nodes = route_start_nodes.reshape(-1, 1)
        route_end_nodes = routes[:,1]
        route_end_nodes = self.encoder.transform(route_end_nodes)
        route_end_nodes = route_end_nodes.reshape(-1, 1)
        self.routes = np.hstack((route_start_nodes, route_end_nodes))

        costs = routes[:,2]
        self.costs = costs.reshape(-1, 1)

        self.interchanges = interchanges
        interchange_col_count = self.interchanges.shape[1]
        # Encode the interchanges
        for interchange in self.interchanges:
            for index in range(interchange_col_count):
                if pd.notnull(interchange[index]):
                    interchange[index] = self.encoder.transform([interchange[index]])[0]

        self.node_id_alias = node_id_alias
        self.node_name_alias = node_name_alias
        self.cost_alias = cost_alias
        self.cost_unit = cost_unit

        self.table_size = self.nodes.shape[0]

        self.cost_max = np.max(self.costs) + self.wait_cost

    async def predict(self, start_node, end_node):

        if start_node not in self.orig_nodes[:,0]:
            status_box.innerText = 'The Start Node ' + start_node + \
                ' does not exist in the dataset.'
            return False, None
        
        if end_node not in self.orig_nodes[:,0]:
            status_box.innerText = 'The End Node ' + end_node + \
                ' does not exist in the dataset.'
            return False, None
        
        if start_node == end_node:
            status_box.innerText = 'The Start Node ' + start_node + \
                ' is the same as the End Node ' + end_node + '.'
            return False, None

        start_node = self.encoder.transform([start_node])[0]
        orig_end_node = end_node
        end_node = self.encoder.transform([end_node])[0]
# =================================================================================================
        forward_route_id = 0
        reverse_route_id = 1
        alt_start_intchg_route_id = 2
        alt_end_intchg_route_id = 3

        if start_node not in self.interchanges and end_node not in self.interchanges:

            total_predictions = 2

            self.generate_consolidated_logs(total_predictions)

            # DEFAULT FORWARD ROUTE
            await self.update_prediction_count(total_predictions)
                
            is_valid_route, best_route, cost_log = \
                await self.get_best_route(start_node, end_node)

            self.update_consolidated_logs(forward_route_id, is_valid_route,
                best_route, cost_log)
            
            # REVERSE ROUTE
            await self.update_prediction_count(total_predictions)
                
            is_valid_route, best_route, cost_log = \
                await self.get_best_route(end_node, start_node)

            if is_valid_route == True:
                best_route.reverse()
                cost_log.reverse()
                wait_cost = cost_log[-1]
                cost_log.pop(-1)
                cost_log.insert(0, wait_cost)

            self.update_consolidated_logs(reverse_route_id, is_valid_route,
                best_route, cost_log)

        else:

            if start_node in self.interchanges and end_node in self.interchanges:
                total_predictions = 3
            else:
                total_predictions = 2
            
            self.generate_consolidated_logs(total_predictions)

            # DEFAULT FORWARD ROUTE
            await self.update_prediction_count(total_predictions)

            is_valid_route, best_route, cost_log = \
                await self.get_best_route(start_node, end_node)
            
            self.update_consolidated_logs(forward_route_id, is_valid_route,
                best_route, cost_log)

            # ALTERNATE START INTERCHANGE ROUTE
            if start_node in self.interchanges:
                await self.update_prediction_count(total_predictions)

                alt_interchange = self.get_alt_interchange(start_node)
                if alt_interchange == end_node:
                    is_valid_route, best_route, cost_log = \
                        await self.get_best_route(start_node, end_node)
                else:
                    is_valid_route, best_route, cost_log = \
                        await self.get_best_route(alt_interchange, end_node)
                    if is_valid_route == True:
                        best_route.insert(0, start_node)
                        cost_log.insert(0,0)
                        cost_log[1] = self.get_cost(start_node, alt_interchange, end_node)
                
                self.update_consolidated_logs(alt_start_intchg_route_id, is_valid_route,
                    best_route, cost_log)

            # ALTERNATIVE END INTERCHANGE ROUTE
            if end_node in self.interchanges:
                await self.update_prediction_count(total_predictions)

                alt_interchange = self.get_alt_interchange(end_node)

                is_valid_route, best_route, cost_log = \
                    await self.get_best_route(start_node, alt_interchange)
                if is_valid_route == True:
                    best_route.append(end_node)
                    cost_log.append(self.get_cost(alt_interchange, end_node, end_node))

                self.update_consolidated_logs(alt_end_intchg_route_id, is_valid_route,
                    best_route, cost_log)
# =================================================================================================
        if self.show_dev_msg == True:
            await sleep(sleep_interval)

            display('PERFORMANCE LOG:', target="perf_log_box", append=True)
            df_perf_log = pd.DataFrame(self.perf_log)
            df_perf_log.columns = ['Prediction', 'Route ID', 'Valid Route', 'Steps', 'Duration']
            df_perf_log = df_perf_log.to_html(index=False)
            perf_log_box.innerHTML = 'PERFORMANCE LOG:<br>' + df_perf_log + '<br>'

            ctr = 1
            decoded_route_log = \
                [self.encoder.inverse_transform(i) for i in self.consolidated_route_log]
            display('PREDICTED ROUTES:', target="perf_log_box", append=True)
            for i in decoded_route_log:
                display(f'Prediction {ctr}: {i}', target="perf_log_box", append=True)
                ctr += 1
# =================================================================================================
        route = self.perf_log[np.where(self.perf_log[:,valid_route_col]==True)[0]]
        if len(route) > 0:
            # Get rows with the minimum total cost
            index = np.where(route[:,total_cost_col]==np.min(route[:,total_cost_col]))[0]
            route = route[index,:]

            # Get rows with the minimum number of nodes
            index = np.where(route[:,node_count_col]==np.min(route[:,node_count_col]))[0]
            route = route[index,:]

            # Get the prediction number of the first of the remaining entries
            index = int(route[0][0]) -1
        else:
            return False, None

        best_route, cost_log = self.consolidated_route_log[index], self.consolidated_cost_log[index]

        best_route = [self.encoder.inverse_transform([i])[0] for i in best_route]

        self.total_cost = sum(cost_log)

        best_route_dict = {
                        'node_id': best_route,
                        'cost': cost_log
                        }
        df_best_route = pd.DataFrame(best_route_dict)
        df_best_route = df_best_route.merge(self.df_nodes, how='left', on='node_id')
        df_best_route = df_best_route[['node_id', 'node_name', 'cost']]

        if self.node_id_alias != None:
            node_id_alias = self.node_id_alias
        else:
            node_id_alias = 'ID'
        
        if self.node_name_alias != None:
            node_name_alias = self.node_name_alias
        else:
            node_name_alias = 'Name'
        
        if self.cost_alias != None:
            cost_alias = self.cost_alias
        else:
            cost_alias = 'Cost'

        if self.cost_unit != None:
            cost_header = cost_alias + ' (' + self.cost_unit + ')'
        else:
            cost_header = cost_alias

        df_best_route = df_best_route.set_axis([node_id_alias, node_name_alias,
                        cost_header], axis=1)
        df_best_route.index = np.arange(1, len(df_best_route)+1)

        return True, df_best_route
# =================================================================================================



# GRID SEARCH FOR FINDING SUITABLE HYPERPARAMETERS, INCLUDED HERE FOR COMPLETENESS ================
# This was desgined to be run from the IDE but can be modified for use with the website.

    # def grid_search(self,
    #                 start_end_nodes,
    #                 routes_true,
    #                 output_csv,
    #                 n_iter,
    #                 epoch_count,
    #                 alpha_range,
    #                 gamma_range,
    #                 reward_coef_range):
        
    #     if self.show_dev_msg == True:
    #         self.show_dev_msg = False
    #         to_reset_dev_msg = True
        
    #     orig_alpha = self.alpha
    #     orig_gamma = self.gamma
    #     orig_epochs = self.epochs
    #     orig_reward_coef = self.reward_coef

    #     self.n_iter = n_iter
    #     self.epochs = epoch_count

    #     self.alpha_range = alpha_range
    #     self.gamma_range = gamma_range
    #     self.reward_coef_range = reward_coef_range

    #     tab = '\t'

    #     print('\nRunning Grid Search now, please wait until completion')
    #     print('=====================================================')
    #     print(f'Index{tab}Alpha{tab}Gamma{tab}Reward Coef{tab}Accuracy')

    #     index_log = []
    #     alpha_log = []
    #     gamma_log = []
    #     reward_coef_log = []
    #     accuracy_log = []

    #     index = 1
        
    #     for i in range(n_iter):
    #         for alpha in alpha_range:
    #             for gamma in gamma_range:
    #                 for reward_coef in reward_coef_range:

    #                     self.alpha = alpha
    #                     self.gamma = gamma
    #                     self.reward_coef = reward_coef
                        
    #                     pred_score_log = []

    #                     for sen in start_end_nodes:

    #                         s_node, e_node = sen[0], sen[1]

    #                         is_valid_route, df_best_route = self.predict(start_node=s_node, end_node=e_node)
    #                         if is_valid_route == True:

    #                             if self.node_id_alias != None:
    #                                 node_id_alias = self.node_id_alias
    #                             else:
    #                                 node_id_alias = 'ID'
    #                             best_route_pred = df_best_route[[node_id_alias]].values.ravel()

    #                             idx = np.where((start_end_nodes == [s_node, e_node]).all(axis=1))[0][0]
    #                             best_route_true = routes_true[idx]
    #                             best_route_true = np.array([x for x in best_route_true if pd.notnull(x)])

    #                             result = np.array_equal(best_route_true, best_route_pred)
    #                             if result ==  True:
    #                                 pred_score_log.append(1)
    #                             else:
    #                                 pred_score_log.append(0)
    #                         else:
    #                             pred_score_log.append(0)
                        
    #                     accuracy = sum(pred_score_log) / len(pred_score_log)

    #                     index_log.append(index)
    #                     alpha_log.append(alpha)
    #                     gamma_log.append(gamma)
    #                     reward_coef_log.append(reward_coef)
    #                     accuracy_log.append(accuracy)

    #                     print(f'{index}{tab}{alpha:.1}{tab}{gamma:.1}{tab}{reward_coef}{tab}{tab}{accuracy:.2}')

    #                     index += 1

    #     # Reset the following hyperparameters
    #     self.alpha = orig_alpha
    #     self.gamma = orig_gamma
    #     self.epochs = orig_epochs
    #     self.reward_coef = orig_reward_coef

    #     if to_reset_dev_msg == True:
    #         self.show_dev_msg = True

    #     plt.plot(range(1, index), accuracy_log)
    #     plt.show()

    #     grid_search_dict = {
    #         'Index': index_log,
    #         'Alpha': alpha_log,
    #         'Gamma': gamma_log,
    #         'Reward Coef': reward_coef_log,
    #         'Accuracy': accuracy_log
    #     }
    #     df_grid_search = pd.DataFrame(grid_search_dict)

    #     if output_csv[-4:] == '.csv':
    #         output_csv = output_csv.rstrip('.csv')

    #     dt = datetime.datetime.now()
    #     dt = dt.strftime("%Y") + '_' + dt.strftime("%m") + '_' + \
    #         dt.strftime("%d") + '_' + dt.strftime("%H") + '_' + \
    #         dt.strftime("%M") + '_' + dt.strftime("%S")

    #     output_csv = output_csv + '_' + dt + '.csv'

    #     df_grid_search.to_csv(output_csv, encoding='utf-8',
    #                         index=False, header=True)

    #     print(f'\nGrid Search completed.')
    #     print(f'The results have been saved into {output_csv}')
# ====================================================================================