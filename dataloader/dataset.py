import networkx as nx
import os
import pickle
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm
from global_settings import global_data_root


# Function to preprocess the map
def preprocess_data_OSM(origin_data):
    E = len(origin_data['edge_id'])

    # Record the coordinates of the nodesand calculate the bounds of weights
    #! 0-indexing
    pos = {}
    edges = []
    for i in range(E):
        u, v = origin_data['from_node_id'][i], origin_data['to_node_id'][i]
        w = origin_data['length'][i] / origin_data['speed_limit'][i]
        edges.append((u, v, w, 10*w))
        if u not in pos:
            pos[u] = (origin_data['from_lon'][i], origin_data['from_lat'][i])
        if v not in pos:
            pos[v] = (origin_data['to_lon'][i], origin_data['to_lat'][i])

    return edges, pos


def preprocess_node(node_dir):
    #! 0-indexing
    pos = pd.read_csv(node_dir)
    pos = { int(nid): (float(lon), float(lat), bool(hasc)) for _, (nid, lon, lat, hasc) in pos.iterrows() }
    return pos

def translate_roadtype_to_capacity(roadtype):
    dic = {'living_street': 1, 'motorway': 10, 'motorway_link': 10, 'primary': 8, 'primary_link': 8, 'residential': 2, 'secondary': 6, 'secondary_link': 6, 'service': 3, 'tertiary': 4, 'tertiary_link': 4, 'trunk': 7, 'trunk_link': 7, 'unclassified': 5}
    return dic[roadtype]

def preprocess_edge(edge_dir):
    #! 0-indexing
    edges = pd.read_csv(edge_dir)
    edges = [(int(o), int(d), float(l), translate_roadtype_to_capacity(c)) for _, (o, d, c, geo, l) in edges.iterrows()
    ]
    return edges

def preprocess_traj(traj_dir):
    trajs = pd.read_csv(traj_dir)
    trajs = [
            {"VehicleID": vid,
             "TripID": tid,
            "Points": [[int(p.split("-")[0]), float(p.split("-")[1])] for p in ps.split("_")],
            "DepartureTime": dt,
            "Duration": dr
        } for _, (vid, tid, ps, dt, dr, l) in tqdm(trajs.iterrows(), desc='Loading trajectories')
    ]
    trajs_ = {}
    for traj in tqdm(trajs, desc='Processing trajectories'):
        if traj["VehicleID"] not in trajs_:
            trajs_[traj["VehicleID"]] = []
        trajs_[traj["VehicleID"]].append(traj)
    return trajs_ 
    # [vehicle_0, vehicle_1, ...], vehicle_i = [traj_0, traj_1, ...], traj_i = {"VehicleID": vid, "TripID": tid, "Points": [[id, time] for p in ps.split("_")], "DepartureTime": dt, "Duration": dr}

# Function to read the map of cities
def read_city(city, root=global_data_root):
    root = os.path.join(root, city)
    assert os.path.exists(root), f'Error: data path {root} does not exist!'

    origin_data_path = os.path.join(root, f'{city}_data.csv')
    if os.path.exists(origin_data_path):
        origin_data = pd.read_csv(origin_data_path).to_dict(orient='list')
        edges, pos = preprocess_data_OSM(origin_data) #! 0-indexing

    else:
        node_dir = os.path.join(root, f'node_{city}.csv')
        edge_dir = os.path.join(root, f'edge_{city}.csv')
        pos = preprocess_node(node_dir) #! 0-indexing
        edges = preprocess_edge(edge_dir) #! 0-indexing

    return edges, pos


def sample_capacity(capacity_scale = 10, edge_num = 100):
    edge_capacity = np.random.uniform(1,capacity_scale,edge_num)
    return edge_capacity

# get weighted adjacency table, return 0-indexing
def get_weighted_adj_table(edges, pos, capacity, normalization = True, quantization_scale = None, max_connection = 4):

    adj_table = np.zeros([len(pos),max_connection, 2]) # [node, connection, [target_node, weight]]

    # add edges to adj_table
    for i in range(len(edges)):
        if np.sum(adj_table[edges[i][0],:,0]!=0) >= max_connection: # already full
            raise ValueError('Error: max_connection is too small')
        elif adj_table[edges[i][0],np.sum(adj_table[edges[i][0],:,0]!=0),0] == max_connection: # duplicate edge
            raise ValueError('Error: duplicate edge')
        else:
            adj_table[edges[i][0],np.sum(adj_table[edges[i][0],:,0]!=0)] = [edges[i][1]+1,capacity[i]] # [target_node, weight], add to the first empty slot
            #! the adj_table[1][0][0] is the first connection of road 2,
            #! the adj_table[1][0][0] = 1 means that road 2 is connected to road 1
            #! the ajd_table[1][0][1] is the road length from road 2 to road 1
    
    if normalization:
        adj_table[:,:,1] = adj_table[:,:,1]/np.max(adj_table[:,:,1])
    if quantization_scale:
        adj_table[:,:,1] = np.ceil(adj_table[:,:,1]/np.max(adj_table[:,:,1])*quantization_scale)
        
    return adj_table #! 0-indexing


# transfer node, wrighted_adj to graph
#! 0-indexing
def transfer_graph(adj_table):
    G = nx.DiGraph()
    for i in range(len(adj_table)):
        G.add_node(i)
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                G.add_edge(i,adj_table[i,j,0]-1,weight=adj_table[i,j,1])
    return G


# get shortest traj, input one adj_table, one OD pair, return one trajectory 1-indexing
#! OD is 1-indexing, trajectory is 1-indexing, graph is 0-indexing
def generate_trajectory_list(G, OD, max_length = 50):
    trajectory = nx.shortest_path(G, (OD[0]-1), (OD[1]-1), weight='weight')
    if len(trajectory) > max_length:
        return None
    
    trajectory = [i+1 for i in trajectory]
    
    return trajectory


# generate OD pairs, return OD 1-indexing
#! 1-indexing
def generate_OD(G, node_num):
    OD = np.random.randint(1,node_num+1,2)
    while nx.has_path(G, OD[0]-1, OD[1]-1) == False or OD[0] == OD[1]:
        OD = np.random.randint(1,node_num+1,2)
    return OD


# for each OD, generate random_sample_num trajectories and adj_table
def simple_simulator(city = 'boston', random_sample_num = 5, max_length = 50, capacity_scale = 10, weight_quantization_scale = None, max_connection = 4):
    edges, pos = read_city(city) #! 0-indexing
    node_num = len(pos)
    edge_num = len(edges)

    all_encoded_trajectories = [] #! 1-indexing
    all_adj_list = [] #! 0-indexing
    OD = None #! 1-indexing
    i = 0
    while i < random_sample_num:
        edge_capacity = sample_capacity(capacity_scale,edge_num)
        adj_table = get_weighted_adj_table(edges, pos, edge_capacity, normalization = True, quantization_scale = weight_quantization_scale, max_connection = max_connection)
        G = transfer_graph(adj_table)
        if OD is None:
            OD = generate_OD(G, node_num) #! 1-indexing
        trajectory = generate_trajectory_list(G, OD, max_length=max_length)

        if trajectory == None:
            # current OD is too long, generate a new OD and restart
            OD = generate_OD(G, node_num)
            all_encoded_trajectories = []
            all_adj_list = []
            i = 0
        else:
            all_encoded_trajectories.append(trajectory)
            all_adj_list.append(adj_table)
            i += 1

    # all_encoded_trajectories: [random_sample_num, max_length] #! 1-indexing
    # all_adj_list: [random_sample_num, node_num, max_connection, 2] #! 0-indexing
    return all_encoded_trajectories, all_adj_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_root', type=str, default=None)
    parser.add_argument('--city', type=str, default='boston')
    parser.add_argument('--simulation_from', type=int, default=0)
    parser.add_argument('--simulation_num', type=int, default=2000000)
    parser.add_argument('--random_sample_num', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--capacity_scale', type=int, default=10)
    parser.add_argument('--weight_quantization_scale', type=int, default=None)
    parser.add_argument('--max_connection', type=int, default=4)
    args = parser.parse_args()

    city = args.city
    data_root = args.tmp_root
    simulation_from = args.simulation_from
    simulation_num = args.simulation_num
    random_sample_num = args.random_sample_num
    max_length = args.max_length
    capacity_scale = args.capacity_scale
    weight_quantization_scale = args.weight_quantization_scale
    max_connection = args.max_connection

    #simulation for simulation_num times

    if data_root is None:
        data_root = os.path.join(global_data_root, city, 'simple_simulator','data_one_by_one')
    else:
        os.mkdir(data_root, exist_ok=True)

    edges, pos = read_city(city)
    node_num = len(pos)
    edge_num = len(edges)

    print('Start generating data...')
    for t in tqdm(range(simulation_from, simulation_from+simulation_num), desc=f'Generating data'):
        all_encoded_trajectories, all_adj_list = simple_simulator(city = city, random_sample_num = random_sample_num, max_length = max_length, capacity_scale = capacity_scale, weight_quantization_scale = weight_quantization_scale, max_connection = max_connection)

        single_data_dir = os.path.join(data_root, str(t))
        if not os.path.exists(single_data_dir):
            os.makedirs(single_data_dir)
        with open (os.path.join(single_data_dir,'trajectory_list.pkl'), 'wb') as file:
            pickle.dump(all_encoded_trajectories, file)
        with open (os.path.join(single_data_dir,'adj_table_list.pkl'), 'wb') as file:
            pickle.dump(all_adj_list, file)

    print(f'one by one saved successfully!')