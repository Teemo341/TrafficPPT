import os
import torch
import pickle
import random
from tqdm import tqdm
import numpy as np
import math

from global_settings import data_root
from .dataset import generate_data, read_city, preprocess_traj, get_weighted_adj_table
from torch.utils.data import Dataset, DataLoader


# Function to read the encoded data from a file and save it as a list
def read_encoded_trajectory(filename):
    with open(filename, 'rb') as file:
        all_encoded_trajectories = pickle.load(file)

    # all_encoded_trajectories: NxT, T can be different for each trajectory
    return all_encoded_trajectories #! 1-indexing


# change inequal length trajectory to equal length trajectory, change list to np
def refine_trajectory(trajectories, block_size):
    all_encoded_trajectories = []
    all_special_mask = np.ones((len(trajectories), block_size),dtype=np.int32)
    for i in range(len(trajectories)):
        traj = trajectories[i]
        traj = [int(code) for code in traj]
        if len(traj) > block_size:
            raise ValueError(f'Trajectory length {len(
                traj)} is greater than block size {block_size}')
        elif len(traj) < block_size:
            all_special_mask[i, len(traj)+1:] = 0
            traj += [0] * (block_size - len(traj))
        all_encoded_trajectories.append(traj)
    all_encoded_trajectories = np.array(all_encoded_trajectories, dtype=np.int32)

    # all_encoded_trajectories: NxT, #! 1-indexing
    # all_special_mask: NxT
    return all_encoded_trajectories, all_special_mask


# read adj table, return np
#! index is 0-indexing, content is 1-indexing
def read_adj_table(filename):
    with open(filename, 'rb') as file:
        all_adj_table = pickle.load(file)
    all_adj_table = np.array(all_adj_table,dtype=np.float32)  # NxVx4x2
    return all_adj_table


def read_data_pkl(idx, block_size, root):
    traj_dir = os.path.join(root, f'data_one_by_one/{idx}/trajectory_list.pkl')
    adj_dir = os.path.join(root, f'data_one_by_one/{idx}/adj_table_list.pkl')
    encoded_trajectory = read_encoded_trajectory(traj_dir)
    encoded_trajectory, special_mask = refine_trajectory(encoded_trajectory, block_size)
    adj_table = read_adj_table(adj_dir)

    # encoded_trajectory: [N x T] #! 1-indexing
    # special_mask: [N x T]
    # adj_table: [N x V x 4 x 2] #! index is 0-indexing, content is 1-indexing
    return encoded_trajectory, special_mask, adj_table


# datasets
class traj_dataset(Dataset):
    def __init__(self, city='boston', data_dir=data_root, simulation_num=1000000, history_num = 5, block_size=60, start_id = 0):
        super(traj_dataset, self).__init__()
        self.data_dir = data_dir
        self.city = city
        self.simulation_num = simulation_num
        self.history_num = history_num
        self.start_id = start_id
        self.block_size = block_size
        if city in ['jinan']:
            self.sample_jinan_data(city)

    def __len__(self):
        return self.simulation_num
    
    def sample_jinan_data(self, city, path='./data_city', max_connection=9):
        edges, pos = read_city(city, path)
        weight = [edge[2] for edge in edges]
        self.adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=max_connection)
        self.adj_table = torch.tensor(self.adj_table, dtype=torch.float32)
        traj_dir = f'./data_city/{city}/traj_{city}.csv'
        traj_dic = preprocess_traj(traj_dir)
        self.traj = []
        self.time_step = []
        for tid in tqdm(range(self.start_id, self.simulation_num,1), desc=f'Transfering {city} points into trajectories'):
            traj, time_step = self.transfer_points_to_traj(traj_dic[tid])
             # traj: [N x T], time_step: [N]
            self.traj.append(traj)
            self.time_step.append(time_step)
        # self.traj: [[signle traj]] BxNxT
        # self.time_step: [[single time step]] BxN
    
    def transfer_points_to_traj(self, traj_points):
        # traj_points: [{"TripID": tid,"Points": [[id, time] for p in ps.split("_")],"DepartureTime": dt,"Duration": dr}]
        traj = []
        time_step = []
        random.shuffle(traj_points)
        for i in range(len(traj_points)):
            if i >= self.history_num:
                break

            # choice time step
            if traj_points[i]["Duration"] <= 60:
                time_step.append(1)
            elif traj_points[i]["Duration"] <= 3600:
                time_step.append(60)
            else:
                time_step.append(3600)

            # repeat times
            repeat_times = []
            for j in range(len(traj_points[i]["Points"])-1): #! 0-indexing
                repeat_times.append(math.ceil((traj_points[i]["Points"][j+1][1]-traj_points[i]["Points"][j][1])/(time_step[i])))
            while np.sum(repeat_times) >= self.block_size:
                repeat_times[ np.argmax(repeat_times) ] -= 1
            traj_ = []
            for j in range(len(traj_points[i]["Points"])-1):
                traj_ += [traj_points[i]["Points"][j][0]+1]*repeat_times[j] #! 1-indexing
            traj_ += [traj_points[i]["Points"][-1][0]+1] #! 1-indexing
            traj.append(torch.tensor(traj_,dtype=torch.int32))
        traj_num = len(traj)
        if traj_num < self.history_num:
            for i in range(self.history_num-traj_num):
                id = random.randint(0,traj_num-1)
                traj.append(traj[id])
                time_step.append(time_step[id])
        return traj, time_step
        # traj: [N x T], time_step: [N]

    def __getitem__(self, idx):
        # return
        # trajectory: [N x T]
        # time_step: [N]
        # special_mask: [N x T]
        # adj_table: [N x V x 2]

        if self.city in ['boston']:
            trajectory, special_mask, adj_table = read_data_pkl(
                idx, self.block_size, root=self.data_dir)
            return torch.tensor(trajectory[:self.history_num]), torch.ones(self.history_num), torch.tensor(special_mask[:self.history_num]), torch.tensor(adj_table[:self.history_num])
        elif self.city in ['jinan']:
            traj = []
            time_step = self.time_step[idx]
            special_mask = []
            for i in range(len(self.traj[idx])):
                special_mask.append(torch.cat([torch.ones(self.traj[idx][i].shape[0], dtype=torch.int32), torch.zeros(self.block_size-self.traj[idx][i].shape[0], dtype=torch.int32)]))
                traj.append(torch.cat([self.traj[idx][i], torch.zeros(self.block_size-self.traj[idx][i].shape[0], dtype=torch.int32)]))
            adj_table = [self.adj_table]*self.history_num
            return torch.stack(traj), torch.tensor(time_step), torch.stack(special_mask), torch.stack(adj_table)


# dataloader with randomize condition function
# boston has 500000 train cars, 2000 test cars
# jinan has 963125 total cars
class traj_dataloader():
    def __init__(self, city='boston', data_dir='./data', test_data_dir = None,
                 simulation_num=800000, test_simulation_num = 163125, condition_num=5, block_size=50, capacity_scale=10, weight_quantization_scale=None, max_connection=4,
                 batch_size=256, shuffle=True, num_workers=8):
        super(traj_dataloader, self).__init__()
        self.city = city
        self.data_dir = data_dir
        self.simulation_num = simulation_num
        self.test_simulation_num = test_simulation_num
        self.condition_num = condition_num
        self.block_size = block_size
        self.capacity_scale = capacity_scale
        self.weight_quantization_scale = weight_quantization_scale
        self.max_connection = max_connection
        self.batch_size = batch_size
        self.shuffle = shuffle

        pos, edges = read_city(city, path='./data_city')
        self.vocab_size = len(pos)+1

        self.train_loader = DataLoader(traj_dataset(city=city, data_dir=data_dir, simulation_num=simulation_num, start_id=0, block_size=block_size),
                                     batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if test_data_dir is not None:
            self.test_loader = DataLoader(traj_dataset(city=city, data_dir=test_data_dir, simulation_num=test_simulation_num, start_id = simulation_num, block_size=block_size),
                                     batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.test_loader = None
        # return trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]
        #! trajectory: 1-indexing, adj_table: index is 0-indexing, content is 1-indexing


    def randomize_condition(self, observe_prob=0.5):
        self.observe_list = np.random.choice(
            (self.vocab_size), int(self.vocab_size*observe_prob), replace=False)+1

    def filter_condition(self, traj_batch):
        unobserved = torch.ones(traj_batch.shape, dtype=torch.int32, device = traj_batch.device)
        for i in range(len(self.observe_list)):
            unobserved *= (traj_batch != self.observe_list[i])
        observed = 1 - unobserved
        traj_batch = traj_batch * observed
        return traj_batch
    
    def filter_random(self,traj_batch, observe_prob = 0.5):
        observed = torch.ones(traj_batch.shape,dtype=torch.float32, device = traj_batch.device)*observe_prob
        observed = torch.bernoulli(observed)
        traj_batch = traj_batch * observed
        return traj_batch

    def generate_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        trajectory_list = []
        special_mask_list = []
        adj_table_list = []

        for i in range(batch_size):
            trajectory, adj_table = generate_data(self.city, self.condition_num, self.block_size, self.capacity_scale,
                                                  self.weight_quantization_scale, self.max_connection)
            trajectory, special_mask = refine_trajectory(trajectory, self.block_size)
            trajectory_list.append(trajectory)
            special_mask_list.append(special_mask)
            adj_table_list.append(adj_table)

        trajectory_list = torch.tensor(np.array(trajectory_list))
        special_mask_list = torch.tensor(np.array(special_mask_list))
        adj_table_list = torch.tensor(np.array(adj_table_list))
        return trajectory_list, special_mask_list, adj_table_list


if __name__ == '__main__':
    # test the dataloader
    loader = traj_dataloader(city='jinan', data_dir='./data',
                             simulation_num=800000, test_simulation_num=163125, condition_num=5, block_size=60, capacity_scale=10, weight_quantization_scale=None, max_connection=9,
                             batch_size=256, shuffle=True, num_workers=8)
    loader.randomize_condition()
    print(1)
    print(loader.observe_list)
    for i, (trajectory, time_step, special_mask, adj_table) in enumerate(loader.train_loader):
        print(i, trajectory.shape, time_step.shape, special_mask.shape, adj_table.shape)


