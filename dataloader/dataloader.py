import os
import torch
import pickle
import random
from tqdm import tqdm
import numpy as np
import math

from global_settings import global_data_root
from .dataset import read_city, preprocess_traj, get_weighted_adj_table
from torch.utils.data import Dataset, DataLoader

# datasets
class one_by_one_dataset(Dataset):
    def __init__(self, data_root, data_num=1000000, history_num = 5, block_size=60, weight_quantization_scale=None):
        super(one_by_one_dataset, self).__init__()
        one_by_one_root = os.path.join(data_root, 'data_one_by_one')
        assert os.path.exists(one_by_one_root), f'Data directory {one_by_one_root} does not exist.'
        file_num = len(os.listdir(one_by_one_root))
        assert file_num >= data_num, f'Data directory {one_by_one_root} has only {file_num} files, less than data_num {data_num}.'

        self.data_root = data_root
        self.data_num = data_num
        self.history_num = history_num
        self.block_size = block_size
        self.weight_quantization_scale = weight_quantization_scale

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        '''
        # return
            trajectory: [N x T]
            time_step: [N]
            special_mask: [N x T]
            adj_table: [N x V x 2]
        '''
        trajectory, special_mask, adj_table = self.read_data(
            idx, self.block_size, root=self.data_root)
        return torch.tensor(trajectory[:self.history_num]), torch.ones(self.history_num), torch.tensor(special_mask[:self.history_num]), torch.tensor(adj_table[:self.history_num])
    
    def read_data(self, idx, block_size, root):
        traj_dir = os.path.join(root, f'data_one_by_one/{idx}/trajectory_list.pkl')
        adj_dir = os.path.join(root, f'data_one_by_one/{idx}/adj_table_list.pkl')
        encoded_trajectory = self.read_encoded_trajectory(traj_dir)
        encoded_trajectory, special_mask = self.refine_trajectory(encoded_trajectory, block_size)
        adj_table = self.read_adj_table(adj_dir)
        if self.weight_quantization_scale is not None:
            adj_table[:,:,1] = np.ceil(adj_table[:,:,1]/np.max(adj_table[:,:,1])*self.weight_quantization_scale)
        # encoded_trajectory: [N x T] #! 1-indexing
        # special_mask: [N x T]
        # adj_table: [N x V x 4 x 2] #! index is 0-indexing, content is 1-indexing
        return encoded_trajectory, special_mask, adj_table
    
    def read_encoded_trajectory(self, filename):
        '''Function to read the encoded data from a file and save it as a list'''
        with open(filename, 'rb') as file:
            all_encoded_trajectories = pickle.load(file)
        # all_encoded_trajectories: NxT, T can be different for each trajectory
        return all_encoded_trajectories #! 1-indexing
    
    def refine_trajectory(self, trajectories, block_size):
        '''change inequal length trajectory to equal length trajectory, change list to np'''
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
    
    def read_adj_table(self, filename):
        # read adj table, return np
        #! index is 0-indexing, content is 1-indexing
        with open(filename, 'rb') as file:
            all_adj_table = pickle.load(file)
        all_adj_table = np.array(all_adj_table,dtype=np.float32)  # NxVx4x2
        return all_adj_table
    
class merged_dataset(Dataset):
    def __init__(self, city, data_num=1000000, history_num = 5, block_size=60, store=False, max_connection=9, weight_quantization_scale=None):
        super(merged_dataset, self).__init__()
        self.data_num = data_num
        self.history_num = history_num
        self.block_size = block_size
        self.store = store
        self.max_connection = max_connection
        self.weight_quantization_scale = weight_quantization_scale
        self.read_data(city, max_connection)

    def __len__(self):
        return self.data_num
    
    def read_data(self, city, max_connection=9):
        edges, pos = read_city(city)
        weight = [edge[2] for edge in edges]
        self.adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=max_connection, quantization_scale=self.weight_quantization_scale)
        self.adj_table = torch.tensor(self.adj_table, dtype=torch.float32)
        traj_dir = os.path.join(global_data_root, city, 'traj_'+city+'.csv')
        self.traj_dic = preprocess_traj(traj_dir) # original data
        assert len(self.traj_dic) >= self.data_num, f'Trajectory data number {len(self.traj_dic)} is less than data_num {self.data_num}.'
        if self.store:
            self.traj = []
            self.time_step = []
            for tid in tqdm(range(self.data_num), desc=f'Transfering {city} points into trajectories'):
                traj, time_step = self.transfer_points_to_traj(self.traj_dic[tid])
                self.traj.append(traj)
                self.time_step.append(time_step)
    
    def transfer_points_to_traj(self, traj_points):
        # traj_points: [{"TripID": tid,"Points": [[id, time] for p in ps.split("_")],"DepartureTime": dt,"Duration": dr}]
        traj = []
        time_step = []
        idx_list = list(range(len(traj_points)))
        random.shuffle(idx_list)
        for i, idx in enumerate(idx_list):
            if i >= self.history_num:
                break

            # choice time step
            if traj_points[idx]["Duration"] <= 60:
                time_step.append(1)
            elif traj_points[idx]["Duration"] <= 3600:
                time_step.append(60)
            else:
                time_step.append(3600)

            # repeat time to simulate duration
            repeat_times = []
            for j in range(len(traj_points[idx]["Points"])-1): #! 0-indexing
                repeat_times.append(math.ceil((traj_points[idx]["Points"][j+1][1]-traj_points[idx]["Points"][j][1])/(time_step[i])))
            while np.sum(repeat_times) >= self.block_size:
                repeat_times[ np.argmax(repeat_times) ] -= 1
            traj_ = []
            for j in range(len(traj_points[idx]["Points"])-1):
                traj_ += [traj_points[idx]["Points"][j][0]+1]*repeat_times[j] #! 1-indexing
            traj_ += [traj_points[idx]["Points"][-1][0]+1] #! 1-indexing
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
        '''
        # return
            trajectory: [N x T]
            time_step: [N]
            special_mask: [N x T]
            adj_table: [N x V x 2]
        '''
        traj = []
        special_mask = []
        if self.store:
            traj_ = self.traj[idx]
            time_step = self.time_step[idx]
        else:
            traj_ = self.traj_dic[idx]
            traj_, time_step = self.transfer_points_to_traj(traj_)
        for i in range(len(traj_)):
            special_mask.append(torch.cat([torch.ones(traj_[i].shape[0], dtype=torch.int32), torch.zeros(self.block_size-traj_[i].shape[0], dtype=torch.int32)]))
            traj.append(torch.cat([traj_[i], torch.zeros(self.block_size-traj_[i].shape[0], dtype=torch.int32)]))
        adj_table = [self.adj_table]*self.history_num
        return torch.stack(traj), torch.tensor(time_step), torch.stack(special_mask), torch.stack(adj_table)


# dataloader with randomize condition function
# boston has 500000 train cars, 2000 test cars
# jinan has 963125 total cars
class traj_dataloader():
    def __init__(self, city='boston', data_root=global_data_root, data_type='simple_simulator', data_num=800000, test_data_num = 163125, history_num=5, block_size=50, weight_quantization_scale=None, max_connection=4, batch_size=256, num_workers=8, seed=3407, store=True):
        super(traj_dataloader, self).__init__()
        self.max_connection = max_connection
        self.batch_size = batch_size

        edges, pos = read_city(city)
        self.vocab_size = len(pos)+1

        if data_type == 'simple_simulator':
            data_root = os.path.join(data_root, city, 'simple_simulator')
            data_set = one_by_one_dataset(data_root=data_root, data_num=data_num+test_data_num, history_num=history_num, block_size=block_size, weight_quantization_scale=weight_quantization_scale)
        elif data_type == 'CBEngine':
            datas_root = os.path.join(data_root, city, 'CBEngine')
            data_set = one_by_one_dataset(data_root=datas_root, data_num=data_num+test_data_num, history_num=history_num, block_size=block_size, weight_quantization_scale=weight_quantization_scale)
        elif data_type == 'real':
            data_set = merged_dataset(city=city, data_num=data_num+test_data_num, history_num=history_num, block_size=block_size, max_connection=max_connection, store=store, weight_quantization_scale=weight_quantization_scale)
        else:
            raise ValueError(f'Unknown data_type {data_type}.')
        
        assert data_num + test_data_num <= len(data_set), f'Data number {len(data_set)} is less than train + test data num {data_num + test_data_num}.'
        train_set, test_set = torch.utils.data.random_split(data_set, [data_num, test_data_num], generator=torch.Generator().manual_seed(seed))
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # return trajectory: [B x N x T], time_step: [B x N], special_mask: [B x N x T], adj_table: [B x N x V x 2]
        #! trajectory: 1-indexing, adj_table: index is 0-indexing, content is 1-indexing


    def randomize_condition(self, observe_prob=0.5):
        self.observe_list = np.random.choice(
            (self.vocab_size), int(self.vocab_size*observe_prob), replace=False)+1
        self.observe_list = torch.as_tensor(self.observe_list, dtype=torch.int32)
    
    def filter_condition(self, traj_batch):
        self.observe_list = self.observe_list.to(traj_batch.device)
        observed = torch.isin(traj_batch, self.observe_list)
        return traj_batch * observed.to(traj_batch.dtype)
    
    def filter_random(self,traj_batch, observe_prob = 0.5):
        observed = torch.ones(traj_batch.shape,dtype=torch.float32, device = traj_batch.device)*observe_prob
        observed = torch.bernoulli(observed)
        traj_batch = traj_batch * observed
        return traj_batch


if __name__ == '__main__':
    # dataset = one_by_one_dataset(data_root='./data/boston/simple_simulator', data_num=10, history_num=5, block_size=50)
    # print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape, dataset[0][3].shape)

    # dataset = merged_dataset(city='jinan', data_num=100, history_num=5, block_size=50, store=False, max_connection=9)
    # print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape, dataset[0][3].shape)

    loader = traj_dataloader(city='jinan', data_type = 'real',
                             data_num=400, test_data_num=100, history_num=5, block_size=60, weight_quantization_scale=10, max_connection=9,
                             batch_size=32)
    loader.randomize_condition()
    print(1)
    print(loader.observe_list)
    print(len(loader.observe_list))
    for i, (trajectory, time_step, special_mask, adj_table) in enumerate(loader.train_loader):
        print(i, trajectory.shape, time_step.shape, special_mask.shape, adj_table.shape)
        filtered_trajectory = loader.filter_condition(trajectory)
        if i >= 4:
            break

    pass