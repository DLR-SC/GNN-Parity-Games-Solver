""" Data set Generation  for Edge classification 

"""

import argparse as ag_parse
import numpy as np 
import os
import pandas as pd
import pg_parser
from pre_process_game_data import ModifyGameData

import torch
from torch_geometric.data import InMemoryDataset, Data

 

class ModifiedGameDataset(InMemoryDataset):

    def __init__(self, root, mod_games, solutions, transform=None, pre_transform=None, pre_filter=None):
        self._mod_games = mod_games # store data frames here 
        self._solutions = solutions # store sol text data here 
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return['mod_data.pt']

    def make_graph(self, mod_game_df, solution):

        nodes, edges, node_attr = ModifyGameData.parse_mod_game_file(mod_game_df)

        regions_0, strategy_0, regions_1, strategy_1 = pg_parser.parse_solution(solution) 
        
        y_nodes = torch.zeros(node_attr.shape[0], dtype=torch.long)
        y_nodes[regions_1] = 1  
        
        y_edges = torch.zeros(edges.shape[0], dtype=torch.long)
        index_0 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_0]
        index_1 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_1]
        y_edges[index_0] = 1
        y_edges[index_1] = 1

        return Data(x=torch.tensor(node_attr, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(), y_nodes=y_nodes, y_edges=y_edges)
        
    def process(self):
        # Read data into huge `Data` list.
        data_list = [self.make_graph(mod_game, solution) for (mod_game, solution) in zip(self._mod_games, self._solutions)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
 
def main(): 
    
    mod_game_root_dir = '/home/muth_sw/PG/GNNPG/modified_dataset'
    sol_root_dir = '/home/muth_sw/PG/GNNPG/predict_data_set_sol' 
    mod_games = [pd.read_csv(os.path.join(os.path.normpath(mod_game_root_dir) , file))  for file in  os.listdir(mod_game_root_dir)]
    solutions = []
    
    for sol_file in os.listdir(sol_root_dir): 
        with open(os.path.join(os.path.normpath(sol_root_dir) , sol_file)) as f: 
            solutions.append(f.readlines())


    root = 'pg_data_20230123'

    data = ModifiedGameDataset(root, mod_games, solutions)



if __name__ == "__main__":
    main()
        
            





