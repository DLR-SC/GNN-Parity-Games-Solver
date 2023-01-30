""" Data set Modifier  for Edge classification 

    This script modifies the game_xxxx.txt files and generates a new CSV file containing the 
    winning regions calculated from node classification 
    
    Functions: 

    (1) read_dataset: 
        
            Reads the evaluations results (evaluations results file should contain the path of the original game file and it's corresponding winning regions)
        
    (2) modify_game_file: 
    
            Creates a csv file for each modified game file with the columns:  'Node_ID',  'priority', 'owner', 'successors', 'name', 'winning node']
            the 'winning node' is a flag indicating if the corresponding node belongs to the winning region of player 0

    (3) parse_mod_game_file:

            Adds the node attributes for the modified game file. The new node attributes are: "Normalized color" | " One hot encoding to indicate Node Owner " | " One hot encoding to indicate winning Node of 0 "   
    
    (4) pre_process_train_dataset: 

            Creates modified game files from training data
    
    (5) pre_process_predict_results: 

            Creates modified game files from test data used for prediction in pahse 1 implementation

"""

import argparse as ag_parse
import numpy as np 
import os
import pandas as pd
import pg_parser

class ModifyGameData:

    def read_dataset( input_file_path = None):
        
        if input_file_path == None:
        
            input_file_path = '/home/muth_sw/PG/GNNPG/results.csv'
        try: 
            return (pd.read_csv(input_file_path, header= None).to_numpy())
        
                
        except Exception as e:
            
            print("Error occured while reading prediction results: " +  e)
            return None
            
        
    def modify_game_file(results, output_file_location = None):
        
        if output_file_location == None:
        
            output_file_location = '/home/muth_sw/PG/GNNPG/modified_dataset'

        for result in results: 
            
            try: 

                game_data = pd.read_csv(result[0].strip(), sep=" ", header=None, skiprows=1)
                game_data.columns = ["Node_ID",  "priority", "owner", "successors", "name"]
                game_data["winning node"] = 0
                new_file_name = ("mod_" + os.path.basename(os.path.normpath(result[0].strip()))).replace('.txt', '.csv')
                new_file_location = os.path.join(os.path.normpath(output_file_location), new_file_name)
                game_data.set_index('winning node')
                game_data.iloc[np.array(result[1].split(" ")).astype(int), game_data.columns.get_loc('winning node')] = 1
                game_data.to_csv(new_file_location, mode = 'w', index=False)

            except Exception as e: 
                print("Error occurced while processing the file: " + result[0] + e)

    def pre_process_predict_results( predict_results_file): 

        results_df = pd.read_csv(os.path.normpath(predict_results_file), header = None)
        winning_regions = []
        file_paths = []
        mod_results = {}
        for i  in range(0, len(results_df)):
            line = results_df.iloc[i, -1]
            file_path, winning_region = line.split(" ", 1)
            file_paths.append(file_path)
            winning_regions.append(winning_region)

        mod_results['Game files'] = file_paths
        mod_results['Winning regions'] = winning_regions

        mod_results_df = pd.DataFrame(mod_results)
        # print(mod_results_df)
        mod_results_df.to_csv(os.path.join(os.path.dirname((os.path.normpath(predict_results_file))), "mod_results.csv"), index= False, header= False)
        return os.path.join(os.path.dirname((os.path.normpath(predict_results_file))), "mod_results.csv")

    def pre_process_train_dataset( game_files_dir, sol_files_dir, train_data_location):
        
        """ 
        
            Use this function only when there is not enough data after prediction from phase 1 implementation 

        """
        
        game_files = []
        winning_regions = []
        train = {}

        game_file_names = os.listdir(os.path.normpath(game_files_dir))
        game_file_names.sort()
        
        sol_file_names = os.listdir(os.path.normpath(sol_files_dir))
        sol_file_names.sort()

        for game_file in game_file_names: 
            game_files.append(os.path.join(os.path.normpath(game_files_dir), game_file ) )


        for sol_file in sol_file_names: 
            with open(os.path.join(os.path.normpath(sol_files_dir) , sol_file)) as f: 
                regions_0, strategy_0, regions_1, strategy_1 = pg_parser.parse_solution(f.readlines())
                winning_regions.append(" ".join(regions_0.astype(str).tolist()))
        
        train['Game Files'] = game_files
        train['winnign regions'] = winning_regions


        pd.DataFrame(train).to_csv(os.path.join(os.path.normpath(train_data_location), "Phase2_train_data.csv"), index= False, header= False)
        return os.path.join(os.path.normpath(train_data_location), "Phase2_train_data.csv")
        

    def parse_mod_game_file(mod_game_df):
                                
        nodes = mod_game_df.iloc[:,0].to_numpy()
        edges = np.concatenate([(np.array(list(np.broadcast([mod_game_df.iloc[i, 0]], mod_game_df.iloc[i, 3].split(',')))).astype(int)) for i in range(len(mod_game_df))])
        
        node_attr_train_1 = np.append(

            np.expand_dims(mod_game_df.iloc[:,1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float)), axis=1), # Normalizing the range of colours
            [[1, 0] if mod_game_df.iloc[i, 2] == '0' else [0, 1] for i  in range(len(mod_game_df))], 
             axis=1 # Categorical encoding
        )

        node_attr_train_2 = np.append(
            node_attr_train_1, 
            [[1, 0] if mod_game_df.iloc[i, 5] == '0' else [0, 1] for i  in range(len(mod_game_df))], axis = 1)  # One hot encoding for winning regions of 0

        return(nodes, edges, node_attr_train_2)

 
def main(): 
    
    #input_file_path = dataset.pre_process_predict_results('/home/muth_sw/PG/GNNPG/results_1.csv')
    #dataset.modify_game_file(results= (dataset.read_dataset(input_file_path)))
    ModifyGameData.pre_process_train_dataset('/home/muth_sw/PG/GNNPG/games-small/games', '/home/muth_sw/PG/GNNPG/games-small/solutions', '/home/muth_sw/PG/GNNPG/games-small')

if __name__ == "__main__":
    main()
        
            