# Libraries 
# Catch warnings 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from typing import Tuple
import data_loader as dataloader

import pickle
import json
import pandas as pd
import numpy as np
import os
import glob
import sys 
from tqdm import tqdm
from joblib import Parallel, delayed


# WD
# ---------------------------initialize --------------------------- #
class DPModel:
    """ Main body of the machine learning model for estimating bankfull width 
        and depth

        Parameters
        ----------
        rand_state : int
            A custom integer for randomness
    """
    def __init__(self, rand_state) -> None:
        # os.chdir(b'/home/arash.rad/river_3d/conus-fhg/')
        pd.options.display.max_columns = 30
        self.rand_state                = rand_state

# --------------------------- Load trained models  --------------------------- #    
    def loadModel(self, out_feature: str, vote_flag: bool = False, meta_flag: bool = False,
                    best_flag: bool = True, file: str = 'bf', model_type: str = 'xgb',
                    ) -> Tuple[any, list]:
        """ Load the trained transformed models

        Parameters
        ----------
        out_feature : str
            Name of the FHG coefficients
        vote_flag : bool
            Whether to use vote model 
            Options are:
            - True
            - False
        meta_flag : bool
            Whether to use meta model
            Options are:
            - True
            - False
        best_flag : bool
            Whether to use best model
            Options are:
            - True
            - False
        file : str
            Prefix of trained models
            Options are:
            - any string
        model_type: str
            The best model choices
            Options are:
            - xgb
            - lgb
        
       ` Example
        --------
        >>> DPModel.loadData(out_feature = 'Y-bf', vote_flag = False, meta_flag = False,
                    best_flag = True, file = 'bf', model_type = 'xgb')
        """
        
        # Load ML models
        if meta_flag:
            model = pickle.load(open('models/'+file+'_'+out_feature+'_final_Meta_Model.pickle.dat', "rb"))
            best_model = pickle.load(open('models/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        elif vote_flag:
            model = pickle.load(open('models/'+file+'_'+out_feature+'_final_Voting_Model.pickle.dat', "rb"))
            best_model = pickle.load(open('models/'+file+'_'+out_feature+'_'+model_type+'_final_Best_Model.pickle.dat', "rb"))
        elif best_flag:
            model = best_model = pickle.load(open(f"models/trained_xgboost_model_update_{out_feature}_final.pickle.dat", "rb"))
        
        # # Extract feature names
        # json_trans_path = 'model_space/trans_feats'+'_'+out_feature+"_"+'.json'
        # json_model_path = 'model_space/model_feats'+'_'+out_feature+'_'+'.json'

        # # Read the JSON file and convert its contents into a Python list
        # with open(json_trans_path, 'r') as json_file:
        #     trans_list = json.load(json_file)
        # with open(json_model_path, 'r') as json_file:
        #     model_list = json.load(json_file)

        # # Function to reconstruct the original list from the serialized format
        # def restore_order(item):
        #     return item['value']

        # # Reconstruct the original list while preserving the order
        # trans_feats = [restore_order(item) for item in trans_list]
        # model_feats = [restore_order(item) for item in model_list]
 
        return model
    
    def process_target(self, dl_obj, CHUNK_SIZE: int, COMID_COL_NAME: str, target_name: str, final_model_features_from_file: list,
                        vote_flag: bool=False, meta_flag: bool=False, best_flag: bool=True, file: str='bf', model_type: str='xgb') -> None:
        
        PREDICTIONS_OUTPUT_DIR = f'data/{target_name}_out/'
        os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
        if target_name == 'TW_bf':
            model_type = 'xgb'
            x_transform = False
            y_transform = True
        elif target_name == 'Y_bf':
            model_type = 'xgb'
            x_transform = False
            y_transform = True
        else:
            model_type = 'xgb'
            x_transform = False
            y_transform = True

        model = self.loadModel(target_name, vote_flag=False, meta_flag=False, 
                                      best_flag=True, file='NWM', model_type=model_type)

        # dl_obj.transformXData(out_feature=target_name, trans_feats=trans_feats,
        #                         t_type='power', x_transform=x_transform)
        # has_missing_y = np.isnan(dl_obj.data).any()
        rows_with_nan = dl_obj.data[dl_obj.data.isnull().any(axis=1)]
        # if has_missing_y:
        print("Part2 Rows with NaN values:")
        print(rows_with_nan)

        # data_in = dl_obj.buildPCA(target_name)

        print(f"Starting prediction in chunks of {CHUNK_SIZE} rows...")
        chunk_num_counter = 0
        chunk_files_to_cleanup = []

        for i in range(0, len(dl_obj.data), CHUNK_SIZE):
            chunk_df = dl_obj.data.iloc[i : i + CHUNK_SIZE]

            if chunk_df.empty:
                continue

            print(f"  Processing chunk {chunk_num_counter + 1} ({len(chunk_df)} rows)...")


            comids = chunk_df[COMID_COL_NAME].copy()

            X_predict = chunk_df[final_model_features_from_file]
            
            log_predictions = model.predict(X_predict)

            predictions_original_scale = np.abs(np.expm1(log_predictions))
            result_chunk_df = pd.DataFrame({
                COMID_COL_NAME: comids,
                'prediction': predictions_original_scale 
            })

            # 3e. Save the results chunk to a Parquet file
            output_filename = os.path.join(PREDICTIONS_OUTPUT_DIR, f"predictions_chunk_{chunk_num_counter}.parquet")
            result_chunk_df.to_parquet(output_filename, index=False, engine='pyarrow')
            chunk_files_to_cleanup.append(output_filename)
            print(f"  Saved {output_filename} ({len(result_chunk_df)} rows)")
            
            chunk_num_counter += 1

        # Read and concatenate all files
        df_concat = pd.concat([pd.read_parquet(file) for file in chunk_files_to_cleanup], ignore_index=True)

        df_concat.to_parquet(f"{PREDICTIONS_OUTPUT_DIR}/{target_name}_predictions.parquet", index=False, engine='pyarrow')
        for file_path in chunk_files_to_cleanup:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"  Error removing file {file_path}: {e}")
        print(f"\nFinished processing. Predictions saved in '{PREDICTIONS_OUTPUT_DIR}'. Total chunks processed: {chunk_num_counter}")

        return
    
    def checkBounds(self, df):
        mask = df['owp_tw_inchan'] > df['owp_tw_bf']
        df.loc[mask, ['owp_tw_bf', 'owp_tw_inchan']] = df.loc[mask, ['owp_tw_inchan', 'owp_tw_bf']].values

        mask = df['owp_y_inchan'] > df['owp_y_bf']
        df.loc[mask, ['owp_y_bf', 'owp_y_inchan']] = df.loc[mask, ['owp_y_inchan', 'owp_y_bf']].values
        return df

# --------------------------- A driver class --------------------------- #           
class RunDeploy:
    @staticmethod
    def main(argv):
        """ The driver class to run ML models
        
        Parameters 
        ----------
        argv: list
            taken from bash script
        """
        nthreads     = int(argv[0])
        SI           = True # all in SI by default
        rand_state   = 105

        # Additional parameters
        vote_flag           = False
        meta_flag           = False
        best_flag           = True
        file                = 'bf'
        model_type          = 'xgb'
        log_y_t             = True
        CHUNK_SIZE          = 250000
        COMID_COL_NAME      = 'FEATUREID'

        # Load targets
        temp        = json.load(open('data/model_feature_names.json'))
        target_list = temp.get('out_features')
        # target_list = ['TW_bf', 'TW_in']
        out_vars    = []
        del temp
        for target_name in tqdm(target_list):
            print(f"\n{'$'*75}")
            print(f" Processing: {target_name}")
            print(f"{'$'*75}")
            # Load data
            dl_obj = dataloader.DataLoader(rand_state, target_name)
            dl_obj.readFiles()
            dl_obj.imputeData()
            dl_obj.engineer_features()
            final_model_features_from_file = dl_obj.clean_predictors(COMID_COL_NAME)

            deploy_obj = DPModel(rand_state)
            
            deploy_obj.process_target(dl_obj, CHUNK_SIZE, COMID_COL_NAME, target_name, final_model_features_from_file, vote_flag, meta_flag, best_flag, file, model_type)

        print("\n ------------- ML inference complete ----------- \n")
        return

if __name__ == "__main__":
    RunDeploy.main(sys.argv[1:])
    # RunDeploy.main([-1])