# Libraries
import pandas as pd
import numpy as np
import pickle
import os
import json
import re
import fnmatch
from pyproj import Transformer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer


# Custom dataset
# --------------------------- Read data files --------------------------- #
class DataLoader:
    """ Main body of the data loader for preparing data for ML models

    Parameters
    ----------
    data_path : str
        The path to data that is used in ML model
    rand_state : int
        A random state number
    Example
    --------
    >>> DataLoader(rand_state = 105, data_path = 'data/input.parquet')
        
    """
    # ml_inputs  nwm_conus_input
    def __init__(self, rand_state: int, out_feature: str, data_path: str = 'data/ml_inputs.parquet') -> None:
        pd.options.display.max_columns  = 60
        self.out_feature                = out_feature
        self.data_path                  = data_path
        self.data                       = pd.DataFrame([])
        self.rand_state                 = rand_state
        np.random.seed(self.rand_state)
        # ___________________________________________________
        # Check directories
        if not os.path.isdir(os.path.join(os.getcwd(),"models/")):
            os.mkdir(os.path.join(os.getcwd(),"models/"))
        if not os.path.isdir(os.path.join(os.getcwd(),"data/")):
            os.mkdir(os.path.join(os.getcwd(),"data/"))
        if not os.path.isdir(os.path.join(os.getcwd(),'model_space/')):
            os.mkdir(os.path.join(os.getcwd(),'model_space/'))

    def readFiles(self) -> None:
        """ Read files from the directories
        """
        try:
            self.data = pd.read_parquet(self.data_path, engine='pyarrow')
            self.data.reset_index(drop=True, inplace=True)
        except:
            print('Wrong address or data format. Please use correct parquet file.')
        
        return

    # --------------------------- Add Features --------------------------- #
    def engineer_features(self, center_x=-22894.21, center_y=-281.82):
        """
        Applies all feature engineering steps to the input DataFrame.
        This includes creating interaction terms, log transforms, and one-hot encoding.
        """
        epsilon = 1e-6 

        self.data['StreamPower_Index'] = self.data['nwm_max'] * self.data['slope']
        self.data['unit_discharge'] = self.data['bf_ff'] / (self.data['TotDASqKM'] + epsilon)
        self.data['vegetation_stability_index'] = self.data['LAI'] / (self.data['slope'] + epsilon)
        self.data['rock_resistance_index'] = self.data['RckDepWs'] * self.data['slope']
        self.data['log_discharge'] = np.log1p(self.data['bf_ff'])
        self.data['log_drainage_area'] = np.log1p(self.data['TotDASqKM'])
        self.data['slope_squared'] = self.data['slope']**2
        self.data['Runoff_Stability_Index'] = self.data['qsb_tavg'] / (self.data['bf_ff'] + epsilon)
        self.data['Runoff_Stability_Index'] = self.data['Runoff_Stability_Index'].clip(0, 1)
        self.data['BankStability_Index'] = self.data['LAI'] / (self.data['StreamPower_Index'] + epsilon)
        self.data['Erodibility_Index'] = self.data['StreamPower_Index'] * self.data['RckDepWs']
        self.data['Sinuosity_Slope_Ratio'] = self.data['Sinuosity'] / (self.data['slope'] + epsilon)

        # --- One-hot encode categorical features ---
        print("One-hot encoding 'flowline_type'...")
        self.data = pd.get_dummies(self.data, columns=['flowline_type'], prefix='flowline')

        if 'lat' in self.data.columns and 'long' in self.data.columns:
            print("  -> Engineering spatial features from lat/lon...")
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
            easting, northing = transformer.transform(self.data['long'].values, self.data['lat'].values)
            self.data['projected_x'] = easting
            self.data['projected_y'] = northing
            
            if center_x is not None and center_y is not None:
                self.data['dist_from_center'] = np.sqrt((self.data['projected_x'] - center_x)**2 + (self.data['projected_y'] - center_y)**2)
                self.data['angle_from_center'] = np.arctan2(self.data['projected_y'] - center_y, self.data['projected_x'] - center_x)
        else:
            print("  -> WARNING: 'latitude' or 'longitude' columns not found. Skipping spatial feature engineering.")
        
        return self.data

    # --------------------------- Imputation --------------------------- #
    def imputeData(self) -> None:
        # Data imputation 
        impute = "median"
        if impute == "zero":
            self.data = self.data.fillna(-1) # a temporary brute force way to deal with NAN
        if impute == "median":
            IMPUTATION_VALUES_PATH = f"model_space/metrics/median_imput_{self.out_feature}.parquet" 
            try:
                median_values_df = pd.read_parquet(IMPUTATION_VALUES_PATH)
                median_values_to_impute = median_values_df['Median'].to_dict()
                print(f"Successfully loaded median values for {len(median_values_to_impute)} features.")
            except FileNotFoundError:
                print(f"CRITICAL ERROR: Imputation file not found at '{IMPUTATION_VALUES_PATH}'. Exiting.")
                exit()
            print(f"Imputing missing values using saved medians...")
            self.data.fillna(value=median_values_to_impute, inplace=True)

        return
    
    #---------------------------  Feature Selection ---------------------------- #
    def clean_predictors(self, COMID_COL_NAME) -> list:
        FEATURE_ORDER_PATH = f"model_space/final_model_features_{self.out_feature}.json"
        with open(FEATURE_ORDER_PATH, 'r') as f:
            final_model_features_from_file = json.load(f)
        print(f"Loaded model expecting {len(final_model_features_from_file)} features in a specific order.")

        required_cols = final_model_features_from_file + [COMID_COL_NAME]
        for col in required_cols:
            if col not in self.data.columns:
                print(f"  WARNING: Column '{col}' not found in input data. Adding it and filling with 0.")
                self.data[col] = 0
        print("Feature engineering and column alignment complete.")

        return final_model_features_from_file

    # --------------------------- Dimention Reduction --------------------------- #     
    # PCA model
    def buildPCA(self, variable)  -> None:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        matching_files = []
        folder_path = 'models'
        full_path = os.path.join(os.getcwd(), folder_path)
        # Iterate through the files in the folder
        for root, dirs, files in os.walk(full_path):
            for filename in files:
                # Check if both "PCA" and "Y_bf" are present in the file name
                search_pattern = f'*PCA*{variable}*'
                if all(fnmatch.fnmatch(filename, f'*{part}*') for part in search_pattern.split('*')):
                    matching_files.append(os.path.join(root, filename))

        # Extract the text between "PCA" and the 'vars' value using regular expressions
        pattern = f'{re.escape(variable)}(.*?)PCA'
        captured_texts = []
        for filename in matching_files:
            match = re.search(pattern, filename)
            if match:
                captured_texts.append(match.group(1))
            else:
                captured_texts.append("No match found")

        temp = json.load(open('model_space/dimension_space.json'))
        
        # Print the list of matching files
        for pca_item, text in zip(matching_files, captured_texts):
            pca =  pickle.load(open(pca_item, "rb"))
            temp_data = self.data[temp.get(text[1:-1])]
            new_data_pca = pca.transform(temp_data)
            max_n = min(5, new_data_pca.shape[1])
            # max_n = min(5, len(temp.get(text[1:-1])))
            for i in range(0, max_n, 1):
                self.data[str(text[1:-1])+"_"+str(i)] = new_data_pca[:, i]

        return self.data
    
    # --------------------------- Data transformation --------------------------- #     
    # Input and output transformation
    def transformXData(self, out_feature: str, trans_feats : list,
                        t_type: str = 'power', x_transform: bool = False)  -> pd.DataFrame:
        """ Apply scaling and normalization to data
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')
        self.data.reset_index(drop=True)
        trans_data = self.data[trans_feats]

        # Always perform transformation on PCA
        in_features = self.data.columns.tolist()
        in_feats = set(in_features) - set(trans_feats)

        if t_type!='log':
            trans =  pickle.load(open('models/train_x_'+out_feature+'_tansformation.pkl', "rb"))
            min_max_scaler = pickle.load(open('models/train_x_'+out_feature+'_scaler_tansformation.pkl', "rb"))
            scaler_data = min_max_scaler.transform(trans_data)
            data_transformed = trans.transform(scaler_data)
            data_transformed = pd.DataFrame(data_transformed, columns=trans_data.columns)
            if not x_transform:
                self.data = pd.concat([data_transformed, self.data[in_feats]], axis=1)
            else:
                self.data = data_transformed.copy()
        else:
            # Replace NA and inf
            self.data = np.log(np.abs(self.data)).fillna(0)
            self.data.replace([np.inf, -np.inf], -100, inplace=True)

        # Tests
        is_inf_data = self.data.isin([np.inf, -np.inf]).any().any()
        if is_inf_data:
            print('---- found inf in X {0} !!!!'.format(out_feature))
        has_missing_data  = self.data.isna().any().any()
        if has_missing_data:
            print('---- found nan in X {0} !!!!'.format(out_feature))

        return 
    
    def transformYData(self, out_feature, data, t_type: str = 'power', y_transform: bool = False)  -> np.array:
        """ Builds a PCA and extracts new dimensions
        
        Parameters:
        ----------
        variable: str
            A string of target variable to be transformed

        Returns:
        ----------

        """
        print('transforming and plotting ...')

        if y_transform:
            if t_type!='log':
                def applyScalerY(arr):
                    min_max_scaler = pickle.load(open('models/train_y_'+out_feature+'_scaler_tansformation.pkl', "rb"))
                    # min_max_scaler = pickle.load(open('/mnt/d/Lynker/R2_out/New/'+folder+'/conus-fhg/'+file+'/model/'+'train_'+arr.name+'_scaled.pkl', "rb"))
                    data_minmax = min_max_scaler.transform(arr.values.reshape(-1, 1))
                    return data_minmax.flatten()
                
                trans =  pickle.load(open('models/train_y_'+out_feature+'_tansformation.pkl', "rb"))
                #scaler_data = data.apply(applyScalerY)
                data = pd.DataFrame(data, columns=[out_feature])
                data = trans.inverse_transform(data) # .reshape(-1,1)

            else:
                # Replace NA and inf
                data = np.log(np.abs(data)).fillna(0)
                data.replace([np.inf, -np.inf], -100, inplace=True)

        print('--------------- End of Y transformation ---------------')
        
        # Tests
        is_inf_y = np.isinf(data).any()
        if is_inf_y:
            print('---- found inf in Y {0} !!!!'.format(out_feature))
        
        has_missing_y = np.isnan(data).any()
        if has_missing_y:
            print('---- found nan in Y {0} !!!!'.format(out_feature))

        return data
        