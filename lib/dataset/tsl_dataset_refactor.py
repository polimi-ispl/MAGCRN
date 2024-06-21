import os.path
import sys
import pandas as pd
from tsl.data import SpatioTemporalDataset, SynchMode, TemporalSplitter, SpatioTemporalDataModule, AtTimeStepSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import TabularDataset
import numpy as np
from lib.utils import encode_time_series_datetime, encode_weekday, ROOT_PATH

# create a dataset of torch Spatiotemporal with dimension [TIME, NODES, FEATURES]
# TSL: TabularDataset are assumed to be 3-dimensional arrays where the dimensions represent time, nodes and features, respectively.

# ——— Data
Mad_data_2019 = pd.read_csv(os.path.join(ROOT_PATH, '/lib/dataset/Madrid_2019.csv'))

#  ——— Df info
timestamp_index = Mad_data_2019['timestamp'].unique()
datetime_index = pd.to_datetime(timestamp_index)
time_len = len(timestamp_index)
num_nodes = 24

# ——— Target/Covariates
target_cols = 'NO2'
reduced_set_cols = ['NO2', 'w_speed', 'w_direction', 'temperature', 'humidity', 'pressure', 'solar_radiation',
                    'traffic_intensity', 'traffic_occupation_time', 'traffic_load', 'traffic_avg_speed']
met_cols = ['w_speed', 'w_direction', 'temperature', 'humidity', 'pressure', 'solar_radiation']
past_cov_cols = [item for item in reduced_set_cols if item not in met_cols and item != target_cols]
# past_covariates_cols = [item for item in reduced_set_cols if item != target_cols] # consider also meteo as past covariates
fut_cov_cols = met_cols

if also_meteo_as_past_cov:
    past_cov_cols = [item for item in reduced_set_cols if item != target_cols] # consider also meteo as past covariates
else:
    past_cov_cols = [item for item in reduced_set_cols if item != target_cols and item not in met_cols]

print(f'Target: {target_cols}\nPast covariates: {past_cov_cols}\nFuture covariates: {fut_cov_cols}\n')

# ——— Create the single-feature placeholder df with the correct shape [TIME, NODES]
data = np.random.randn(time_len, num_nodes)
target_df = pd.DataFrame(data, index=datetime_index, columns=[f'node_{i+1}' for i in range(num_nodes)])

# ——— Populate the nodes with the data from the corresponding values in the original df [TIME, NODES, 1]
for n in range(num_nodes):
    target_df[f'node_{n+1}'] = Mad_data_2019[['NO2']][n::num_nodes].NO2.values

# ——— Create the placeholder array for the covariates
past_cov_arr = np.zeros((time_len, num_nodes, len(past_cov_cols)))
fut_cov_arr = np.zeros((time_len, num_nodes, len(fut_cov_cols)))

# ——— Populate the nodes with the data from the corresponding values in the original df [TIME, NODES, FEATURES]
for n in range(num_nodes):
    for i, col in enumerate(past_cov_cols):
        past_cov_arr[:, n, i] = Mad_data_2019[[col]][n::num_nodes][col].values
    for i, col in enumerate(fut_cov_cols):
        fut_cov_arr[:, n, i] = Mad_data_2019[[col]][n::num_nodes][col].values

# ——— Create a TSL TabularDataset only for the target, and from it, we infer the time index and the nodes information
target_tabular_dataset = TabularDataset(target=target_df, name='Madrid', precision=16)

# ——— Encode calendar-related information as covariates
day_sin_cos = encode_time_series_datetime(target_tabular_dataset.index)
weekdays = encode_weekday(target_tabular_dataset.index)

# ——— Add the covariates to the dataset
# 'global'prefix is needed for the covariates to be recognized as global [TIME, FEATURES], instead of [TIME, NODES, FEATURES]
target_tabular_dataset.add_exogenous(name='global_u', value=np.concatenate([day_sin_cos, weekdays], axis=-1))
target_tabular_dataset.add_covariate(name='past_cov', value=past_cov_arr, pattern='tnf')
target_tabular_dataset.add_covariate(name='fut_cov', value=fut_cov_arr, pattern='tnf')
target_tabular_dataset.set_mask(np.ones(target_tabular_dataset.target.shape, dtype=bool))

# ——— Save the dataset
target_tabular_dataset.save_pickle(os.path.join(ROOT_PATH, '/lib/dataset/Madrid_2019_tsl.pkl'))
