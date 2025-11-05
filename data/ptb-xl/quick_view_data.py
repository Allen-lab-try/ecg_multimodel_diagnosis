# ---------------------------------------------------------------
# PTB-XL ECG Loader (view the first 500 data sturctures and plot the first record
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
path = "E:/professor_yen/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
sampling_rate = 500
num_records_to_load = 500  # assign how many data you want to view

# ---------------------------------------------------------------
# Load and convert annotation data
# ---------------------------------------------------------------
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# 只取前 N 筆
Y = Y.iloc[:num_records_to_load]

# ---------------------------------------------------------------
# Load raw signal data with progress bar
# ---------------------------------------------------------------
def load_raw_data(df, sampling_rate, path):
    data = []
    if sampling_rate == 100:
        filenames = df.filename_lr
    else:
        filenames = df.filename_hr

    print(f"📦 Loading {len(filenames)} ECG records at {sampling_rate} Hz ...")
    for f in tqdm(filenames, desc="Loading ECG files", ncols=80):
        signal, _ = wfdb.rdsamp(path + f)
        data.append(signal)
    return np.array(data)

X = load_raw_data(Y, sampling_rate, path)

# ---------------------------------------------------------------
# Load scp_statements.csv for diagnostic aggregation
# ---------------------------------------------------------------
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# ---------------------------------------------------------------
# Split data into train/test (for completeness)
# ---------------------------------------------------------------
test_fold = 10
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[Y.strat_fold != test_fold].diagnostic_superclass
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

# ---------------------------------------------------------------
# Inspect dataset structure and memory info
# ---------------------------------------------------------------
print("\n Data Summary")
print(f"X shape: {X.shape}  (samples, timesteps, leads)")
print(f"Y shape: {Y.shape}")
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"\n Approx memory usage of X: {X.nbytes / 1024**2:.2f} MB")

print("\n Sample diagnostic labels (first 5):")
print(Y['diagnostic_superclass'].head())

print("\n First record detail:")
print(f"ECG sample shape: {X[0].shape}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {X[0].shape[0] / sampling_rate:.2f} seconds")
print(f"Leads: {X[0].shape[1]} (standard 12-lead ECG)")

# ---------------------------------------------------------------
# Visualization of the first record (12-lead ECG)
# ---------------------------------------------------------------
print("\n Plotting the first 12-lead ECG record...")
signal = X[0]  # 第一筆 ECG
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

plt.figure(figsize=(15, 10))
for i in range(12):
    plt.subplot(6, 2, i + 1)
    plt.plot(signal[:, i], linewidth=0.8)
    plt.title(leads[i])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.suptitle("PTB-XL Example ECG (12 Leads, 500 Hz)", y=1.02, fontsize=16)
plt.show()


