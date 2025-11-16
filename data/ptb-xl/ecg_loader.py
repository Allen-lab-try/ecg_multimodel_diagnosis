# ================================================================
# ecg_loader.py  —  PTB-XL ECG Loading & Visualization Module
# ================================================================
import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


# ================================================================
# Main Function: Load PTB-XL ECG Subset
# ================================================================
def load_ptbxl_subset(path, sampling_rate=500, num_records=500, max_workers=8):
    """
    Load a subset of the PTB-XL dataset for fast testing or visualization.

    Returns:
        X : np.ndarray of shape (N, T, 12)  - ECG signal data
        Y : pd.DataFrame with annotations (includes diagnostic_superclass)
    """

    # --- Check dataset path ---
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset path not found: {path}")

    # --- Read annotation CSV ---
    csv_path = os.path.join(path, 'ptbxl_database.csv')
    Y = pd.read_csv(csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y = Y.iloc[:num_records]

    # --- Load ECG signals in parallel ---
    X = _load_raw_data(Y, sampling_rate, path, max_workers=max_workers)

    # --- Load diagnostic mapping table ---
    agg_path = os.path.join(path, 'scp_statements.csv')
    agg_df = pd.read_csv(agg_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        """Aggregate diagnostic superclass information"""
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Add diagnostic superclass column
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    print(f"✅ Loaded shape: {X.shape}, labels: {len(Y)}")
    return X, Y


# ================================================================
# Sub-function: Parallel File Loading for Speed
# ================================================================
def _load_raw_data(df, sampling_rate, path, max_workers=8):
    """
    Load ECG signal records using ThreadPoolExecutor (parallel read).
    """
    filenames = df.filename_hr if sampling_rate == 500 else df.filename_lr
    print(f"📦 Loading {len(filenames)} ECG records at {sampling_rate} Hz using {max_workers} threads...")

    def read_one(f):
        """Read a single ECG record"""
        signal, _ = wfdb.rdsamp(os.path.join(path, f))
        return signal

    # Parallel execution
    data = [None] * len(filenames)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_one, f): i for i, f in enumerate(filenames)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading ECG files", ncols=80):
            idx = futures[future]
            try:
                data[idx] = future.result()
            except Exception as e:
                print(f"❌ Error reading {filenames[idx]}: {e}")
                data[idx] = np.zeros((5000, 12))  # fallback zero signal
    return np.array(data)


# ================================================================
# Plotting Function — Visualize First Record (12 Leads)
# ================================================================
def plot_first_record(X, leads=None, sampling_rate=500):
    """
    Plot the first ECG record and display all 12 leads.
    """
    if leads is None:
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    signal = X[0]
    plt.figure(figsize=(15, 10))
    for i in range(12):
        plt.subplot(6, 2, i + 1)
        plt.plot(signal[:, i], linewidth=0.8)
        plt.title(leads[i])
        plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.suptitle(f"PTB-XL Example ECG (12 Leads, {sampling_rate} Hz)", y=1.02)
    plt.show()


# ================================================================
# Standalone Test Execution
# ================================================================
if __name__ == "__main__":

    # 📁 Modify to your PTB-XL dataset path
    path = r"E:\professor_yen\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    print("🚀 Running ecg_loader standalone test ...")
    try:
        X, Y = load_ptbxl_subset(path, sampling_rate=500, num_records=21837, max_workers=8)
        print(f"✅ Sample loaded: {X.shape}")
        print(Y[['strat_fold', 'diagnostic_superclass']].head())

        plot_first_record(X)

    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)


