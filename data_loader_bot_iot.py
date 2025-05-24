import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_bot_iot_data(data_dir, file_name="UNSW_2018_IoT_Botnet_Full5pc_1.csv", sample_frac=0.1, test_size=0.3, random_state=42):
    """
    Loads and preprocesses a sample of the BoT-IoT dataset.
    """
    file_path = os.path.join(data_dir, "BoT-IoT", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. "
                                "Please download BoT-IoT dataset (e.g., from Kaggle: 'UNSW IoT Botnet Dataset') "
                                "and place it in 'data/BoT-IoT/' directory.")

    print(f"Loading data from {file_path}...")
    # Load a fraction of the dataset to speed up processing
    # Use a chunk loader if the full file is too large to sample directly
    try:
        df = pd.read_csv(file_path)
        if sample_frac < 1.0:
             df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    except MemoryError:
        print("MemoryError: File is too large. Consider using a smaller file or chunking.")
        # Basic chunking for sampling if direct load fails
        chunk_list = []
        for chunk in pd.read_csv(file_path, chunksize=500000):
            chunk_list.append(chunk.sample(frac=sample_frac, random_state=random_state))
        df = pd.concat(chunk_list, axis=0).reset_index(drop=True)


    print(f"Loaded {len(df)} samples.")

    # Basic Preprocessing (specific to BoT-IoT structure)
    # Drop unnecessary columns (e.g., flow_id, time-based if not used for sequences)
    # For BoT-IoT, 'pkSeqID', 'stime', 'ltime' might be less relevant for non-sequential models
    # 'category' and 'subcategory' are detailed labels, 'attack' is binary.
    df.drop(['pkSeqID', 'stime', 'ltime'], axis=1, errors='ignore', inplace=True)

    # Identify target variable
    # 'attack' is binary (0 for normal, 1 for attack)
    # 'category' and 'subcategory' give more detail
    if 'attack' not in df.columns:
        raise ValueError("'attack' column not found. Check dataset structure.")
    
    y = df['attack']
    X = df.drop(['attack', 'category', 'subcategory'], axis=1) # Drop detailed labels too

    # Convert relevant columns to numeric, coercing errors
    # This is a common step, actual columns depend on the dataset variant
    potential_numeric_cols = ['flgs_number', 'proto_number', 'state_number',
                              'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss',
                              'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts',
                              'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
                              'Stcpb', 'Dtcpb', 'sTos', 'dTos', 'sVid', 'dVid',
                              'seq', 'ack', 'sHops', 'dHops', 'rate'] # Add more as per your specific BoT-IoT version
    
    for col in X.columns:
        if col in potential_numeric_cols or X[col].dtype == 'object': # try to convert object if it might be numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except: # Some columns might be truly categorical strings like IP addresses
                pass
    
    X.fillna(0, inplace=True) # Simple NA fill, consider more sophisticated methods

    # Identify feature types (example)
    # This requires knowing the dataset schema or inspecting df.info() and unique values
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # For BoT-IoT specific example:
    # Numerical: 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', etc.
    # Categorical: 'proto', 'state', 'flgs', 'sport', 'dport', 'saddr', 'daddr'
    # We might want to treat high-cardinality categoricals like 'saddr', 'daddr', 'sport', 'dport' differently
    # For ICSE, paper suggests top 3-5 categorical. Let's pick some common ones.
    
    # Refine based on typical BoT-IoT structure (you might need to adjust)
    all_cols = X.columns.tolist()
    # Example categorical selection for ICSE:
    categorical_for_icse = ['proto', 'state', 'flgs'] 
    # Ensure these are in X and are categorical (or convert them)
    for col in categorical_for_icse:
        if col not in X.columns:
            print(f"Warning: ICSE candidate feature '{col}' not in X. Skipping.")
            categorical_for_icse.remove(col)
        else:
            X[col] = X[col].astype(str) # Ensure string type for ICSE

    # Other categoricals (e.g., IPs, ports - high cardinality, might be dropped or specially handled)
    # For this demo, we'll treat remaining object columns as 'other_categorical'
    other_categorical = [col for col in categorical_features if col not in categorical_for_icse]
    
    # Final numerical features list
    numerical_features = [col for col in numerical_features if col not in categorical_for_icse and col not in other_categorical]

    print(f"Numerical features identified: {numerical_features[:5]}... ({len(numerical_features)} total)")
    print(f"Categorical features for ICSE: {categorical_for_icse}")
    print(f"Other categorical features: {other_categorical[:5]}... ({len(other_categorical)} total)")


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_for_icse, other_categorical