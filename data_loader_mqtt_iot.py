import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_mqtt_iot_data(data_dir, file_name="mqtt_dataset.csv", sample_frac=0.1, test_size=0.3, random_state=42):
    """
    Loads and preprocesses the MQTT-IoT-IDS2021 (MQTTset) dataset.
    Assumes a single CSV file.
    """
    file_path = os.path.join(data_dir, "MQTT-IoT-IDS2021", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MQTT-IoT dataset file not found: {file_path}. "
                                "Please download an MQTT dataset (e.g., MQTTset from Kaggle, "
                                "and place it in 'data/MQTT-IoT-IDS2021/' directory, renaming if necessary.")

    print(f"Loading MQTT-IoT data from {file_path}...")
    try:
        # Some MQTT datasets can be large
        df = pd.read_csv(file_path)
        if sample_frac < 1.0:
             df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    except MemoryError:
        print("MemoryError during MQTT load. Trying to sample with chunking.")
        chunk_list = []
        for chunk in pd.read_csv(file_path, chunksize=200000): # Adjust chunksize as needed
            chunk_list.append(chunk.sample(frac=sample_frac, random_state=random_state))
        df = pd.concat(chunk_list, axis=0).reset_index(drop=True)
        
    print(f"Loaded {len(df)} samples for MQTT-IoT.")

    # Preprocessing (specific to a common MQTTset structure)
    # Target variable: often 'label' or 'target'. Attacks vs. 'Legitimate' or 'Normal'
    # Example column names: 'target' (0/1), or 'label' (text)
    # Let's assume 'label' needs encoding: 'Legitimate' -> 0, others -> 1
    
    target_col = None
    if 'target' in df.columns and df['target'].nunique() == 2: # Often 0/1 already
        target_col = 'target'
        y = df[target_col]
    elif 'label' in df.columns:
        target_col = 'label'
        # Assuming 'Legitimate' or 'Normal' is the normal class. Adjust if different.
        normal_class_names = ['Legitimate', 'Normal', 'legitimate', 'normal']
        y = df[target_col].apply(lambda x: 0 if x in normal_class_names else 1)
    else:
        raise ValueError("Target column ('target' or 'label') not found in MQTT dataset.")

    # Drop irrelevant or highly detailed label columns if 'y' is derived
    cols_to_drop = [target_col, 'class', 'sub_class', 'frame.time_epoch', 'mqtt.protoname', 'mqtt.conack.flags', 'mqtt.conack.val', 'mqtt.sub.qos', 'mqtt. unsub.qos']
    # Also drop high cardinality string features that are hard to encode simply unless specifically used
    # e.g., mqtt.clientid, mqtt.username, mqtt.passwd, mqtt.msg (payload)
    cols_to_drop.extend(['mqtt.clientid', 'mqtt.username', 'mqtt.passwd', 'mqtt.msg', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp'])
    
    X = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Identify feature types
    # Convert potential numericals, handle NaNs
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                # Attempt to convert to numeric if possible (e.g. "1.0" as string)
                X[col] = pd.to_numeric(X[col], errors='raise') 
            except (ValueError, TypeError):
                X[col].fillna('unknown', inplace=True) # For true categoricals
                X[col] = X[col].astype(str)
        else: # Numerical
            X[col].fillna(X[col].median(), inplace=True) # Fill numerical NaNs with median

    # Handle boolean columns explicitly -> convert to int 0/1
    for col in X.select_dtypes(include='bool').columns:
        X[col] = X[col].astype(int)

    # Re-identify after conversion
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # For MQTT-IoT:
    # Numerical: 'tcp.len', 'mqtt.len', 'mqtt.topic_len', 'mqtt.kalive', 'mqtt.msgid', 'mqtt.seq', etc.
    # Categorical for ICSE: 'mqtt.msgtype', 'mqtt.qos', 'tcp.flags' (if present & categorical)
    # Other Categorical: 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'mqtt.topic'
    
    # Ensure tcp.flags is treated as categorical if it exists
    if 'tcp.flags' in X.columns:
        X['tcp.flags'] = X['tcp.flags'].astype(str) # Often hex strings
        
    categorical_for_icse = ['mqtt.msgtype'] # mqtt.qos and tcp.flags might not always be suitable or present
    if 'mqtt.qos' in X.columns and X['mqtt.qos'].nunique() < 10: # check cardinality
        categorical_for_icse.append('mqtt.qos')
    if 'tcp.flags' in X.columns and X['tcp.flags'].nunique() < 20: # check cardinality
         categorical_for_icse.append('tcp.flags')

    # Ensure these are in X and are strings
    valid_cat_for_icse = []
    for col in categorical_for_icse:
        if col in X.columns:
            X[col] = X[col].astype(str)
            valid_cat_for_icse.append(col)
        else:
            print(f"Warning (MQTT-IoT): ICSE candidate feature '{col}' not in X. Skipping.")
    categorical_for_icse = valid_cat_for_icse

    all_object_cols = X.select_dtypes(include='object').columns.tolist()
    other_categorical = [col for col in all_object_cols if col not in categorical_for_icse]
    
    # Refine numerical features list
    numerical_features = [col for col in X.columns if X[col].dtype in [np.number, 'int64', 'float64'] and col not in categorical_for_icse and col not in other_categorical]

    print(f"MQTT-IoT - Numerical features ({len(numerical_features)}): {numerical_features[:5]}...")
    print(f"MQTT-IoT - Categorical for ICSE: {categorical_for_icse}")
    print(f"MQTT-IoT - Other categorical ({len(other_categorical)}): {other_categorical[:5]}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_for_icse, other_categorical