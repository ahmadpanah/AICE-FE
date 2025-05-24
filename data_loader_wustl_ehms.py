import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_wustl_ehms_data(data_dir, train_file="EHMS_Train.csv", test_file="EHMS_Test.csv", sample_frac_train=1.0, sample_frac_test=1.0, random_state=42):
    """
    Loads and preprocesses the WUSTL-EHMS-2020 dataset.
    """
    train_file_path = os.path.join(data_dir, "WUSTL-EHMS-2020", train_file)
    test_file_path = os.path.join(data_dir, "WUSTL-EHMS-2020", test_file)

    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"WUSTL-EHMS dataset files not found. "
                                "Please download EHMS_Train.csv and EHMS_Test.csv "
                                "and place them in 'data/WUSTL-EHMS-2020/' directory.")

    print(f"Loading WUSTL-EHMS training data from {train_file_path}...")
    df_train = pd.read_csv(train_file_path)
    if sample_frac_train < 1.0:
        df_train = df_train.sample(frac=sample_frac_train, random_state=random_state).reset_index(drop=True)

    print(f"Loading WUSTL-EHMS test data from {test_file_path}...")
    df_test = pd.read_csv(test_file_path)
    if sample_frac_test < 1.0:
        df_test = df_test.sample(frac=sample_frac_test, random_state=random_state).reset_index(drop=True)

    print(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples for WUSTL-EHMS.")

    # Preprocessing
    # Columns: Time, Source, Destination, Protocol, Length, Info, Label
    
    # Target variable 'Label'
    # "Normal" -> 0, Attacks ("DoS", "Data_Theft", "MITM") -> 1
    for df in [df_train, df_test]:
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Normal' else 1)

    y_train = df_train['Label']
    X_train = df_train.drop(['Time', 'Info', 'Label'], axis=1, errors='ignore') # Drop Time and Info

    y_test = df_test['Label']
    X_test = df_test.drop(['Time', 'Info', 'Label'], axis=1, errors='ignore')
    
    # Fill NA values - simple fill with 0 for numerical and 'unknown' for categorical
    # Identify feature types more robustly after initial cleaning
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col].fillna('unknown', inplace=True)
            X_test[col].fillna('unknown', inplace=True)
            # Ensure consistency for one-hot encoding
            common_categories = list(set(X_train[col].unique()) | set(X_test[col].unique()))
            X_train[col] = pd.Categorical(X_train[col], categories=common_categories)
            X_test[col] = pd.Categorical(X_test[col], categories=common_categories)
            X_train[col] = X_train[col].astype(str) # For ICSE and OHE
            X_test[col] = X_test[col].astype(str)
        else: # Numerical
            median_val_train = X_train[col].median() # Use median for robustness
            X_train[col].fillna(median_val_train, inplace=True)
            X_test[col].fillna(median_val_train, inplace=True) # Use train median for test

    # Identify feature types
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # For WUSTL-EHMS:
    # Numerical: 'Length'
    # Categorical for ICSE: 'Protocol'
    # Other Categorical: 'Source', 'Destination' (IP addresses)
    
    categorical_for_icse = ['Protocol']
    # Ensure these are in X_train and are strings
    for col in categorical_for_icse:
        if col not in X_train.columns:
            print(f"Warning (WUSTL-EHMS): ICSE candidate feature '{col}' not in X_train. Skipping.")
            categorical_for_icse.remove(col)
        else:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

    all_object_cols = X_train.select_dtypes(include='object').columns.tolist()
    other_categorical = [col for col in all_object_cols if col not in categorical_for_icse]

    # Re-check numerical features (if any categorical were misidentified)
    numerical_features = [col for col in X_train.columns if col not in categorical_for_icse and col not in other_categorical]
    
    print(f"WUSTL-EHMS - Numerical features: {numerical_features}")
    print(f"WUSTL-EHMS - Categorical for ICSE: {categorical_for_icse}")
    print(f"WUSTL-EHMS - Other categorical: {other_categorical}")

    return X_train, X_test, y_train, y_test, numerical_features, categorical_for_icse, other_categorical