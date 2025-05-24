import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time
import warnings

from aice_fe_components import AICEFEPreprocessor # AAWS and ICSE are used within this
from data_loader_bot_iot import load_bot_iot_data

warnings.filterwarnings('ignore', category=UserWarning) # To suppress some sklearn warnings

def calculate_safety_metric(y_true, y_pred, attack_label=1):
    """ Safety Metric (1 - FNR) """
    cm = confusion_matrix(y_true, y_pred, labels=[0, attack_label]) # Normal=0, Attack=1
    TP = cm[1, 1] # Attacks correctly identified
    FN = cm[1, 0] # Attacks missed (False Negatives)
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
    return 1 - FNR

def run_experiment():
    DATA_DIR = "./data" # Adjust if your data is elsewhere
    BOT_IOT_FILE = "UNSW_2018_IoT_Botnet_Full5pc_1.csv" # Use a smaller BoT-IoT file for quicker test

    print("Loading and preprocessing BoT-IoT data...")
    try:
        X_train, X_test, y_train, y_test, \
        numerical_features, cat_for_icse, other_cat_features = \
            load_bot_iot_data(DATA_DIR, file_name=BOT_IOT_FILE, sample_frac=0.05) # Use 5% sample
    except FileNotFoundError as e:
        print(e)
        print("Please ensure the BoT-IoT dataset is correctly placed and named.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --- Define Preprocessing Methods ---

    # 1. Raw + StdScale (Baseline)
    # For baseline, handle other_cat_features with OneHotEncoding
    baseline_transformers = []
    if numerical_features:
        baseline_transformers.append(('num', StandardScaler(), numerical_features))
    
    # For baseline, all categoricals (ICSE ones + others) are one-hot encoded
    all_categoricals_for_baseline = cat_for_icse + other_cat_features
    if all_categoricals_for_baseline:
         # Ensure all categoricals are strings for OHE
        for df_ in [X_train, X_test]:
            for col in all_categoricals_for_baseline:
                if col in df_.columns:
                    df_[col] = df_[col].astype(str)
        baseline_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), all_categoricals_for_baseline))

    if not baseline_transformers:
        print("Warning: No features selected for baseline preprocessor.")
        preprocessor_baseline = 'passthrough' # or an empty transformer
    else:
        preprocessor_baseline = ColumnTransformer(transformers=baseline_transformers, remainder='drop')


    # 2. AICE-FE
    # Ensure ICSE features are present
    valid_cat_for_icse = [f for f in cat_for_icse if f in X_train.columns]
    valid_other_cat = [f for f in other_cat_features if f in X_train.columns]

    if not numerical_features and not valid_cat_for_icse and not valid_other_cat:
        print("Error: No valid features for AICE-FE. Exiting.")
        return

    preprocessor_aice_fe = AICEFEPreprocessor(
        numerical_features=numerical_features,
        categorical_features_for_icse=valid_cat_for_icse,
        other_categorical_features=valid_other_cat, # Will be one-hot encoded by default
        aaws_lambda=1.0,
        icse_kappa=5.0,
        icse_default_score='global_attack_rate', # Let ICSE calculate this
        normal_label=0, # Assuming 0 is normal in y_train
        attack_label=1  # Assuming 1 is attack in y_train
    )
    
    # --- Define Classifiers ---
    # Using simple params for speed. Paper mentions optimized params.
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1)
    }

    # --- Experiment Loop ---
    results = []

    # Check if any preprocessors are defined
    if preprocessor_baseline == 'passthrough' and not valid_cat_for_icse and not numerical_features:
         print("Skipping experiments as no preprocessors could be defined.")
         return
         
    feature_engineering_methods = {}
    if preprocessor_baseline != 'passthrough':
        feature_engineering_methods["Baseline (StdScale+OHE)"] = preprocessor_baseline
    if numerical_features or valid_cat_for_icse or valid_other_cat: # only add if AICE-FE has features to work on
        feature_engineering_methods["AICE-FE"] = preprocessor_aice_fe
    
    if not feature_engineering_methods:
        print("No feature engineering methods defined. Exiting.")
        return

    for fe_name, preprocessor in feature_engineering_methods.items():
        print(f"\n--- Feature Engineering: {fe_name} ---")
        
        # Create a temporary pipeline for fitting the preprocessor
        # This is to handle cases where preprocessor might not be a full sklearn pipeline object initially
        # For AICEFEPreprocessor, it's fine. For ColumnTransformer, also fine.
        if isinstance(preprocessor, (ColumnTransformer, AICEFEPreprocessor)):
            temp_pipeline = Pipeline([('preprocessor', preprocessor)])
        else: # passthrough or other simple string
            temp_pipeline = Pipeline([('preprocessor', 'passthrough')])


        start_fe_fit = time.time()
        print("Fitting preprocessor...")
        # Need to handle potential errors if no features of a certain type exist
        try:
            if fe_name == "AICE-FE" or isinstance(preprocessor, AICEFEPreprocessor):
                 # AICE-FE needs y for fitting its components
                temp_pipeline.fit(X_train.copy(), y_train) # Use .copy() to avoid SettingWithCopyWarning
            else: # Baseline ColumnTransformer
                temp_pipeline.fit(X_train.copy())
        except ValueError as e:
            print(f"ValueError during preprocessor fitting for {fe_name}: {e}")
            print(f"Numerical features: {numerical_features}")
            print(f"Categorical for ICSE: {cat_for_icse}")
            print(f"Other categoricals: {other_cat_features}")
            print("Skipping this FE method.")
            continue
            
        fe_fit_time = time.time() - start_fe_fit
        print(f"Preprocessor fitting took {fe_fit_time:.2f}s")

        print("Transforming training data...")
        start_fe_transform_train = time.time()
        X_train_transformed = temp_pipeline.transform(X_train.copy())
        fe_transform_train_time = time.time() - start_fe_transform_train
        print(f"Training data transformation took {fe_transform_train_time:.2f}s")

        print("Transforming test data...")
        start_fe_transform_test = time.time()
        X_test_transformed = temp_pipeline.transform(X_test.copy())
        fe_transform_test_time = time.time() - start_fe_transform_test
        print(f"Test data transformation took {fe_transform_test_time:.2f}s")
        
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                transformed_feature_names = preprocessor.get_feature_names_out()
                print(f"Number of transformed features: {len(transformed_feature_names)}")
                # print(f"Transformed feature names (sample): {transformed_feature_names[:10]}")
            except Exception as e:
                print(f"Could not get feature names out: {e}. Shape: {X_train_transformed.shape[1]}")
        else:
            print(f"Number of transformed features: {X_train_transformed.shape[1]}")


        for clf_name, classifier in classifiers.items():
            print(f"\nTraining {clf_name} with {fe_name} features...")
            pipeline = Pipeline([
                # Preprocessor is already fitted and data transformed
                # So we just pass the classifier
                ('classifier', classifier)
            ])
            
            start_train_time = time.time()
            pipeline.fit(X_train_transformed, y_train)
            train_time = time.time() - start_train_time + fe_fit_time + fe_transform_train_time # Total training time

            start_detection_time = time.time()
            y_pred = pipeline.predict(X_test_transformed)
            # Per-sample detection time: (total_prediction_time_on_test_set + transformation_time_for_test_set) / num_test_samples
            detection_time_total = (time.time() - start_detection_time) + fe_transform_test_time
            detection_time_avg_ms = (detection_time_total / len(X_test)) * 1000 if len(X_test) > 0 else 0

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) # Use macro for multiclass, or binary if y is binary
            safety = calculate_safety_metric(y_test, y_pred)
            
            print(f"{clf_name} with {fe_name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score (Macro): {f1:.4f}")
            print(f"  Safety Metric: {safety:.4f}")
            print(f"  Total Training Time (FE fit+transform + Model fit): {train_time:.2f}s")
            print(f"  Avg Detection Time per sample (FE transform + Model predict): {detection_time_avg_ms:.4f}ms")
            
            results.append({
                "FE_Method": fe_name,
                "Classifier": clf_name,
                "Accuracy": acc,
                "F1_Score": f1,
                "Safety": safety,
                "Train_Time_s": train_time,
                "Detection_Time_ms": detection_time_avg_ms
            })

    print("\n--- Summary of Results ---")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("aice_fe_experiment_results.csv", index=False)
    print("\nResults saved to aice_fe_experiment_results.csv")

if __name__ == "__main__":
    run_experiment()