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
import argparse 

from aice_fe_components import AICEFEPreprocessor
from data_loader_bot_iot import load_bot_iot_data
from data_loader_wustl_ehms import load_wustl_ehms_data
from data_loader_mqtt_iot import load_mqtt_iot_data

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def calculate_safety_metric(y_true, y_pred, normal_label=0, attack_label=1):
    """ Safety Metric (1 - FNR) """
    labels = sorted(list(set(y_true) | set(y_pred)))
    if normal_label not in labels or attack_label not in labels:
        # If only one class is present in predictions or true labels, CM might be smaller
        # This usually indicates a problem or extreme imbalance.
        # Fallback: if all are normal, safety is 1; if any attack is missed (FN>0), safety is impacted.
        if attack_label not in y_true: return 1.0 # No attacks to detect
        FN = np.sum((y_true == attack_label) & (y_pred == normal_label))
        TP = np.sum((y_true == attack_label) & (y_pred == attack_label))
        if TP + FN == 0: return 1.0 # No positive instances, or perfect recall if TP > 0
        return TP / (TP + FN) # This is Recall for the positive class (attack_label)

    cm = confusion_matrix(y_true, y_pred, labels=[normal_label, attack_label])
    TP = cm[1, 1]  # Attacks correctly identified
    FN = cm[1, 0]  # Attacks missed (False Negatives)
    
    # FNR = FN / (TP + FN)
    # Safety = 1 - FNR = TP / (TP + FN)  (This is Recall of the attack class)
    if (TP + FN) > 0:
        safety = TP / (TP + FN)
    else: # No actual attack instances in y_true, or none predicted as attack
        if np.sum(y_true == attack_label) == 0: # No attacks in ground truth
            safety = 1.0
        else: # Attacks in ground truth, but none identified as TP or FN (implies all TN or FP for attack class)
            safety = 0.0 
    return safety


def run_experiment(dataset_name, sample_frac=0.1):
    DATA_DIR = "./data"
    NORMAL_LABEL = 0 # Standardized by data loaders
    ATTACK_LABEL = 1 # Standardized by data loaders

    print(f"\n===== Running Experiment for Dataset: {dataset_name.upper()} =====\n")

    if dataset_name.lower() == "bot-iot":
        BOT_IOT_FILE = "UNSW_2018_IoT_Botnet_Full5pc_1.csv" # Adjust if needed
        print("Loading and preprocessing BoT-IoT data...")
        X_train, X_test, y_train, y_test, \
        numerical_features, cat_for_icse, other_cat_features = \
            load_bot_iot_data(DATA_DIR, file_name=BOT_IOT_FILE, sample_frac=sample_frac)
    elif dataset_name.lower() == "wustl-ehms":
        print("Loading and preprocessing WUSTL-EHMS-2020 data...")
        X_train, X_test, y_train, y_test, \
        numerical_features, cat_for_icse, other_cat_features = \
            load_wustl_ehms_data(DATA_DIR, sample_frac_train=sample_frac, sample_frac_test=sample_frac)
    elif dataset_name.lower() == "mqtt-iot":
        MQTT_FILE = "mqtt_dataset.csv" # Adjust if needed, e.g. "train70_test30_balanced_mqttDataset.csv"
        print("Loading and preprocessing MQTT-IoT-IDS2021 data...")
        X_train, X_test, y_train, y_test, \
        numerical_features, cat_for_icse, other_cat_features = \
            load_mqtt_iot_data(DATA_DIR, file_name=MQTT_FILE, sample_frac=sample_frac)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'bot-iot', 'wustl-ehms', 'mqtt-iot'.")

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Unique y_train values: {np.unique(y_train, return_counts=True)}")
    print(f"Unique y_test values: {np.unique(y_test, return_counts=True)}")


    # --- Define Preprocessing Methods ---
    baseline_transformers = []
    if numerical_features:
        baseline_transformers.append(('num', StandardScaler(), numerical_features))
    
    all_categoricals_for_baseline = list(set(cat_for_icse + other_cat_features)) # Use set to avoid duplicates
    if all_categoricals_for_baseline:
        for df_ in [X_train, X_test]: # Ensure string type for OHE
            for col in all_categoricals_for_baseline:
                if col in df_.columns:
                     df_[col] = df_[col].astype(str)
        baseline_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), all_categoricals_for_baseline))

    preprocessor_baseline = ColumnTransformer(transformers=baseline_transformers, remainder='drop') if baseline_transformers else 'passthrough'
    
    valid_num_feats = [f for f in numerical_features if f in X_train.columns]
    valid_cat_for_icse = [f for f in cat_for_icse if f in X_train.columns]
    valid_other_cat = [f for f in other_cat_features if f in X_train.columns]

    preprocessor_aice_fe = AICEFEPreprocessor(
        numerical_features=valid_num_feats,
        categorical_features_for_icse=valid_cat_for_icse,
        other_categorical_features=valid_other_cat,
        aaws_lambda=1.0,
        icse_kappa=5.0, # Paper mentions 5 as default or tuned.
        icse_default_score='global_attack_rate',
        normal_label=NORMAL_LABEL,
        attack_label=ATTACK_LABEL
    )
    
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1)
    }
    results = []
    
    feature_engineering_methods = {}
    if preprocessor_baseline != 'passthrough':
        feature_engineering_methods["Baseline (StdScale+OHE)"] = preprocessor_baseline
    if valid_num_feats or valid_cat_for_icse or valid_other_cat:
        feature_engineering_methods["AICE-FE"] = preprocessor_aice_fe
    
    if not feature_engineering_methods:
        print(f"No features identified or preprocessors configured for {dataset_name}. Skipping.")
        return pd.DataFrame()


    for fe_name, preprocessor in feature_engineering_methods.items():
        print(f"\n--- Feature Engineering: {fe_name} for {dataset_name} ---")
        
        current_X_train, current_X_test = X_train.copy(), X_test.copy()

        start_fe_fit_transform = time.time()
        try:
            if fe_name == "AICE-FE":
                preprocessor.fit(current_X_train, y_train)
            else: # Baseline ColumnTransformer
                preprocessor.fit(current_X_train)
        except Exception as e:
            print(f"Error fitting preprocessor {fe_name} for {dataset_name}: {e}")
            print(f"Num feats: {valid_num_feats}, ICSE_cat: {valid_cat_for_icse}, Other_cat: {valid_other_cat}")
            continue # Skip this FE method if fitting fails

        X_train_transformed = preprocessor.transform(current_X_train)
        X_test_transformed = preprocessor.transform(current_X_test)
        fe_total_time = time.time() - start_fe_fit_transform # Combined fit+transform for train
        
        # Split time for reporting as per paper (TrT includes FE fit + transform on train)
        # For AICE-FE, transform on test is part of DT.
        # For Baseline, transform on test is also part of DT.

        fe_fit_time_approx = fe_total_time # Rough estimate for combined fit/transform on train
        # A more precise TrT for FE would separate fit time from transform time on train data.
        # For simplicity, we'll group them here.
        
        num_transformed_features = X_train_transformed.shape[1]
        print(f"FE processing time (fit on train, transform train): {fe_total_time:.2f}s")
        print(f"Number of transformed features: {num_transformed_features}")
        if num_transformed_features == 0 and (valid_num_feats or valid_cat_for_icse or valid_other_cat):
            print(f"Warning: Zero features after transformation with {fe_name}. Check preprocessor logic.")
            continue


        for clf_name, classifier_template in classifiers.items():
            print(f"\nTraining {clf_name} with {fe_name} features on {dataset_name}...")
            
            # Re-initialize classifier for fresh training
            classifier = type(classifier_template)(**classifier_template.get_params())
            
            pipeline = Pipeline([('classifier', classifier)]) # Preprocessing is done outside pipeline for clarity
            
            model_fit_start_time = time.time()
            pipeline.fit(X_train_transformed, y_train)
            model_fit_time = time.time() - model_fit_start_time
            
            train_time_total = fe_fit_time_approx + model_fit_time # TrT in paper

            # Detection Time (DT) calculation
            # Time to transform test set + time to predict on transformed test set
            dt_fe_transform_start = time.time()
            # X_test_transformed_for_dt = preprocessor.transform(X_test.copy()) # Already transformed
            dt_fe_transform_time = time.time() - dt_fe_transform_start # This is near zero as it's done

            dt_predict_start = time.time()
            y_pred = pipeline.predict(X_test_transformed)
            dt_predict_time = time.time() - dt_predict_start
            
            # The paper's DT: "average time taken (in milliseconds) to classify a single instance from the test set,
            # including the feature transformation time for that instance."
            # So, (transform_test_time + predict_test_time) / num_test_samples
            # Our X_test_transformed is already done. If we re-transform, we need to be careful.
            # Let's use the already transformed X_test_transformed. The fe_transform_time_for_test
            # needs to be estimated or measured carefully if preprocessor.transform is called again.
            # For now, assume test transform time is part of the loop before.
            # Simplified DT: (predict_time_on_test) / num_test_samples.
            # For a more accurate DT as per paper: we need to measure preprocessor.transform(one_sample)
            
            # Let's re-evaluate how DT is best calculated to match the paper's spirit.
            # TrT = FE_fit_train + FE_transform_train + Model_fit_train
            # DT = (FE_transform_test_instance + Model_predict_test_instance)
            # For batch: DT_avg = (Total_FE_transform_test + Total_Model_predict_test) / num_test_samples
            
            # Estimate transform time on test set if not already captured.
            # Since X_test_transformed is already available, this is tricky.
            # For simplicity, we'll use the predict time. A full DT would re-transform.
            # Let's assume the transform time for the entire test set was part of the initial `preprocessor.transform(X_test.copy())`
            # We should capture that duration.
            _dummy_test_transform_start = time.time()
            _ = preprocessor.transform(X_test.copy()[:10]) # Transform a small part to estimate
            _dummy_test_transform_per_10 = time.time() - _dummy_test_transform_start
            estimated_transform_time_per_sample_ms = (_dummy_test_transform_per_10 / 10) * 1000
            
            detection_time_avg_ms = (dt_predict_time / len(X_test) * 1000) + estimated_transform_time_per_sample_ms if len(X_test) > 0 else 0


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            safety = calculate_safety_metric(y_test, y_pred, normal_label=NORMAL_LABEL, attack_label=ATTACK_LABEL)
            
            print(f"{clf_name} with {fe_name} on {dataset_name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score (Macro): {f1:.4f}")
            print(f"  Safety Metric (Recall_Attack): {safety:.4f}")
            print(f"  Total Training Time (TrT): {train_time_total:.2f}s") # FE_fit_transform_train + Model_fit
            print(f"  Avg Detection Time per sample (DT): {detection_time_avg_ms:.4f}ms")
            
            results.append({
                "Dataset": dataset_name,
                "FE_Method": fe_name,
                "Classifier": clf_name,
                "Accuracy": acc,
                "F1_Score": f1,
                "Safety": safety,
                "Train_Time_s": train_time_total,
                "Detection_Time_ms": detection_time_avg_ms,
                "Num_Features": num_transformed_features
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AICE-FE experiments on IoT datasets.")
    parser.add_argument("--dataset", type=str, default="bot-iot", 
                        choices=["bot-iot", "wustl-ehms", "mqtt-iot", "all"],
                        help="Dataset to use: bot-iot, wustl-ehms, mqtt-iot, or all.")
    parser.add_argument("--sample_frac", type=float, default=0.1,
                        help="Fraction of data to sample (e.g., 0.1 for 10%%). Use 1.0 for full dataset (can be slow).")
    
    args = parser.parse_args()

    all_results = []

    if args.dataset == "all":
        datasets_to_run = ["bot-iot", "wustl-ehms", "mqtt-iot"]
    else:
        datasets_to_run = [args.dataset]

    for ds_name in datasets_to_run:
        try:
            dataset_results_df = run_experiment(dataset_name=ds_name, sample_frac=args.sample_frac)
            if not dataset_results_df.empty:
                all_results.append(dataset_results_df)
        except FileNotFoundError as e:
            print(f"Could not run experiment for {ds_name}: {e}")
            print("Please ensure the dataset is correctly placed in the 'data' directory.")
        except Exception as e:
            print(f"An unexpected error occurred for dataset {ds_name}: {e}")
            import traceback
            traceback.print_exc()


    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        print("\n\n===== ===== Summary of All Results ===== =====")
        print(final_results_df)
        results_filename = f"aice_fe_all_experiments_sample{args.sample_frac}.csv"
        final_results_df.to_csv(results_filename, index=False)
        print(f"\nAll results saved to {results_filename}")
    else:
        print("No results generated.")