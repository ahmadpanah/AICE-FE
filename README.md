# AICE-FE: Enhancing IoT Intrusion Detection

This repository provides a Python implementation of the "AICE-FE: Enhancing IoT Intrusion Detection through Adaptive Scaling and Interactive Contextual Feature Engineering" framework, as described in the research paper.

AICE-FE introduces two novel feature engineering components:
1.  **Adaptive Anomaly-Weighted Scaling (AAWS):** Intelligently scales numerical features based on their statistical distinctiveness between normal and attack classes.
2.  **Interactive Contextual State Encoding (ICSE):** Creates a new, highly informative feature by identifying and scoring composite states derived from multiple categorical features based on their empirical association with malicious activity.

This implementation allows for experimentation with AICE-FE on various IoT intrusion detection datasets and compares its performance against a baseline feature engineering approach.

## Features

*   Implementation of AAWS and ICSE components.
*   A combined `AICEFEPreprocessor` for easy integration into Scikit-learn pipelines.
*   Data loaders and preprocessing scripts for three common IoT IDS datasets:
    *   BoT-IoT
    *   WUSTL-EHMS-2020
    *   MQTT-IoT-IDS2021 (MQTTset)
*   Experiment script to evaluate AICE-FE using Decision Tree and Random Forest classifiers.
*   Calculates standard performance metrics: Accuracy, F1-Score, Safety Metric (Attack Recall), Training Time (TrT), and average Detection Time per sample (DT).
*   Command-line interface to select datasets and data sampling fractions.
*   Outputs results to a CSV file.

## Directory Structure
```
aice-fe-iot-ids/
|-- data/
| |-- BoT-IoT/
| | |-- UNSW_2018_IoT_Botnet_Full5pc_1.csv (Example - Place your BoT-IoT CSVs here)
| |-- WUSTL-EHMS-2020/
| | |-- EHMS_Train.csv
| | |-- EHMS_Test.csv
| |-- MQTT-IoT-IDS2021/
| | |-- mqtt_dataset.csv (Example - Place your MQTT dataset CSV here)
|-- aice_fe_components.py # Core AAWS, ICSE, and AICEFEPreprocessor classes
|-- data_loader_bot_iot.py # Data loading and preprocessing for BoT-IoT
|-- data_loader_wustl_ehms.py # Data loading and preprocessing for WUSTL-EHMS-2020
|-- data_loader_mqtt_iot.py # Data loading and preprocessing for MQTT-IoT-IDS2021
|-- main_experiment.py # Main script to run experiments
|-- requirements.txt # Python dependencies
|-- README.md # This file
```
## Prerequisites

*   Python 3.7+
*   Required Python libraries (see `requirements.txt`):
    *   pandas
    *   numpy
    *   scikit-learn

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/aice-fe-iot-ids.git
    cd aice-fe-iot-ids
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` is not provided, install manually: `pip install pandas numpy scikit-learn`)

3.  **Download Datasets:**
    You need to download the datasets and place them in the `data/` directory as per the structure above.

    *   **BoT-IoT:**

        *   Source: [UNSW IoT Botnet Dataset Page](https://research.unsw.edu.au/projects/bot-iot-dataset)
        *   Place CSV files (e.g., `UNSW_2018_IoT_Botnet_Full5pc_1.csv`) in `data/BoT-IoT/`.
    *   **WUSTL-EHMS-2020:**
        *   Source: [WUSTL EHMS Dataset Page](https://www.cse.wustl.edu/~jain/ehms/index.html)
        *   Download `EHMS_Train.csv` and `EHMS_Test.csv`.
        *   Place them in `data/WUSTL-EHMS-2020/`.
    *   **MQTT-IoT-IDS2021 (MQTTset):**
        *   Source: [MQTT-IoT-IDS2021 (MQTTset) Page](https://www.kaggle.com/datasets/cnrieiit/mqttset/data)
        *   You might need to rename the downloaded file (e.g., to `mqtt_dataset.csv` as used by default in the loader) or adjust the filename in `data_loader_mqtt_iot.py`.
        *   Place the CSV file in `data/MQTT-IoT-IDS2021/`.

    **Note:** The data loaders (`data_loader_*.py`) contain preprocessing logic specific to common versions of these datasets. If you use a different version or file structure, you may need to adjust the column names, feature identification, and target variable processing within these loader scripts.

## Usage

Run experiments using the `main_experiment.py` script. You can specify the dataset and the fraction of data to sample.

**Command-line arguments:**

*   `--dataset`: Which dataset to use.
    *   `bot-iot` (default)
    *   `wustl-ehms`
    *   `mqtt-iot`
    *   `all` (runs experiments for all three datasets sequentially)
*   `--sample_frac`: Fraction of the dataset to sample for training/testing (e.g., `0.1` for 10%). Default is `0.1`. Use `1.0` for the full dataset (can be very time-consuming and memory-intensive).

**Examples:**

*   **Run AICE-FE on a 10% sample of the BoT-IoT dataset:**
    ```bash
    python main_experiment.py --dataset bot-iot --sample_frac 0.1
    ```

*   **Run AICE-FE on a 20% sample of the WUSTL-EHMS-2020 dataset:**
    ```bash
    python main_experiment.py --dataset wustl-ehms --sample_frac 0.2
    ```

*   **Run AICE-FE on a 5% sample of the MQTT-IoT dataset:**
    ```bash
    python main_experiment.py --dataset mqtt-iot --sample_frac 0.05
    ```

*   **Run AICE-FE on a 10% sample of all available datasets:**
    ```bash
    python main_experiment.py --dataset all --sample_frac 0.1
    ```

The script will output performance metrics to the console and save a summary of all results to a CSV file (e.g., `aice_fe_all_experiments_sample0.1.csv`).

## Implementation Details

*   **`aice_fe_components.py`**: Contains the `AAWS`, `ICSE`, and `AICEFEPreprocessor` classes, designed to be compatible with Scikit-learn's `BaseEstimator` and `TransformerMixin`.
*   **`data_loader_*.py` files**: Handle loading, basic cleaning, feature type identification, and train-test splitting for each specific dataset. They standardize the target labels (0 for normal, 1 for attack).
*   **`main_experiment.py`**: Orchestrates the experiments, applies the chosen feature engineering method (Baseline or AICE-FE), trains classifiers, and evaluates performance.

## Customization

*   **Adding New Datasets:**
    1.  Create a new `data_loader_yourdataset.py` file, implementing a function similar to the existing ones to load, preprocess, and split your data. Ensure it returns `X_train, X_test, y_train, y_test, numerical_features, categorical_for_icse, other_categorical_features`.
    2.  Import your new loader in `main_experiment.py` and add an `elif` condition to the dataset selection logic.
*   **Changing Classifiers:**
    Modify the `classifiers` dictionary in `main_experiment.py` to include other Scikit-learn classifiers or adjust their hyperparameters.
*   **Feature Selection for ICSE:**
    The current implementation heuristically selects categorical features for ICSE within each data loader. For more robust selection, you could integrate MDA (Mean Decrease in Accuracy) or other feature importance techniques.
*   **Hyperparameter Tuning:**
    The current script uses default hyperparameters for AICE-FE components and classifiers. For optimal performance, consider implementing hyperparameter tuning (e.g., using `GridSearchCV` or `RandomizedSearchCV` from Scikit-learn).

## TODO / Potential Enhancements

*   Implement MDA-Selected+StdScale and PCA comparison methods from the paper.
*   Add MLP and SVM classifiers.
*   More sophisticated handling of high-cardinality categorical features.
*   Automated selection of categorical features for ICSE using MDA.
*   Integration of hyperparameter optimization for AICE-FE and ML models.
*   More detailed logging and visualization of results.