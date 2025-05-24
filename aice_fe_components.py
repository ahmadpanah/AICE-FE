import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter

class AAWS(BaseEstimator, TransformerMixin):
    """
    Adaptive Anomaly-Weighted Scaling (AAWS)
    Scales numerical features based on their statistical distinctiveness
    between normal and attack classes.
    """
    def __init__(self, lambda_val=1.0, epsilon=1e-6, normal_label=0, attack_label=1):
        self.lambda_val = lambda_val
        self.epsilon = epsilon
        self.normal_label = normal_label
        self.attack_label = attack_label
        self.params_ = {} # To store mu_N, sigma_N, gamma for each feature

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.features_ = X.columns
        y_series = pd.Series(y)

        for col in self.features_:
            X_col = X[col]
            
            X_normal = X_col[y_series == self.normal_label]
            X_attack = X_col[y_series == self.attack_label]

            mu_N_k = X_normal.mean()
            sigma_N_k = X_normal.std()
            if pd.isna(sigma_N_k) or sigma_N_k < self.epsilon:
                sigma_N_k = self.epsilon

            if len(X_attack) > 1:
                mu_A_k = X_attack.mean()
                sigma_A_k = X_attack.std()
                if pd.isna(sigma_A_k) or sigma_A_k < self.epsilon:
                    sigma_A_k = self.epsilon
            elif len(X_attack) == 1: # Only one attack sample
                mu_A_k = X_attack.iloc[0]
                sigma_A_k = self.epsilon # No std dev for a single point
            else: # No attack samples for this feature's split
                mu_A_k = mu_N_k 
                sigma_A_k = sigma_N_k # or self.epsilon

            numerator_gamma = np.abs(mu_A_k - mu_N_k)
            denominator_gamma = sigma_N_k + sigma_A_k + self.epsilon
            
            gamma_k = 1 + self.lambda_val * (numerator_gamma / denominator_gamma)
            
            self.params_[col] = {'mu_N': mu_N_k, 'sigma_N': sigma_N_k, 'gamma': gamma_k}
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features_)
        
        X_transformed = X.copy()
        for col in self.features_:
            if col in self.params_:
                p = self.params_[col]
                X_transformed[col] = p['gamma'] * (X[col] - p['mu_N']) / (p['sigma_N'] + self.epsilon)
            else:
                # Should not happen if fit was called correctly
                print(f"Warning: Feature {col} not found in learned AAWS parameters.")
                X_transformed[col] = (X[col] - X[col].mean()) / (X[col].std() + self.epsilon) # Fallback
        return X_transformed.values # Return NumPy array for consistency with sklearn

class ICSE(BaseEstimator, TransformerMixin):
    """
    Interactive Contextual State Encoding (ICSE)
    Creates a new feature by scoring composite states derived from multiple
    categorical features based on their association with malicious activity.
    """
    def __init__(self, categorical_features_to_encode, kappa=5.0, 
                 default_score_unseen=0.0, normal_label=0, attack_label=1,
                 delimiter='-'):
        self.categorical_features_to_encode = categorical_features_to_encode
        self.kappa = kappa
        self.default_score_unseen = default_score_unseen # Or global attack rate
        self.normal_label = normal_label
        self.attack_label = attack_label
        self.delimiter = delimiter
        self.state_scores_ = {}

    def _create_state_string(self, row):
        return self.delimiter.join(str(row[col]) for col in self.categorical_features_to_encode)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            # Assuming X has columns in the order they were passed if it's a numpy array
            # This is risky, better to enforce DataFrame input for ICSE
            raise ValueError("ICSE requires a Pandas DataFrame for X during fit.")

        y_series = pd.Series(y)
        
        # Create state strings
        state_strings = X[self.categorical_features_to_encode].astype(str).apply(self._create_state_string, axis=1)
        
        state_counts_total = Counter(state_strings)
        
        # Count states in attack samples
        attack_state_strings = state_strings[y_series == self.attack_label]
        state_counts_attack = Counter(attack_state_strings)
        
        # Calculate scores
        for state, N_su in state_counts_total.items():
            Z_su = state_counts_attack.get(state, 0)
            score = Z_su / (N_su + self.kappa)
            self.state_scores_[state] = score
            
        # Optional: calculate global attack rate for default_score_unseen if not set
        if self.default_score_unseen == 'global_attack_rate': # Custom value to trigger this
            global_attack_rate = (y_series == self.attack_label).mean()
            self.default_score_unseen_ = global_attack_rate
            print(f"ICSE: Using global attack rate {global_attack_rate:.4f} for unseen states.")
        else:
            self.default_score_unseen_ = self.default_score_unseen

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
             raise ValueError("ICSE requires a Pandas DataFrame for X during transform.")

        state_strings = X[self.categorical_features_to_encode].astype(str).apply(self._create_state_string, axis=1)
        
        icse_feature = state_strings.map(self.state_scores_).fillna(self.default_score_unseen_)
        
        # Return as a 2D NumPy array (single column)
        return icse_feature.values.reshape(-1, 1)


class AICEFEPreprocessor(BaseEstimator, TransformerMixin):
    """
    Combines AAWS for numerical features and ICSE for selected categorical features.
    Remaining categorical features can be one-hot encoded or dropped.
    """
    def __init__(self, numerical_features, categorical_features_for_icse, 
                 other_categorical_features=None,
                 aaws_lambda=1.0, aaws_epsilon=1e-6,
                 icse_kappa=5.0, icse_default_score=0.0, # 0 or 'global_attack_rate'
                 normal_label=0, attack_label=1,
                 handle_other_categorical='onehot'): # 'onehot', 'drop'
        
        self.numerical_features = numerical_features
        self.categorical_features_for_icse = categorical_features_for_icse
        self.other_categorical_features = other_categorical_features if other_categorical_features else []
        
        self.aaws_params = {'lambda_val': aaws_lambda, 'epsilon': aaws_epsilon, 
                            'normal_label': normal_label, 'attack_label': attack_label}
        self.icse_params = {'kappa': icse_kappa, 'default_score_unseen': icse_default_score,
                            'normal_label': normal_label, 'attack_label': attack_label}
        self.handle_other_categorical = handle_other_categorical

        self.aaws_transformer_ = AAWS(**self.aaws_params)
        self.icse_transformer_ = ICSE(self.categorical_features_for_icse, **self.icse_params)
        
        if self.handle_other_categorical == 'onehot' and self.other_categorical_features:
            self.onehot_encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            self.onehot_encoder_ = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("AICEFEPreprocessor requires X to be a Pandas DataFrame.")

        self.feature_names_in_ = X.columns.tolist() # Store original feature names

        # Fit AAWS
        if self.numerical_features:
            X_num = X[self.numerical_features]
            self.aaws_transformer_.fit(X_num, y)
        
        # Fit ICSE
        if self.categorical_features_for_icse:
            X_cat_icse = X[self.categorical_features_for_icse]
            self.icse_transformer_.fit(X_cat_icse, y)
            
        # Fit OneHotEncoder for other categoricals
        if self.onehot_encoder_ and self.other_categorical_features:
            X_cat_other = X[self.other_categorical_features].astype(str) # Ensure string type
            self.onehot_encoder_.fit(X_cat_other)
            
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("AICEFEPreprocessor requires X to be a Pandas DataFrame.")

        transformed_parts = []
        self.transformed_feature_names_ = []

        # Transform AAWS
        if self.numerical_features:
            X_num = X[self.numerical_features]
            X_num_transformed = self.aaws_transformer_.transform(X_num)
            transformed_parts.append(X_num_transformed)
            self.transformed_feature_names_.extend([f"aaws_{col}" for col in self.numerical_features])
        
        # Transform ICSE
        if self.categorical_features_for_icse:
            X_cat_icse = X[self.categorical_features_for_icse]
            X_icse_transformed = self.icse_transformer_.transform(X_cat_icse)
            transformed_parts.append(X_icse_transformed)
            self.transformed_feature_names_.append("icse_feature")
            
        # Transform OneHotEncoder for other categoricals
        if self.onehot_encoder_ and self.other_categorical_features:
            X_cat_other = X[self.other_categorical_features].astype(str)
            X_cat_other_transformed = self.onehot_encoder_.transform(X_cat_other)
            transformed_parts.append(X_cat_other_transformed)
            # Get feature names from OneHotEncoder
            try:
                ohe_feature_names = self.onehot_encoder_.get_feature_names_out(self.other_categorical_features)
                self.transformed_feature_names_.extend(ohe_feature_names)
            except AttributeError: # older sklearn
                 # Manually create names if get_feature_names_out not available or fails
                for i, col in enumerate(self.other_categorical_features):
                    for j in range(X_cat_other_transformed.shape[1]): # This part is tricky without knowing categories
                        if X_cat_other_transformed[:,j].sum() > 0: # A rough way
                             self.transformed_feature_names_.append(f"ohe_{col}_{j}")


        if not transformed_parts:
            return np.array([]) # Or raise error

        return np.concatenate(transformed_parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        # This method allows compatibility with sklearn's get_feature_names_out
        # if called after transform.
        if hasattr(self, 'transformed_feature_names_'):
            return self.transformed_feature_names_
        else:
            # Fallback or raise error if transform hasn't been called
            raise NotFittedError("Transformer has not been fitted or transformed yet.")