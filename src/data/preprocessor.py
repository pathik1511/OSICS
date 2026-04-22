import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class OSICPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.cat_cols = ['Sex', 'SmokingStatus']
        self.num_cols = ['Percent', 'Age', 'baselined_week', 'base_FVC']
        self.patient_baselines = None

    def _extract_baselines(self, df):
        df = df.copy()
        df['Weeks'] = df['Weeks'].astype(int)
        df['min_week'] = df.groupby('Patient')['Weeks'].transform('min')

        # In inference, we might not have 'FVC' in the merged sub_df if we're not careful.
        # But we actually NEED FVC from the test_df to create base_FVC.
        base_df = df.loc[df.Weeks == df.min_week].copy()
        if 'FVC' in base_df.columns:
            base_df = base_df[['Patient', 'FVC', 'min_week']]
            base_df.columns = ['Patient', 'base_FVC', 'min_week']
        else:
            # Fallback if FVC not present (should not happen if test_df is merged correctly)
            base_df = base_df[['Patient', 'min_week']]

        base_df = base_df.groupby('Patient').first().reset_index()
        return base_df

    def fit_transform(self, df):
        self.patient_baselines = self._extract_baselines(df)

        df = df.merge(self.patient_baselines, on='Patient', how='left')
        df['baselined_week'] = df['Weeks'] - df['min_week']

        # One-hot encoding
        dfs_to_concat = [df]
        for col in self.cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            dfs_to_concat.append(dummies)

        df_encoded = pd.concat(dfs_to_concat, axis=1)

        # Only include numeric and encoded columns in features
        self.feature_cols = self.num_cols + [c for c in df_encoded.columns if any(c.startswith(cat) for cat in self.cat_cols) and c not in self.cat_cols]

        df_encoded[self.num_cols] = self.scaler.fit_transform(df_encoded[self.num_cols])
        return df_encoded[self.feature_cols].values.astype('float32'), df_encoded['FVC'].values.reshape(-1, 1).astype('float32')

    def transform(self, df):
        # Use provided df which should have FVC for baseline week (merged from test_df)
        baselines = self._extract_baselines(df)
        df = df.merge(baselines, on='Patient', how='left')
        df['baselined_week'] = df['Weeks'] - df['min_week']

        dfs_to_concat = [df]
        for col in self.cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            dfs_to_concat.append(dummies)
        df_encoded = pd.concat(dfs_to_concat, axis=1)

        # Add missing columns with 0s
        for col in self.feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded[self.num_cols] = self.scaler.transform(df_encoded[self.num_cols])
        return df_encoded[self.feature_cols].values.astype('float32')
