import pandas as pd
import numpy as np
import tensorflow as tf
import os

def quantile_sum(x):
    return x[0] + tf.cumsum(x[1], axis=1)

def generate_submission(test_df, sample_sub, preprocessor, config):
    sub_df = sample_sub.copy()
    sub_df[['Patient', 'Weeks']] = sub_df.Patient_Week.str.split("_", expand=True)
    sub_df['Weeks'] = sub_df['Weeks'].astype(int)

    sub_df = sub_df.merge(test_df.drop('Weeks', axis=1), on="Patient")

    X_test = preprocessor.transform(sub_df)

    test_preds = np.zeros((X_test.shape[0], 3))
    n_folds = config['n_folds']

    for fold in range(n_folds):
        model_path = os.path.join('models', f'best_model_fold_{fold+1}.keras')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects={'quantile_sum': quantile_sum})
            test_preds += model.predict(X_test, batch_size=config['batch_size'], verbose=0) / n_folds

    sub_df['FVC'] = test_preds[:, 1]
    sub_df['Confidence'] = test_preds[:, 2] - test_preds[:, 0]
    sub_df['Confidence'] = np.maximum(sub_df['Confidence'], 70.0)

    for i in range(len(test_df)):
        patient = test_df.Patient.iloc[i]
        week = test_df.Weeks.iloc[i]
        mask = sub_df.Patient_Week == f"{patient}_{week}"
        sub_df.loc[mask, 'FVC'] = test_df.FVC.iloc[i]
        sub_df.loc[mask, 'Confidence'] = 70.0

    return sub_df[['Patient_Week', 'FVC', 'Confidence']]
