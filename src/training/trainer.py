import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from src.data.preprocessor import OSICPreprocessor
from src.models.fvc_net import build_model
from src.utils.metrics import combined_loss, competition_metric

def train_pipeline(train_df, config):
    os.makedirs('models', exist_ok=True)

    preprocessor = OSICPreprocessor()
    X, y = preprocessor.fit_transform(train_df)

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training Fold {fold+1}...")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = build_model(X.shape[1], config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(loss=combined_loss(config['lambda']), optimizer=optimizer, metrics=[competition_metric])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_competition_metric', patience=config['patience'], mode='max', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_competition_metric', factor=0.5, patience=config['patience']//2, mode='max', min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('models', f'best_model_fold_{fold+1}.keras'), monitor='val_competition_metric', save_best_only=True, mode='max')
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            verbose=0,
            callbacks=callbacks
        )

        val_score = max(history.history['val_competition_metric'])
        fold_scores.append(val_score)
        print(f"Fold {fold+1} Score: {val_score:.4f}")

    avg_score = np.mean(fold_scores)
    print(f"Average CV Score: {avg_score:.4f}")

    # Save performance report
    with open('performance_report.txt', 'w') as f:
        f.write("Model Performance Report\n")
        f.write("=" * 30 + "\n")
        f.write(f"Fold Scores: {fold_scores}\n")
        f.write(f"Average CV Competition Metric: {avg_score:.4f}\n")

    return fold_scores, preprocessor
