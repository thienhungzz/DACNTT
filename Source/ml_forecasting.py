import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import os

def train_xgboost(X_train, y_train, X_val, y_val, model_dir):
    print("--- Training XGBoost ---")
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'count:poisson',
        'eval_metric': 'rmse',
    }
    model = xgb.XGBRegressor(**params)
    early_stop = xgb.callback.EarlyStopping(rounds=30, metric_name='rmse', data_name='validation_1', save_best=True)

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=50, callbacks=[early_stop])
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "XGBoost_Final.json"))
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, model_dir):
    print("--- Training LightGBM ---")
    params = {
        'objective': 'poisson',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 127,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    model = lgb.LGBMRegressor(n_estimators=1000, **params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, first_metric_only=True),
        lgb.log_evaluation(period=50)
    ]
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], callbacks=callbacks)
    os.makedirs(model_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(model_dir, "LightGBM_Final.txt"))
    return model

def train_two_stage(X_train, y_train, X_val, y_val, model_dir):
    print("--- Training Two-Stage Model ---")
    y_train_capped = np.clip(y_train, 0, 50)
    y_val_capped = np.clip(y_val, 0, 50)

    y_train_clf = (y_train_capped > 0).astype(int)
    y_val_clf = (y_val_capped > 0).astype(int)

    model_clf = lgb.LGBMClassifier(n_estimators=1000, objective='binary', learning_rate=0.05, n_jobs=-1)
    model_clf.fit(X_train, y_train_clf, eval_set=[(X_val, y_val_clf)], callbacks=[lgb.early_stopping(30)])

    mask_train_pos = y_train_capped > 0
    mask_val_pos = y_val_capped > 0
    model_reg = lgb.LGBMRegressor(n_estimators=1000, objective='tweedie', learning_rate=0.05, n_jobs=-1)
    model_reg.fit(X_train[mask_train_pos], y_train_capped[mask_train_pos], eval_set=[(X_val[mask_val_pos], y_val_capped[mask_val_pos])], callbacks=[lgb.early_stopping(30)])

    os.makedirs(model_dir, exist_ok=True)
    model_clf.booster_.save_model(os.path.join(model_dir, "TwoStage_Classifier.txt"))
    model_reg.booster_.save_model(os.path.join(model_dir, "TwoStage_Regressor.txt"))
    return model_clf, model_reg

def train_lstm(train_ds, val_ds, TIME_STEPS, X_seq_train, num_stores, num_items, num_events, model_dir):
    print("--- Training LSTM Model ---")
    seq_input = Input(shape=(TIME_STEPS, X_seq_train.shape[2]), name='seq_input')
    lstm_out = LSTM(64, return_sequences=False)(seq_input)

    store_input = Input(shape=(1,), name='store_input')
    store_emb = Embedding(input_dim=num_stores, output_dim=4, name='store_emb')(store_input)
    store_vec = Flatten()(store_emb)

    item_input = Input(shape=(1,), name='item_input')
    item_emb = Embedding(input_dim=num_items, output_dim=16, name='item_emb')(item_input)
    item_vec = Flatten()(item_emb)

    event_input = Input(shape=(1,), name='event_input')
    event_emb = Embedding(input_dim=num_events, output_dim=4, name='event_emb')(event_input)
    event_vec = Flatten()(event_emb)

    merged = Concatenate()([lstm_out, store_vec, item_vec, event_vec])
    dense_1 = Dense(64, activation='relu')(merged)
    drop_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(32, activation='relu')(drop_1)
    output = Dense(1, activation='relu', dtype='float32', name='sales_output')(dense_2)

    model_emb = Model(inputs=[seq_input, store_input, item_input, event_input], outputs=output)
    model_emb.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    model_path = os.path.join(model_dir, "LSTM_Embedding_Final.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model_emb.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks, verbose=1)
    return model_emb
