import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
import joblib
import optuna
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# seed = random.randint(1, 10000)
seed = 1594
print(f"Seed: {seed}")

data = pd.read_csv('train.csv')

data['Цена_лог'] = np.log1p(data['Цена'])

features = ['Тип_жилья', 'Индекс',
            'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
            'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
            'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
            'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
            'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']

cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

for col in cat_features:
    data[col] = data[col].astype('category')

for col in features:
    if col in cat_features:
        data[col] = data[col].cat.add_categories('unknown')
        data[col] = data[col].fillna('unknown')
    else:
        data[col] = data[col].fillna(data[col].median())

X = data[features]
y = data['Цена_лог']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1e-15, random_state=seed)

data['Цена_лог'] = np.log1p(data['Цена'])

X = data[features]

y = data['Цена_лог']

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_features,
    free_raw_data=True
)

val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

# def objective(trial):
#     params = {
#         'objective': 'huber',
#         'metric': 'mape',
#         'boosting_type': 'gbdt',
#         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
#         'num_leaves': trial.suggest_int('num_leaves', 100, 2000),
#         'max_depth': trial.suggest_int('max_depth', 5, 15),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
#         'verbose': -1,
#         'feature_pre_filter': False
#     }
#
#     opt_model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000)
#     preds = opt_model.predict(X_val)
#     return mape(y_val, preds)
#
#
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=300)
#
# best_params = study.best_params
# print(best_params)

my_params = {
    'objective': 'huber',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'learning_rate': 0.019810775340455174,
    'num_leaves': 167,
    'max_depth': 17,
    'min_data_in_leaf': 19,
    'feature_fraction': 0.7394057006553179,
    'bagging_fraction': 0.9601934105863176,
    'bagging_freq': 8,
    'lambda_l1': 0.00013536407723488536,
    'lambda_l2': 0.0008946862064586757,
    'min_split_gain': 0.004565945687306958,
    'path_smooth': 0.3357692235912594,
    'verbose': -1,
    'is_unbalance': True
}

model = lgb.train(
    my_params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=3033,
    callbacks=[lgb.log_evaluation(0)]
)

# Сохранение модели
joblib.dump(model, f'lgb_model_{seed}.pkl')
