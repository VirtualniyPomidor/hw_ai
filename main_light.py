import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import optuna
from sklearn.metrics import mean_absolute_percentage_error as mape
from optuna.pruners import MedianPruner
import warnings
from sklearn.preprocessing import RobustScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Фиксация сидов
seed = 62053
np.random.seed(seed)
print(f"Seed: {seed}")

# Загрузка данных
data = pd.read_csv('train.csv')

# Удаление выбросов
data = data.loc[~((data['Тип_жилья'] == 'дуплекс') & (data['Площадь'] > 200))]
data = data.loc[~((data['Тип_жилья'] == 'комната') & (data['Площадь'] > 50))]
data = data.loc[~((data['Тип_жилья'] == 'лофт') & (data['Площадь'] > 500))]
data = data.loc[~((data['Тип_жилья'] == 'разное') & (data['Площадь'] > 15000))]
data = data.loc[~((data['Тип_жилья'] == 'квартира') & (data['Площадь'] < 10) & (data['Кво_комнат'] > 1))]
data = data.loc[~((data['Тип_жилья'] == 'квартира') & (data['Кво_комнат'] >= data['Площадь']))]
data = data.loc[~((data['Тип_жилья'] == 'дом') & (data['Кво_комнат'] >= data['Площадь']))]
data = data.loc[~((data['Тип_жилья'] == 'участок с землей') & (data['Площадь'] < 30))]

# Преобразование типов
data.loc[(data['Тип_жилья'] == 'дом') & (data['Площадь'] > 400), 'Тип_жилья'] = 'вилла'
data.loc[(data['Тип_жилья'] == 'вилла') & (data['Площадь'] < 400), 'Тип_жилья'] = 'дом'
data.loc[(data['Тип_жилья'] == 'дом') & (data['Площадь'] < 35), 'Тип_жилья'] = 'разное'
data.loc[(data['Тип_жилья'] == 'квартира') & (data['Площадь'] > 400), 'Тип_жилья'] = 'большая квартира'

# Обработка пропусков
mean_area_by_type = data.groupby('Тип_жилья')['Площадь'].median()
for tip, mean in mean_area_by_type.items():
    data.loc[(data['Тип_жилья'] == tip) & (data['Площадь'].isna()), 'Площадь'] = mean

data.loc[(data['Этаж'].isna()), 'Этаж'] = -1

data['Цена_лог'] = np.log1p(data['Цена'])

features = ['Тип_жилья', 'Индекс',
            'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
            'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
            'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
            'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
            'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']

# Обработка категориальных признаков
cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

# Преобразование категорий
for col in cat_features:
    data[col] = data[col].astype('category')

# Заполнение пропусков
for col in features:
    if col in cat_features:
        # Для категориальных признаков заполняем 'unknown'
        data[col] = data[col].cat.add_categories('unknown')  # Добавляем 'unknown' в категории
        data[col] = data[col].fillna('unknown')
    else:
        # Для числовых признаков заполняем медианой
        data[col] = data[col].fillna(data[col].median())

# Разделение данных
X = data[features]
y = data['Цена_лог']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1e-20, random_state=seed)

# Логарифмирование целевой переменной
data['Цена_лог'] = np.log1p(data['Цена'])

# Разделение данных (ИСПРАВЛЕНО test_size)
X = data[features]

y = data['Цена_лог']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.00000000002, random_state=seed)

# Создание датасета
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_features,  # Явное указание категорий
    free_raw_data=False
)

val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

# num_features = [col for col in features if col not in cat_features]
# scaler = RobustScaler()
# X_train[num_features] = scaler.fit_transform(X_train[num_features])
# X_val[num_features] = scaler.transform(X_val[num_features])
# joblib.dump(scaler, f'scaler_{seed}.pkl')

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
# # Лучшие параметры
# best_params = study.best_params
# print(best_params)

my_params = {
    # 'objective': 'huber',
    # 'metric': 'mape',
    # 'boosting_type': 'gbdt',
    # 'learning_rate': 0.0050303002211824184,
    # 'num_leaves': 1223,
    # 'max_depth': 12,
    # 'min_data_in_leaf': 86,
    # 'feature_fraction': 0.9990972023080763,
    # 'bagging_fraction': 0.7343383797011853,
    # 'bagging_freq': 8,
    # 'verbose': -1
    'objective': 'huber',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'learning_rate': 0.041964711162529034,
    'num_leaves': 628,
    'max_depth': 12,
    'min_data_in_leaf': 26,
    'feature_fraction': 0.8049426223737289,
    'bagging_fraction': 0.9713618894571485,
    'bagging_freq': 6,
    'verbose': -1,
}

# Параметры модели
params_light = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 10000,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Обучение
model = lgb.train(
    my_params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=1655,
    callbacks=[
        # lgb.early_stopping(stopping_rounds=100),  # Остановка, если нет улучшений 200 раундов
        lgb.log_evaluation(100)
    ]
)

# Сохранение модели
joblib.dump(model, f'lgb_model_{seed}.pkl')
