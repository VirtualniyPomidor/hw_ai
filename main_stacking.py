import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Фиксация сидов
seed = 80978
np.random.seed(seed)
print(f"Seed: {seed}")

# Загрузка данных
data = pd.read_csv('train.csv')

# Логарифмирование целевой переменной
data['Цена_лог'] = np.log1p(data['Цена'])

features = [
    'Тип_жилья', 'Индекс',
    'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
    'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
    'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
    'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
    'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота'
]

# Обработка категориальных признаков
cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

# Преобразование категорий
for col in cat_features:
    data[col] = data[col].astype('category')

# Заполнение пропусков
for col in features:
    if col in cat_features:
        data[col] = data[col].cat.add_categories('unknown')
        data[col] = data[col].fillna('unknown')
    else:
        data[col] = data[col].fillna(data[col].median())

# Разделение данных
X = data[features]
y = data['Цена_лог']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.000002, random_state=seed)

# Создание датасетов для LightGBM и XGBoost
train_data_lgb = lgb.Dataset(X_train, label=y_train)
val_data_lgb = lgb.Dataset(X_val, label=y_val)

train_data_xgb = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
val_data_xgb = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

# Параметры для LightGBM
params_lgb = {
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

# Параметры для XGBoost
params_xgb = {
    'objective': 'reg:absoluteerror',
    'n_estimators': 3900,  # 3900
    'learning_rate': 0.04,
    'max_depth': 7,  # 7
    'colsample_bytree': 0.95,
    'alpha': 8,
    'random_state': seed,
    'min_child_weight': 35
}

# Обучение LightGBM
model_lgb = lgb.train(params_lgb, train_data_lgb, valid_sets=[val_data_lgb], num_boost_round=1650)

# Обучение XGBoost
model_xgb = xgb.train(params_xgb, train_data_xgb, num_boost_round=10000)

# Получение предсказаний на тренировочных данных
train_preds_lgb = model_lgb.predict(X_train)
train_preds_xgb = model_xgb.predict(train_data_xgb)

# Получение предсказаний на валидационных данных
val_preds_lgb = model_lgb.predict(X_val)
val_preds_xgb = model_xgb.predict(val_data_xgb)

# Создание новых признаков для мета-модели
X_train_stacked = np.column_stack((train_preds_lgb, train_preds_xgb))
X_val_stacked = np.column_stack((val_preds_lgb, val_preds_xgb))

# Обучение мета-модели (линейная регрессия)
meta_model = LinearRegression()
meta_model.fit(X_train_stacked, y_train)

# Предсказания мета-модели
val_preds_stacked = meta_model.predict(X_val_stacked)

# Оценка качества
print(f"MAPE стекинга: {mape(y_val, val_preds_stacked)}")

# Сохранение моделей
joblib.dump(model_lgb, f'lgb_model_{seed}.pkl')
joblib.dump(model_xgb, f'xgb_model_{seed}.pkl')
joblib.dump(meta_model, f'meta_model_{seed}.pkl')