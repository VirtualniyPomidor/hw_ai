import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
import joblib
import optuna
from optuna.pruners import MedianPruner
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Создание датасетов для XGBoost
train_data_xgb = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
val_data_xgb = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

# # Функция для подбора гиперпараметров через Optuna
# def objective(trial):
#     params = {
#         'objective': 'reg:absoluteerror',  # Используем MAE для регрессии
#         'eval_metric': 'mape',  # Метрика для оценки
#         'boosting_type': 'gbtree',
#         'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
#         'max_depth': trial.suggest_int('max_depth', 4, 14),
#         'min_child_weight': trial.suggest_int('min_child_weight', 20, 70),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),  # L1-регуляризация
#         'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),  # L2-регуляризация
#         'seed': seed,
#         'verbosity': 0
#     }
#
#     model = xgb.train(params, train_data_xgb, num_boost_round=1000)
#     preds = model.predict(val_data_xgb)
#     return mape(y_val, preds)
#
# # Оптимизация гиперпараметров
# study = optuna.create_study(direction='minimize', pruner=MedianPruner())
# study.optimize(objective, n_trials=100)
#
# # Лучшие параметры
# best_params = study.best_params
# print(f"Лучшие параметры: {best_params}")

my_params = {
    'objective': 'reg:absoluteerror',  # Используем MAE для регрессии
    'eval_metric': 'mape',  # Метрика для оценки
    'boosting_type': 'gbtree',
    # 'n_estimators': 3900,
    'learning_rate': 0.04883828743738249,
    'max_depth': 14,
    'min_child_weight': 27,
    'subsample': 0.9026852802508798,
    'colsample_bytree': 0.7249666172295455,
    'alpha': 6.143381686034998e-06,
    'lambda': 6.196512774354873e-07
}
# Обучение модели с лучшими параметрами
final_model = xgb.train(my_params, train_data_xgb, num_boost_round=1000)

# Сохранение модели
joblib.dump(final_model, f'xgb_model_{seed}.pkl')