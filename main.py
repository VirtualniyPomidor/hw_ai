import random

import pandas as pd
import numpy as np
import os
import joblib
import warnings

import xgboost as xg
import catboost

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error as mape
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)

seed = random.randint(1, 100000)
random.seed(seed)

seed = 78157
# seed = 24919

print(f"seed: {seed}")

# Загружаем данные
data = pd.read_csv('train.csv')

# Преобразуем категориальный признак в числовой с помощью LabelEncoder
encoder = LabelEncoder()
data['Тип_жилья'] = encoder.fit_transform(data['Тип_жилья'])
data['Направление'] = encoder.fit_transform(data['Направление'])
data['Город'] = encoder.fit_transform(data['Город'])
data['Ктгр_энергоэффективности'] = encoder.fit_transform(data['Ктгр_энергоэффективности'])
data['Ктгр_вредных_выбросов'] = encoder.fit_transform(data['Ктгр_вредных_выбросов'])

data['Соотношение_этажей'] = data['Этаж'] / data['Верхний_этаж']
data['Эффективность_энергии'] = data['Ктгр_энергоэффективности'] / (data['Расход_тепла'] + 1)

# Логарифмирование числовых признаков с большим разбросом
data['Площадь_лог'] = np.log1p(data['Площадь'])

data.fillna(-1, inplace=True)

# Разделяем данные на признаки (X) и целевую переменную (y)
y = data['Цена'].copy()

X = data[
    ['Эффективность_энергии', 'Площадь_лог', 'Тип_жилья', 'Индекс',
     'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото',
     'Нлч_гаража', 'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
     'Ктгр_вредных_выбросов',
     'Размер_участка', 'Нлч_балкона', 'Ктгр_энергоэффективности', 'Направление', 'Кво_спален',
     'Кво_ванных', 'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']]

cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности', 'Ктгр_вредных_выбросов', 'Направление']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001, random_state=seed)

sample_weight = [1.0] * len(X_train)

sample_weight[4] = 10.0
sample_weight[11] = 5.0
sample_weight[12] = 5.0
sample_weight[6] = 10.0
sample_weight[6] = 10.0
sample_weight[8] = 10.0
sample_weight[13] = 5.0
sample_weight[18] = 5.0
sample_weight[2] = 10.0
sample_weight[24] = 5.0
sample_weight[25] = 5.0
# sample_weight[] = 5.0
# sample_weight[] = 5.0
# sample_weight[] = 5.0



# Масштабируем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Преобразуем в PyTorch тензоры
X_train_tensor = torch.Tensor(X_train_scaled)
y_train_tensor = torch.Tensor(y_train.values)

X_val_tensor = torch.Tensor(X_val_scaled)
y_val_tensor = torch.Tensor(y_val.values)

# Создаем TensorDataset и DataLoader для PyTorch (если будем использовать нейросети)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


class Models:
    KNeighborsRegressor_model = KNeighborsRegressor(
        n_neighbors=6
    )

    RandomForestRegressor_model = RandomForestRegressor(
        n_estimators=600,
        random_state=seed,
        max_depth=20,
        min_samples_split=10
    )

    SVR_model = SVR(
        kernel='rbf',
        C=10,
        gamma=0.1,
        epsilon=0.01
    )

    XGBRegressor_model = xg.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=3900, # 3900
        learning_rate=0.04,
        max_depth=7, # 7
        colsample_bytree=0.95,
        alpha=8,
        random_state=seed,
        min_child_weight=35,
    )
    CatBoostRegressor_model = catboost.CatBoostRegressor(
        iterations=6000,
        learning_rate=0.07,
        depth=8,
        l2_leaf_reg=3,
        cat_features=cat_features,
        eval_metric='R2',
        early_stopping_rounds=100,
        random_seed=seed,
        verbose=100,
        loss_function='MAE',  # 'MAE', 'RMSE', 'MAPE'
        grow_policy='Lossguide',
        max_leaves=64  # Максимальное число листьев при Lossguide
    )
    estimators = [
        ('xgb', xg.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=3900,
        learning_rate=0.04,
        max_depth=7,
        colsample_bytree=0.95,
        alpha=8,
        random_state=seed,
        min_child_weight=35)),
        ('catboost', catboost.CatBoostRegressor(
        iterations=2000,
        learning_rate=0.07,
        depth=8,
        l2_leaf_reg=3,
        # cat_features=cat_features,
        eval_metric='R2',
        early_stopping_rounds=100,
        random_seed=seed,
        verbose=100,
        loss_function='MAE',  # 'MAE', 'RMSE', 'MAPE'
        grow_policy='Lossguide',
        max_leaves=64))
    ]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())



models = [Models.XGBRegressor_model]

TestModels = pd.DataFrame(columns=['Model', 'R2', 'MSE', 'RMSE', 'MAPE'])

# Итерация по моделям
for model in models:
    model_name = str(model.__class__.__name__)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Вычисляем метрики
    r2_val = r2_score(y_val, model.predict(X_val))
    mse_val = mean_squared_error(y_val, model.predict(X_val))
    mape_val = mape(y_val, model.predict(X_val))

    # Добавляем результаты в DataFrame
    tmp = pd.DataFrame({'Model': [model_name], 'R2': [r2_val],
                        'MSE': [mse_val], 'RMSE': [np.sqrt(mse_val)],
                        'MAPE': [mape_val]})
    TestModels = pd.concat([TestModels, tmp], ignore_index=True)

    TestModels.set_index('Model', inplace=False)

    model_path = os.path.join('.', f"{model_name}.pkl")

    if model_name == "XGBRegressor":
        model.save_model(model_path.replace('.pkl', '.json'))  # Для XGB используем save_model
    else:
        joblib.dump(model, model_path)  # Сохранение модели с использованием joblib

    model_path = os.path.join('.', f"{model_name}.pkl")
    loaded_models = {}

    if model_name == "XGBRegressor":
        loaded_models[model_name] = xg.XGBRegressor()
        loaded_models[model_name].load_model(model_path.replace('.pkl', '.json'))  # Для XGB используем load_model
    else:
        loaded_models[model_name] = joblib.load(model_path)  # Загрузка модели с использованием joblib

print(TestModels)

# # Преобразуем категориальные признаки в тип category
# data['Тип_жилья'] = data['Тип_жилья'].astype("category")
# data['Направление'] = data['Направление'].astype("category")
# data['Город'] = data['Город'].astype("category")
# data['Ктгр_энергоэффективности'] = data['Ктгр_энергоэффективности'].astype("category")
# data['Ктгр_вредных_выбросов'] = data['Ктгр_вредных_выбросов'].astype("category")
#
# # Обрабатываем категориальные столбцы отдельно
# for col in ['Тип_жилья', 'Направление', 'Город', 'Ктгр_энергоэффективности', 'Ктгр_вредных_выбросов']:
#     # Добавляем новую категорию "-1" для обработки пропущенных значений
#     data[col] = data[col].cat.add_categories("-1")
#     # Заполняем пропущенные значения новой категорией "-1"
#     data[col] = data[col].fillna("-1")
#
# # Для числовых столбцов заполняем пропущенные значения -1
# numeric_cols = data.select_dtypes(exclude="category").columns  # Находим числовые столбцы
# data[numeric_cols] = data[numeric_cols].fillna(-1)
#
#
# # Разделяем данные на признаки (X) и целевую переменную (y)
# y = data['Цена'].copy()
# X = data[['Тип_жилья', 'Индекс', 'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото',
#           'Нлч_гаража', 'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
#           'Ктгр_вредных_выбросов',
#           'Размер_участка', 'Нлч_балкона', 'Ктгр_энергоэффективности', 'Направление', 'Кво_спален',
#           'Кво_ванных', 'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']]
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=seed)
#
# # Указываем индексы категориальных признаков
# cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности', 'Ктгр_вредных_выбросов', 'Направление']
#
# class Models:
#     KNeighborsRegressor_model = KNeighborsRegressor(
#         n_neighbors=6
#     )
#
#     RandomForestRegressor_model = RandomForestRegressor(
#         n_estimators=600,
#         random_state=seed,
#         max_depth=20,
#         min_samples_split=10
#     )
#
#     SVR_model = SVR(
#         kernel='rbf',
#         C=10,
#         gamma=0.1,
#         epsilon=0.01
#     )
#
#     XGBRegressor_model = xg.XGBRegressor(
#         objective='reg:absoluteerror',
#         n_estimators=8000,
#         learning_rate=0.05,
#         max_depth=7,
#         colsample_bytree=0.95,
#         alpha=8,
#         random_state=seed,
#         min_child_weight=35,
#
#     )
#
#     CatBoostRegressor_model = catboost.CatBoostRegressor(
#         iterations=2000,
#         learning_rate=0.07,
#         depth=8,
#         l2_leaf_reg=3,
#         cat_features=cat_features,  # Передаем список категориальных признаков
#         eval_metric='R2',
#         early_stopping_rounds=100,
#         random_seed=seed,
#         verbose=100,
#         loss_function='MAE',  # 'MAE', 'RMSE', 'MAPE'
#         grow_policy='Lossguide',
#         max_leaves=64  # Максимальное число листьев при Lossguide
#     )
#
#
# models = [Models.XGBRegressor_model]
#
# TestModels = pd.DataFrame(columns=['Model', 'R2', 'MSE', 'RMSE', 'MAPE'])
#
# # Итерация по моделям
# for model in models:
#     model_name = str(model.__class__.__name__)
#     if model_name == "CatBoostRegressor":
#         model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))  # Указываем eval_set для CatBoost
#     else:
#         model.fit(X_train, y_train)
#
#     # Вычисляем метрики
#     r2_val = r2_score(y_val, model.predict(X_val))
#     mse_val = mean_squared_error(y_val, model.predict(X_val))
#     mape_val = mape(y_val, model.predict(X_val))
#
#     # Добавляем результаты в DataFrame
#     tmp = pd.DataFrame({'Model': [model_name], 'R2': [r2_val],
#                         'MSE': [mse_val], 'RMSE': [np.sqrt(mse_val)],
#                         'MAPE': [mape_val]})
#     TestModels = pd.concat([TestModels, tmp], ignore_index=True)
#
#     TestModels.set_index('Model', inplace=False)
#
#     model_path = os.path.join('.', f"{model_name}.pkl")
#
#     if model_name == "XGBRegressor":
#         model.save_model(model_path.replace('.pkl', '.json'))  # Для XGB используем save_model
#     else:
#         joblib.dump(model, model_path)  # Сохранение модели с использованием joblib
#
#     model_path = os.path.join('.', f"{model_name}.pkl")
#     loaded_models = {}
#
#     if model_name == "XGBRegressor":
#         loaded_models[model_name] = xg.XGBRegressor()
#         loaded_models[model_name].load_model(model_path.replace('.pkl', '.json'))  # Для XGB используем load_model
#     else:
#         loaded_models[model_name] = joblib.load(model_path)  # Загрузка модели с использованием joblib
#
#     print(TestModels)
