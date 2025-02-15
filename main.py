import random

import pandas as pd
import numpy as np
import os
import joblib
import warnings

import xgboost as xg

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error as mape
# from catboost import CatBoostRegressor

import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)

seed = random.randint(1, 100000)
random.seed(seed)

seed = 228

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

data.fillna(-1, inplace=True)

# Разделяем данные на признаки (X) и целевую переменную (y)
y = data['Цена'].copy()
X = data[['Тип_жилья', 'Индекс', 'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото',
          'Нлч_гаража', 'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
          'Ктгр_вредных_выбросов',
          'Размер_участка', 'Нлч_балкона', 'Ктгр_энергоэффективности', 'Направление', 'Кво_спален',
          'Кво_ванных', 'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=seed)

# Масштабируем данные
scaler = MaxAbsScaler()
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

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# models = [LinearRegression(),
#           KNeighborsRegressor(n_neighbors=6),
#           RandomForestRegressor(n_estimators=600, random_state=seed),
#           SVR(kernel='rbf'),
#           xg.XGBRegressor(objective='reg:absoluteerror', n_estimators=600, random_state=seed),
#           SGDRegressor(tol=1e-4)]

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
        n_estimators=900,
        learning_rate=0.05,
        max_depth=7,
        colsample_bytree=0.95,
        alpha=8,
        random_state=seed,
        min_child_weight = 35,
    )
    # CatBoostRegressor_model = CatBoostRegressor(
    #     iterations=500,
    #     learning_rate=0.05,
    #     depth=16,
    #     l2_leaf_reg=3,
    #     random_seed=seed,
    #     verbose=False
    #     )


models = [Models.XGBRegressor_model]

TestModels = pd.DataFrame(columns=['Model', 'R2', 'MSE', 'RMSE', 'MAPE'])

# Итерация по моделям
for model in models:
    model_name = str(model.__class__.__name__)
    model.fit(X_train, y_train)

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

