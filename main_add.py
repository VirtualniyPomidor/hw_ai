import random
import pandas as pd
import numpy as np
import os
import joblib
import warnings

import xgboost as xg
import catboost

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

warnings.simplefilter(action='ignore', category=FutureWarning)

# Фиксация сида
seed = random.randint(1, 100000)

# seed = 46899

np.random.seed(seed)
print(f"seed: {seed}")

# Загрузка данных
data = pd.read_csv('train.csv')

# Обработка категориальных признаков
cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
               'Ктгр_вредных_выбросов', 'Направление']

encoder = LabelEncoder()
for col in cat_features:
    data[col] = encoder.fit_transform(data[col])

# Feature engineering
data['Соотношение_этажей'] = data['Этаж'] / (data['Верхний_этаж'] + 1e-6)
data['Эффективность_энергии'] = data['Ктгр_энергоэффективности'] / (data['Расход_тепла'] + 1)
data['Площадь_лог'] = np.log1p(data['Площадь'])
data['Цена_лог'] = np.log1p(data['Цена'])  # Логарифмирование целевой переменной

# # Повторить те же преобразования
# data['Ктгр_энергоэффективности'] = pd.to_numeric(data['Ктгр_энергоэффективности'], errors='coerce')
# data['Расход_тепла'] = pd.to_numeric(data['Расход_тепла'], errors='coerce')
#
# data['Ктгр_энергоэффективности'] = data['Ктгр_энергоэффективности'].fillna(-1)
# data['Расход_тепла'] = data['Расход_тепла'].fillna(-1)
#
# data['Эффективность_энергии'] = data['Ктгр_энергоэффективности'] / (data['Расход_тепла'] + 1)

# Заполнение пропусков
data.fillna(-1, inplace=True)

# Выбор признаков
features = [
    'Эффективность_энергии', 'Площадь_лог', 'Тип_жилья', 'Индекс',
    'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
    'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
    'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
    'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
    'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота'
]

X = data[features]
y = data['Цена_лог']  # Используем логарифмированную целевую переменную

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Автоматическое определение весов для выбросов
q95 = y_train.quantile(0.95)
sample_weight = np.where(y_train >= q95, 5.0, 1.0)

# Модели
xgb_model = xg.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=3900,  # 3900
    learning_rate=0.04,
    max_depth=7,  # 7
    colsample_bytree=0.95,
    alpha=8,
    random_state=seed,
    min_child_weight=35
)

cat_model = catboost.CatBoostRegressor(
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

# Стекинг моделей
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model)
    ],
    final_estimator=Ridge(alpha=100)
)

# Обучение
print("Training XGBoost...")
xgb_model.fit(X_train, y_train, sample_weight=sample_weight)

print("\nTraining CatBoost...")
cat_model.fit(X_train, y_train, cat_features=cat_features,
             eval_set=(X_val, y_val), verbose=100)

print("\nTraining Stacking Model...")
stack_model.fit(X_train, y_train)

# Предсказание и оценка
def evaluate(model, X, y):
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)  # Возвращаем исходный масштаб
    return mape(np.expm1(y), preds)

print("\nValidation Scores:")
print(f"XGBoost MAPE: {evaluate(xgb_model, X_val, y_val):.4f}")
print(f"CatBoost MAPE: {evaluate(cat_model, X_val, y_val):.4f}")
print(f"Stacking MAPE: {evaluate(stack_model, X_val, y_val):.4f}")

# Сохранение моделей
joblib.dump(xgb_model, f'xgb_{seed}.pkl')
joblib.dump(cat_model, f'cat_{seed}.pkl')
joblib.dump(stack_model, f'stack_{seed}.pkl')