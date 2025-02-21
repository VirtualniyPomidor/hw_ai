import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_percentage_error as mape
import joblib

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
        # Для категориальных признаков заполняем 'unknown'
        data[col] = data[col].cat.add_categories('unknown')  # Добавляем 'unknown' в категории
        data[col] = data[col].fillna('unknown')
    else:
        # Для числовых признаков заполняем медианой
        data[col] = data[col].fillna(data[col].median())

# Разделение данных
X = data[features]
y = data['Цена_лог']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.002, random_state=seed)

# Создание датасета
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Параметры модели
params_light = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 310,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Обучение
model = lgb.train(params_light, train_data, valid_sets=[val_data], num_boost_round=1000)

# Сохранение модели
joblib.dump(model, f'lgb_model_{seed}.pkl')