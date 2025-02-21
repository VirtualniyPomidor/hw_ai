import random
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
import joblib
import warnings
import keras

# Фиксация сидов
# seed = random.randint(1, 100000)
seed = 80978
np.random.seed(seed)
tf.random.set_seed(seed)
print(f"Seed: {seed}")

# Загрузка данных
data = pd.read_csv('train.csv')

# Обработка категориальных признаков
cat_features = ['Тип_жилья', 'Город', 'Ктгр_энергоэффективности',
                'Ктгр_вредных_выбросов', 'Направление']

encoder = LabelEncoder()
for col in cat_features:
    data[col] = encoder.fit_transform(data[col])

# Инженерия признаков
data['Соотношение_этажей'] = data['Этаж'] / (data['Верхний_этаж'] + 1e-6)
data['Площадь_лог'] = np.log1p(data['Площадь'])
data['Цена_лог'] = np.log1p(data['Цена'])  # Логарифмирование целевой переменной

# Заполнение пропусков
data.fillna(-1, inplace=True)

# Выбор признаков
features = [
    'Площадь_лог', 'Тип_жилья', 'Индекс',
    'Площадь', 'Расход_тепла', 'Кво_комнат', 'Кво_фото', 'Нлч_гаража',
    'Нлч_кондиционера', 'Верхний_этаж', 'Город', 'Этаж', 'Кво_вредных_выбросов',
    'Ктгр_вредных_выбросов', 'Размер_участка', 'Нлч_балкона',
    'Ктгр_энергоэффективности', 'Направление', 'Кво_спален', 'Кво_ванных',
    'Нлч_парковки', 'Нлч_террасы', 'Нлч_подвала', 'Широта', 'Долгота'
]

X = data[features]
y = data['Цена_лог']

# Нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed)

# Архитектура нейросети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='mape',
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name='MAPE')]
)

# Коллбэки
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_MAPE', patience=1000, restore_best_weights=True)

# Обучение
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=4000,
    batch_size=128,
    callbacks=[early_stop],
    verbose=100
)

# Сохранение модели
model.save(f'nn_model_{seed}.keras')
joblib.dump(scaler, f'scaler_{seed}.pkl')