import pandas as pd
import numpy as np
import joblib
from main_xgb import seed
from datetime import datetime
import os

# seed = 80978

# Загрузка данных
test_data = pd.read_csv('public_test.csv')

# Список признаков (должен совпадать с main_light.py)
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
    test_data[col] = test_data[col].astype('category')

# Заполнение пропусков
for col in features:
    if col in cat_features:
        # Для категориальных признаков заполняем 'unknown'
        test_data[col] = test_data[col].cat.add_categories('unknown')  # Добавляем 'unknown' в категории
        test_data[col] = test_data[col].fillna('unknown')
    else:
        # Для числовых признаков заполняем медианой
        test_data[col] = test_data[col].fillna(test_data[col].median())

# Загрузка модели
model = joblib.load(f'xgb_model_{seed}.pkl')

# Предсказание
X_test = test_data[features]
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)  # Обратное преобразование, если использовалось логарифмирование

# Сохранение
name = f'public_test_predict_seed_xgb_{seed}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
path = 'tests'
os.makedirs(path, exist_ok=True)
full_path = os.path.join(path, name)

pd.DataFrame({'id': test_data.id, 'Цена': preds}).to_csv(full_path, index=False)

print(f"Файл сохранен: {full_path}")