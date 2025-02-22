import pandas as pd
import numpy as np
import joblib
from main_stacking import seed  # Импортируем seed из main_light.py
from datetime import datetime
import os
import xgboost as xgb

# seed = 80978

# Загрузка данных
test_data = pd.read_csv('public_test.csv')

# Список признаков (должен совпадать с main_light.py)
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
    test_data[col] = test_data[col].astype('category')

# Заполнение пропусков
for col in features:
    if col in cat_features:
        test_data[col] = test_data[col].cat.add_categories('unknown')
        test_data[col] = test_data[col].fillna('unknown')
    else:
        test_data[col] = test_data[col].fillna(test_data[col].median())

# Загрузка моделей
model_lgb = joblib.load(f'lgb_model_{seed}.pkl')
model_xgb = joblib.load(f'xgb_model_{seed}.pkl')
meta_model = joblib.load(f'meta_model_{seed}.pkl')

# Получение предсказаний базовых моделей
X_test = test_data[features]
test_preds_lgb = model_lgb.predict(X_test)
test_preds_xgb = model_xgb.predict(xgb.DMatrix(X_test, enable_categorical=True))

# Создание новых признаков для мета-модели
X_test_stacked = np.column_stack((test_preds_lgb, test_preds_xgb))

# Получение финальных предсказаний
final_preds = meta_model.predict(X_test_stacked)

# Обратное преобразование целевой переменной (если использовалось логарифмирование)
final_preds = np.expm1(final_preds)

# Сохранение результатов
name = f'public_test_predict_seed_stacking_{seed}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
path = 'tests'
os.makedirs(path, exist_ok=True)
full_path = os.path.join(path, name)

pd.DataFrame({'id': test_data.id, 'Цена': final_preds}).to_csv(full_path, index=False)

print(f"Файл сохранен: {full_path}")