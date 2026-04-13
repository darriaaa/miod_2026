# pandas для роботи з таблицями та CSV-файлами
import pandas as pd
# numpy для роботи з числовими обчисленнями
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Імпортуємо функцію для поділу даних на тренувальну та тестову вибірки
from sklearn.model_selection import train_test_split

# Імпортуємо моделі лінійної регресії:
# LinearRegression — звичайна лінійна регресія
# Ridge — гребенева регресія
from sklearn.linear_model import LinearRegression, Ridge

# Імпортуємо PowerTransformer для перетворення ознак до більш нормального розподілу
from sklearn.preprocessing import PowerTransformer

# Імпортуємо потрібні метрики якості моделі
from sklearn.metrics import (
    mean_absolute_error,       # середня абсолютна помилка
    mean_squared_log_error,    # середньоквадратична логарифмічна помилка
    d2_absolute_error_score    # D² absolute error score
)

# ЗАВДАННЯ 1. ПІДГОТОВКА НАБОРУ ДАНИХ

# Зчитуємо файл sales.csv у DataFrame з назвою df
df = pd.read_csv('sales.csv')
print("\n-------------------- 1.1 Перші 5 рядків --------------------")
print(df.head())

print("Розмір датасету:")
print(df.shape)

print("\n-------------------- Типи даних у початковому наборі --------------------")
print(df.dtypes)

# 1.2 Базова підготовка даних

print("\n-------------------- Форматування order_date --------------------")
# Перетворюємо текстовий стовпчик order_date у формат datetime
df['order_date'] = pd.to_datetime(df['order_date'])
print(df.dtypes) # типи даних після перетворення order_date

print("\n-------------------- Вибір числових колонок --------------------")

df_num = df[['quantity', 'unit_price', 'discount', 'revenue', 'cost', 'profit']].copy() # новий DataFrame тільки з числовими стовпчиками
print(df_num.head()) # перші 5 рядків нового датасету

print("\n-------------------- Перевірка пропущених значень --------------------")
print(df_num.isnull().sum()) # Для кожного стовпчика рахуємо кількість порожніх значень

# Проходимо циклом по всіх числових колонках
for col in df_num.columns:
    # Якщо в поточному стовпчику є хоча б один пропуск
    if df_num[col].isnull().sum() > 0:
        # Заповнюємо пропуски медіаною цього стовпчика
        df_num[col] = df_num[col].fillna(df_num[col].median())

print("\n-------------------- Пропуски після обробки --------------------")
print(df_num.isnull().sum()) # Ще раз перевіряємо, що пропусків більше немає

print("\n-------------------- Фільтрація аномалій --------------------")
# Робимо копію числового датасету, щоб очищати саме її
df_clean = df_num.copy()

# Починаємо цикл по всіх числових колонках
for col in ['quantity', 'unit_price', 'discount', 'revenue', 'cost', 'profit']:

    # Обчислюємо перший квартиль (25%)
    q1 = df_clean[col].quantile(0.25)

    # Обчислюємо третій квартиль (75%)
    q3 = df_clean[col].quantile(0.75)

    # Обчислюємо міжквартильний розмах
    iqr = q3 - q1

    # Нижня межа для виявлення викидів
    lower_bound = q1 - 1.5 * iqr

    # Верхня межа для виявлення викидів
    upper_bound = q3 + 1.5 * iqr

    # Залишаємо тільки ті рядки, де значення входять у допустимий діапазон
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
# Виводимо розмір датасету після очищення
print(df_clean.shape)
# Виводимо перші 5 рядків очищених даних
print(df_clean.head())


# 1.3 Знаходження залежностей

print("\n-------------------- Кореляційна матриця --------------------")

# Обчислюємо кореляційну матрицю для всіх числових змінних через пандас
corr_matrix = df_clean.corr()
print(corr_matrix)

plt.figure(figsize=(8, 6)) # Створюємо область для побудови теплової карти

# Будуємо теплову карту кореляцій
sns.heatmap(
    corr_matrix,       # сама матриця кореляцій
    annot=True,        # показує значення всередині клітинок
    cmap='coolwarm',   # палітра кольорів
    fmt='.2f',         # формат чисел: 2 знаки після коми
    linewidths=0.5     # товщина меж між клітинками
)
plt.title('Матриця кореляцій')
plt.show()


print("\n-------------------- Сила зв’язку ознак з profit --------------------")

# Обираємо цільову змінну
target_col = 'profit'

# Беремо стовпчик кореляцій тільки для profit
# drop('profit') видаляє кореляцію ознаки самої з собою
# sort_values(key=abs, ascending=False) сортує за абсолютною величиною зв’язку
target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
print(target_corr)

strong_corr_threshold = 0.85 # Встановлюємо поріг сильної кореляції

# Виводимо цей поріг
print("\nПоріг сильної кореляції:", strong_corr_threshold)

# Обираємо фінальні ознаки для моделі
# quantity, unit_price, discount, revenue — це будуть наші X
feature_cols = ['quantity', 'unit_price', 'discount', 'revenue']

print("\n-------------------- Фінальні ознаки для моделі --------------------")
print(feature_cols)# Виводимо список ознак
X = df_clean[feature_cols].copy() # Формуємо DataFrame X тільки з вибраними ознаками
y = df_clean[target_col].copy() # Формуємо цільову змінну y зі стовпчика profit


print("\n-------------------- Поділ на train / test --------------------")

# Ділимо дані на тренувальну та тестову вибірки
# test_size=0.2 означає 20% даних на тест
# random_state=42 потрібен для відтворюваності результату
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # random_state=42 - фіксує випадковість.
)

# Виводимо розміри тренувальної та тестової вибірок
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Визначаємо ознаку, яка найбільше пов’язана з profit
most_corr_feature = target_corr.index[0]

# Виводимо її назву
print("Найбільш пов’язана ознака з profit:")
print(most_corr_feature)

# Створюємо допоміжну функцію, яка буде повертати таблицю метрик для моделі
def get_metrics_dataframe(model_name, model_obj, X_train_used, X_test_used, y_train_true, y_test_true, y_train_pred, y_test_pred):
    # model_name - Назва моделі у вигляді тексту. model_obj - Сама модель
    # X_train_used, X_test_used - ознаки, які були використані в моделі
    # y_train_true, y_test_true - Справжні значення profit для train і test
    # y_train_pred, y_test_pred - Прогнози моделі для train і test

    # Для MSLE значення не повинні бути від’ємними, тому прогнози знизу обрізаємо нулем
    y_train_pred_nonneg = np.maximum(y_train_pred, 0)
    y_test_pred_nonneg = np.maximum(y_test_pred, 0)

    # Створюємо DataFrame з метриками для train і test. У стовпчику model буде 2 рядки: для train і test. 
    metrics_df = pd.DataFrame({
        'model': [model_name, model_name],
        'dataset': ['train', 'test'],

        # score_R2 — вбудована метрика .score() для регресії. 
        # Для train: наскільки добре модель пояснює тренувальні данні
        # Для test: наскільки добре модель працює на нових даних
        'score_R2': [
            model_obj.score(X_train_used, y_train_true),
            model_obj.score(X_test_used, y_test_true)
        ],

        # MAE — середня абсолютна помилка
        'MAE': [
            mean_absolute_error(y_train_true, y_train_pred),
            mean_absolute_error(y_test_true, y_test_pred)
        ],

        # MSLE — середньоквадратична логарифмічна помилка
        'MSLE': [
            mean_squared_log_error(y_train_true, y_train_pred_nonneg),
            mean_squared_log_error(y_test_true, y_test_pred_nonneg)
        ],

        # D2 absolute error score
        'D2_absolute_error': [
            d2_absolute_error_score(y_train_true, y_train_pred),
            d2_absolute_error_score(y_test_true, y_test_pred)
        ]
    })
    return metrics_df  # Повертаємо таблицю метрик


# ЗАВДАННЯ 2. ПОБУДОВА РЕГРЕСІЙНИХ МОДЕЛЕЙ


print("-------------------- 2.1 Проста лінійна регресія --------------------")

# Створюємо об’єкт моделі LinearRegression
linear_model = LinearRegression()

# Навчаємо модель на тренувальній вибірці Модель дивиться: які значення мають ознаки, як вони пов’язані з profit, і намагається знайти формулу, яка найкраще описує цей зв’язок.
linear_model.fit(X_train, y_train)

# Отримуємо прогноз для train
y_train_pred_lin = linear_model.predict(X_train)

# Отримуємо прогноз для test
y_test_pred_lin = linear_model.predict(X_test)

# Рахуємо метрики для простої лінійної регресії
metrics_linear = get_metrics_dataframe(
    'LinearRegression',
    linear_model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_pred_lin,
    y_test_pred_lin
)


print("\n-------------------- Метрики простої лінійної регресії --------------------")
print(metrics_linear)

# Будуємо візуалізацію для train і test в одній системі координат
plt.figure(figsize=(10, 6))

# Реальні значення train
plt.scatter(
    X_train[most_corr_feature],
    y_train,
    color='blue',
    alpha=0.35,
    label='Train actual'
)

# Прогноз train
plt.scatter(
    X_train[most_corr_feature],
    y_train_pred_lin,
    color='navy',
    alpha=0.35,
    marker='x',
    label='Train predicted'
)

# Реальні значення test
plt.scatter(
    X_test[most_corr_feature],
    y_test,
    color='green',
    alpha=0.35,
    label='Test actual'
)

# Прогноз test
plt.scatter(
    X_test[most_corr_feature],
    y_test_pred_lin,
    color='red',
    alpha=0.35,
    marker='x',
    label='Test predicted'
)

# Підписуємо графік
plt.title('Проста лінійна регресія: train/test та прогнози')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("-------------------- 2.2 Проста лінійна регресія з перетворенням змінних --------------------")

# Створюємо об’єкт PowerTransformer
power_transformer_lin = PowerTransformer()

# Навчаємо трансформер на тренувальних ознаках і одразу перетворюємо їх
X_train_pt = power_transformer_lin.fit_transform(X_train)

# Перетворюємо тестові ознаки тим самим трансформером
X_test_pt = power_transformer_lin.transform(X_test)

# Створюємо нову модель LinearRegression
linear_model_pt = LinearRegression()

# Навчаємо модель на перетворених ознаках
linear_model_pt.fit(X_train_pt, y_train)

# Прогноз на train
y_train_pred_lin_pt = linear_model_pt.predict(X_train_pt)

# Прогноз на test
y_test_pred_lin_pt = linear_model_pt.predict(X_test_pt)

# Рахуємо метрики
metrics_linear_pt = get_metrics_dataframe(
    'LinearRegression + PowerTransformer',
    linear_model_pt,
    X_train_pt,
    X_test_pt,
    y_train,
    y_test,
    y_train_pred_lin_pt,
    y_test_pred_lin_pt
)

# Виводимо метрики
print("\n-------------------- Метрики лінійної регресії після перетворення --------------------")
print(metrics_linear_pt)

# Будуємо графік
plt.figure(figsize=(10, 6))

plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual')
plt.scatter(X_train[most_corr_feature], y_train_pred_lin_pt, color='navy', alpha=0.35, marker='x', label='Train predicted')
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual')
plt.scatter(X_test[most_corr_feature], y_test_pred_lin_pt, color='red', alpha=0.35, marker='x', label='Test predicted')

plt.title('Лінійна регресія після перетворення змінних')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("-------------------- 2.3 Гребенева регресія -------------------- ")

# Створюємо порожній список для збереження таблиць метрик Ridge
ridge_metrics_list = []

# Створюємо Ridge з параметром за замовчуванням alpha=1.0
ridge_default = Ridge()
# Навчаємо модель
ridge_default.fit(X_train, y_train)
# Робимо прогнози
y_train_pred_ridge_default = ridge_default.predict(X_train)
y_test_pred_ridge_default = ridge_default.predict(X_test)
# Рахуємо метрики
metrics_ridge_default = get_metrics_dataframe(
    'Ridge alpha=1.0',
    ridge_default,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_pred_ridge_default,
    y_test_pred_ridge_default
)
# Додаємо таблицю у список
ridge_metrics_list.append(metrics_ridge_default)

# Створюємо Ridge з alpha=0.1
ridge_01 = Ridge(alpha=0.1)
# Навчаємо модель
ridge_01.fit(X_train, y_train)
# Робимо прогнози
y_train_pred_ridge_01 = ridge_01.predict(X_train)
y_test_pred_ridge_01 = ridge_01.predict(X_test)
# Рахуємо метрики
metrics_ridge_01 = get_metrics_dataframe(
    'Ridge alpha=0.1',
    ridge_01,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_pred_ridge_01,
    y_test_pred_ridge_01
)
# Додаємо таблицю у список
ridge_metrics_list.append(metrics_ridge_01)

# Створюємо Ridge з alpha=10.0
ridge_10 = Ridge(alpha=10.0)
# Навчаємо модель
ridge_10.fit(X_train, y_train)
# Робимо прогнози
y_train_pred_ridge_10 = ridge_10.predict(X_train)
y_test_pred_ridge_10 = ridge_10.predict(X_test)
# Рахуємо метрики
metrics_ridge_10 = get_metrics_dataframe(
    'Ridge alpha=10.0',
    ridge_10,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_pred_ridge_10,
    y_test_pred_ridge_10
)
# Додаємо таблицю у список
ridge_metrics_list.append(metrics_ridge_10)

# Об’єднуємо всі Ridge-метрики в один DataFrame
metrics_ridge_all = pd.concat(ridge_metrics_list, ignore_index=True)
print("\n-------------------- Метрики гребеневої регресії --------------------")
print(metrics_ridge_all)

# Візуалізація для alpha=1.0
plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual') # hеальні значення train. по осі X — найбільш пов’язана ознака (revenue), по осі Y — реальний profit
plt.scatter(X_train[most_corr_feature], y_train_pred_ridge_default, color='navy', alpha=0.35, marker='x', label='Train predicted') # показує прогнози моделі для train
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual') # показує реальні знач test
plt.scatter(X_test[most_corr_feature], y_test_pred_ridge_default, color='red', alpha=0.35, marker='x', label='Test predicted') # показує прогнози для test
plt.title('Ridge alpha=1.0')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Візуалізація для alpha=0.1
plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual')
plt.scatter(X_train[most_corr_feature], y_train_pred_ridge_01, color='navy', alpha=0.35, marker='x', label='Train predicted')
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual')
plt.scatter(X_test[most_corr_feature], y_test_pred_ridge_01, color='red', alpha=0.35, marker='x', label='Test predicted')
plt.title('Ridge alpha=0.1')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Візуалізація для alpha=10.0
plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual')
plt.scatter(X_train[most_corr_feature], y_train_pred_ridge_10, color='navy', alpha=0.35, marker='x', label='Train predicted')
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual')
plt.scatter(X_test[most_corr_feature], y_test_pred_ridge_10, color='red', alpha=0.35, marker='x', label='Test predicted')
plt.title('Ridge alpha=10.0')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("-------------------- 2.4 Гребенева регресія з перетворенням змінних--------------------")
# Створюємо окремий PowerTransformer для Ridge - створює об’єкт трансформації (робить розподіл даних ближчим до нормального
power_transformer_ridge = PowerTransformer()

# Перетворюємо train/ fit - вивчає данні/ transform - змінює їх
X_train_pt_ridge = power_transformer_ridge.fit_transform(X_train)

# Перетворюємо test
X_test_pt_ridge = power_transformer_ridge.transform(X_test)

# Створюємо список для метрик
ridge_pt_metrics_list = []

# Ridge + PT alpha=1.0
ridge_pt_default = Ridge()
ridge_pt_default.fit(X_train_pt_ridge, y_train)  # навчання
y_train_pred_ridge_pt_default = ridge_pt_default.predict(X_train_pt_ridge) # прогноз на тренування
y_test_pred_ridge_pt_default = ridge_pt_default.predict(X_test_pt_ridge) # прогноз на тест
# рахує: R²/MAE/MSLE/D2
metrics_ridge_pt_default = get_metrics_dataframe(
    'Ridge + PT alpha=1.0',
    ridge_pt_default,
    X_train_pt_ridge,
    X_test_pt_ridge,
    y_train,
    y_test,
    y_train_pred_ridge_pt_default,
    y_test_pred_ridge_pt_default
)
ridge_pt_metrics_list.append(metrics_ridge_pt_default)

# Ridge + PT alpha=0.1
ridge_pt_01 = Ridge(alpha=0.1)
ridge_pt_01.fit(X_train_pt_ridge, y_train)
y_train_pred_ridge_pt_01 = ridge_pt_01.predict(X_train_pt_ridge)
y_test_pred_ridge_pt_01 = ridge_pt_01.predict(X_test_pt_ridge)

metrics_ridge_pt_01 = get_metrics_dataframe(
    'Ridge + PT alpha=0.1',
    ridge_pt_01,
    X_train_pt_ridge,
    X_test_pt_ridge,
    y_train,
    y_test,
    y_train_pred_ridge_pt_01,
    y_test_pred_ridge_pt_01
)
ridge_pt_metrics_list.append(metrics_ridge_pt_01)

# Ridge + PT alpha=10.0
ridge_pt_10 = Ridge(alpha=10.0)
ridge_pt_10.fit(X_train_pt_ridge, y_train)
y_train_pred_ridge_pt_10 = ridge_pt_10.predict(X_train_pt_ridge)
y_test_pred_ridge_pt_10 = ridge_pt_10.predict(X_test_pt_ridge)

metrics_ridge_pt_10 = get_metrics_dataframe(
    'Ridge + PT alpha=10.0',
    ridge_pt_10,
    X_train_pt_ridge,
    X_test_pt_ridge,
    y_train,
    y_test,
    y_train_pred_ridge_pt_10,
    y_test_pred_ridge_pt_10
)
ridge_pt_metrics_list.append(metrics_ridge_pt_10)

# Об’єднуємо всі метрики
metrics_ridge_pt_all = pd.concat(ridge_pt_metrics_list, ignore_index=True)

# Виводимо таблицю
print("\n-------------------- Метрики гребеневої регресії після перетворення --------------------")
print(metrics_ridge_pt_all)

# Візуалізація alpha=1.0
plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual')
plt.scatter(X_train[most_corr_feature], y_train_pred_ridge_pt_default, color='navy', alpha=0.35, marker='x', label='Train predicted')
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual')
plt.scatter(X_test[most_corr_feature], y_test_pred_ridge_pt_default, color='red', alpha=0.35, marker='x', label='Test predicted')
plt.title('Ridge + PowerTransformer alpha=1.0')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Візуалізація alpha=0.1
plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual')
plt.scatter(X_train[most_corr_feature], y_train_pred_ridge_pt_01, color='navy', alpha=0.35, marker='x', label='Train predicted')
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual')
plt.scatter(X_test[most_corr_feature], y_test_pred_ridge_pt_01, color='red', alpha=0.35, marker='x', label='Test predicted')
plt.title('Ridge + PowerTransformer alpha=0.1')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Візуалізація alpha=10.0
plt.figure(figsize=(10, 6))
plt.scatter(X_train[most_corr_feature], y_train, color='blue', alpha=0.35, label='Train actual')
plt.scatter(X_train[most_corr_feature], y_train_pred_ridge_pt_10, color='navy', alpha=0.35, marker='x', label='Train predicted')
plt.scatter(X_test[most_corr_feature], y_test, color='green', alpha=0.35, label='Test actual')
plt.scatter(X_test[most_corr_feature], y_test_pred_ridge_pt_10, color='red', alpha=0.35, marker='x', label='Test predicted')
plt.title('Ridge + PowerTransformer alpha=10.0')
plt.xlabel(most_corr_feature)
plt.ylabel('profit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



print("-------------------- 2.5 Сукупне порівняння метрик--------------------")
# На цьому етапі у нас уже є 4 готові таблиці з попередніх пунктів:
# 1) metrics_linear       -> результати простої лінійної регресії
# 2) metrics_linear_pt    -> результати лінійної регресії після PowerTransformer
# 3) metrics_ridge_all    -> результати трьох моделей Ridge
# 4) metrics_ridge_pt_all -> результати трьох моделей Ridge + PowerTransformer

# Кожна з цих таблиць містить метрики для train і test.
# Наше завдання тут:
# - зібрати всі ці результати в одну спільну таблицю
# - за потреби зробити мультиіндекс
# - розфарбувати клітинки кольорами
#   (червоний = менше, зелений = більше)


# КРОК 1. Об'єднуємо всі таблиці метрик в один DataFrame
# ------------------------------------------------------------
# pd.concat() склеює кілька DataFrame один під одним
# ignore_index=True означає, що старі індекси буде відкинуто
# і створено новий суцільний індекс 0, 1, 2, 3, ...
all_metrics = pd.concat(
    [
        metrics_linear,
        metrics_linear_pt,
        metrics_ridge_all,
        metrics_ridge_pt_all
    ],
    ignore_index=True
)
print("\n-------------------- Загальна таблиця метрик --------------------")
print(all_metrics)


# КРОК 2. Створюємо таблицю з мультиіндексом
# ------------------------------------------------------------
# set_index(['model', 'dataset']) робить індекс з двох колонок:
# - model   -> назва моделі
# - dataset -> train або test
# Такий вигляд таблиці зручний для аналізу,
# тому що результати одразу групуються за моделями
comparison_table = all_metrics.set_index(['model', 'dataset'])
print("\n-------------------- Таблиця метрик з мультиіндексом --------------------")
print(comparison_table)


# КРОК 3. Створюємо кольорову стилізовану таблицю
# ------------------------------------------------------------
# .style дозволяє застосувати стилізацію до DataFrame
# .background_gradient() зафарбовує клітинки кольоровим градієнтом
# cmap='RdYlGn' означає палітру:
# - Red   -> менше
# - Yellow-> середні значення
# - Green -> більше
# За умовою : червоний = менше / зелений = більше
styled_table = (
    comparison_table.style

    # Форматуємо всі числа до 6 знаків після коми,
    # щоб таблиця виглядала акуратно
    .format("{:.6f}")

    # Додаємо кольорову шкалу до всіх числових стовпчиків
    # Чим більше значення, тим "зеленіша" клітинка
    # Чим менше значення, тим "червоніша" клітинка
    .background_gradient(cmap='RdYlGn')
)


# КРОК 4. Зберігаємо кольорову таблицю у HTML-файл
# ------------------------------------------------------------
# to_html() зберігає стилізовану таблицю як HTML-сторінку
# Це зручно, тому що:
# - у консолі кольори не відображаються
# - в браузері таблиця буде кольорова і наочна
styled_table.to_html('metrics_comparison.html')

# Повідомляємо користувачу, що файл успішно створено
print("\nКольорова таблиця метрик збережена у файл: metrics_comparison.html")

# КРОК 5. Зберегти звичайну таблицю у CSV
# ------------------------------------------------------------
comparison_table.to_csv('metrics_comparison.csv')
print("Звичайна таблиця метрик також збережена у файл: metrics_comparison.csv")