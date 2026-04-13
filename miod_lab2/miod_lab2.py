import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 1.1 Завантаження набору даних
df = pd.read_csv('job_dataset.csv')

# Виводимо перші 5 рядків, щоб перевірити, що файл зчитався правильно
print("-------------------- Перші 5 рядків --------------------")
print(df.head())

# Обробка пропусків
#print("-------------------- Перевірка пропусків --------------------")
#print(df.isnull().sum())

# Дивимось, який тип має кожен стовпчик
#print("-------------------- Типи даних до змін --------------------")
#print(df.dtypes)

# Форматування даних
print("-------------------- Форматування даних у remote_work --------------------")
df['remote_work'] = df['remote_work'].map({
    'Yes': True,
    'No': False,
    'Hybrid': True
})
print("-------------------- Дані після зміни remote_work --------------------")
print(df)

# Підготовка даних саме для графіка. Залишаємо тільки 5 найпоширеніших посад.
top_jobs = df['job_title'].value_counts().head(5).index
# Створюємо окремий DataFrame лише для побудови графіка. У ньому будуть тільки ті рядки, де job_title входить у топ-5 найчастіших посад
df_plot = df[df['job_title'].isin(top_jobs)]

# ЗАВДАННЯ 2. ВІЗУАЛЬНИЙ АНАЛІЗ ДАНИХ
print("-------------------- Розподіл salary з розбиттям за посадами job_title --------------------")

plt.figure(figsize=(10, 6)) # Задаємо розмір вікна графіка

# Будуємо гістограму:
# data=df_plot       -> використовуємо відфільтрований набір даних
# x='salary'         -> по осі X відкладаємо зарплату
# hue='job_title'    -> різні кольори для різних посад
# kde=True           -> додаємо лінію щільності розподілу - показує що розподіл норм
# bins=25            -> кількість інтервалів гістограми
sns.histplot(
    data=df_plot,
    x='salary',
    hue='job_title',
    kde=True,
    bins=25
)

# Додаємо вертикальну лінію середнього значення зарплати
plt.axvline(
    df_plot['salary'].mean(),
    color='red',
    linestyle='--',
    label=f"Mean: {df_plot['salary'].mean():.1f}"
)

plt.title('Розподіл зарплатні для 5 найпоширеніших посад')
plt.xlabel('Зарплатня')
plt.ylabel('Кількість записів у данному діапазоні')
plt.legend()
plt.show()


print("-------------------- Середня зарплата по посадах з розбиттям за remote_work --------------------")
# Групування даних: обчислюємо середнє значення заробітної плати (salary) для кожної комбінації посади (job_title) та типу роботи (remote_work)
avg_salary = df_plot.groupby(['job_title', 'remote_work'])['salary'].mean().reset_index()

plt.figure(figsize=(12, 6)) # Створення області для побудови графіка

palette = {False: 'blue', True: 'orange'} # кольори для категорій

# Побудова стовпчикової діаграми:
sns.barplot(
    data=avg_salary,
    x='job_title',
    y='salary',
    hue='remote_work', # hue — додатковий вимір (тип роботи)
    palette=palette # щоб відображались саме мої кольори
)

plt.title('Середня зарплата по посадах з розбиттям за типом роботи')
plt.xlabel('Посада')
plt.ylabel('Середня зарплата')

plt.xticks(rotation=20) # Поворот підписів осі X

# Налаштування легенди:
handles, _ = plt.gca().get_legend_handles_labels() # отримуємо поточні графічні елементи (handles)
plt.legend(
    handles=handles,
    labels=['False (офіс)', 'True (віддалено/гібрид)'],
    title='Віддалена робота'
)

# Додавання сітки для кращого візуального сприйняття
plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.5
)
plt.grid(
    axis='x',
    linestyle='--',
    alpha=0.3
)
plt.show()

print("-------------------- Аналіз розкиду та викидів. Розподіл зарплати за посадами --------------------")
plt.figure(figsize=(12, 6))

# Будуємо boxplot
sns.boxplot(
    data=df_plot,
    x='job_title',
    y='salary',
    palette='pastel'
)

# Заголовок і підписи осей
plt.title('Розподіл зарплати за посадами')
plt.xlabel('Посада')
plt.ylabel('Зарплата')

plt.xticks(rotation=20)

# Додаємо горизонтальні лінії сітки для зручнішого читання значень
plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.5
)
plt.show()

print("-------------------- 2.4 Частка типів роботи --------------------")

# Рахуємо кількість кожної категорії
remote_counts = df['remote_work'].value_counts()

# Перетворюємо True/False у зрозумілий текст
labels = ['Віддалено/гібрид' if x else 'Офіс' for x in remote_counts.index]

plt.figure(figsize=(6, 6))

# Побудова кругової діаграми
plt.pie(
    remote_counts,
    labels=labels,
    autopct='%1.1f%%',   # показує відсотки
    startangle=140,      # поворот діаграми
    colors=['orange', 'blue'],  # кольори як у попередньому графіку
    explode=(0.05, 0)    # трохи "висуваємо" один сектор
)

plt.title('Розподіл типів роботи в наборі даних')
plt.show()


print("-------------------- 2.5 Залежність зарплати від досвіду та освіти --------------------")
plt.figure(figsize=(10, 6))

# Побудова точкової діаграми:
sns.scatterplot(
    data=df_plot,
    x='experience_years',
    y='salary',
    hue='education_level',
    alpha=0.5  # прозорість точок 
)

# Додавання лінії тренду: показує загальну тенденцію залежності між змінними
sns.regplot(
    data=df_plot,
    x='experience_years',
    y='salary',
    scatter=False,      # не малюємо точки повторно
    color='red',        # колір лінії тренду
    line_kws={'linewidth': 2}
)

# Оформлення графіка
plt.xlabel('Досвід роботи (роки)')
plt.ylabel('Зарплата')
plt.title('Залежність зарплати від досвіду та рівня освіти')
plt.legend(title='Рівень освіти')
# Додаємо сітку для кращого читання значень
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("-------------------- 2.6 Динаміка або тренд. Зміна середньої зарплати залежно від рівня освіти --------------------")

# Задаємо логічний порядок рівнів освіти
education_order = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']

# Групуємо дані за education_level і рахуємо середню зарплату
trend_data = df.groupby('education_level')['salary'].mean().reindex(education_order)

# Побудова лінійного графіка
plt.figure(figsize=(10, 6))

plt.plot(
    trend_data.index,     # вісь X — рівні освіти у заданому порядку
    trend_data.values,    # вісь Y — середня зарплата
    marker='o',           # позначки на точках
    linestyle='-',        # тип лінії
    linewidth=2,          # товщина лінії
    color='darkblue'      # колір лінії
)

# Оформлення графіка
plt.title('Зміна середньої зарплати залежно від рівня освіти')
plt.xlabel('Рівень освіти')
plt.ylabel('Середня зарплата')

# Додаємо сітку 
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("-------------------- Матриця кореляцій --------------------")

# Обираємо тільки числові колонки
numeric_cols = df[['experience_years', 'skills_count', 'salary', 'certifications']]

# Рахуємо матрицю кореляцій
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(8, 6))

# Будуємо теплову карту
sns.heatmap(
    corr_matrix,
    annot=True,        # показує числа всередині клітинок
    cmap='coolwarm',   # кольорова схема (синій → червоний)
    fmt='.2f',         # формат чисел (2 знаки після коми)
    linewidths=0.5     # тонкі лінії між клітинками
)

plt.title('Матриця кореляцій числових змінних')
plt.show()

print("-------------------- 2.8 Комбінований графік --------------------")

# Крок 1. Створюємо полотно для 4 графіків 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Крок 2. Побудова гістограми розподілу зарплат - відображає, як розподілені значення salary у наборі даних
sns.histplot(
    data=df,
    x='salary',
    bins=30,          # кількість інтервалів
    kde=True,         # додає криву щільності розподілу
    color='skyblue',
    ax=axes[0, 0]     # розміщення графіка (рядок 0, колонка 0)
)
axes[0, 0].set_title('Розподіл зарплат')
axes[0, 0].set_xlabel('Зарплата')
axes[0, 0].set_ylabel('Кількість записів')

# Крок 3. Побудова точкового графіка (scatter plot) - показує залежність зарплати від досвіду роботи
sns.scatterplot(
    data=df,
    x='experience_years',
    y='salary',
    hue='education_level',  # розбиття за категорією
    alpha=0.4,              # прозорість точок
    ax=axes[0, 1]
)
axes[0, 1].set_title('Залежність зарплати від досвіду')
axes[0, 1].set_xlabel('Досвід (роки)')
axes[0, 1].set_ylabel('Зарплата')

# Крок 4. Відбір даних для топ-5 найпоширеніших посад повтор)
top_jobs = df['job_title'].value_counts().head(5).index
df_plot = df[df['job_title'].isin(top_jobs)]

# Крок 5. Побудова boxplot (ящик з вусами)
# Відображає розкид значень зарплати для кожної посади
sns.boxplot(
    data=df_plot,
    x='job_title',
    y='salary',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Розкид зарплат по посадах')
axes[1, 0].set_xlabel('Посада')
axes[1, 0].set_ylabel('Зарплата')
# Повертаємо підписи для зручності читання
axes[1, 0].tick_params(axis='x', rotation=20)

# Крок 6. Групування даних для обчислення середньої зарплати - для кожної посади рахуємо середнє значення salary
avg_salary = df_plot.groupby('job_title')['salary'].mean().reset_index()

# Крок 7. Побудова стовпчикової діаграми - показує середню зарплату для кожної посади
sns.barplot(
    data=avg_salary,
    x='job_title',
    y='salary',
    color='orange',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Середня зарплата по посадах')
axes[1, 1].set_xlabel('Посада')
axes[1, 1].set_ylabel('Середня зарплата')
axes[1, 1].tick_params(axis='x', rotation=20)

# Крок 8. Автоматичне вирівнювання графіків
plt.tight_layout()
plt.show()