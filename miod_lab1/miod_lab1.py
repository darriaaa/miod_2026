import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# 1.1
df = pd.read_csv('job_dataset.csv') # Завантажуємо набір даних у DataFrame без зміни стандартного індексу

print("--------------------перші 5 рядків--------------------")
print(df.head())

# 1.2 Очищення набору даних
print("--------------------наші 8 колонок--------------------")
df = df[['job_title', 'experience_years', 'education_level', 'skills_count', 'remote_work', 'salary', 'company_size', 'certifications']]
print(df)

# 1.3 Обробка пропущених значень
# Видаляємо всі рядки, де в колонці certifications є порожнє значення У НАС ТАКИХ НЕМА ТОМУ КОМЕНТИ
# print("--------------------після видалення пропущених значень--------------------")
df = df[df['certifications'].notna()]
#print(df.isna().sum())

# 1.4 Форматування даних
# Перетворюємо колонку remote_work у булевий тип: Yes -> True, No -> False, Hybrid -> True
print("--------------------форматування даних у remote_work--------------------")
df['remote_work'] = df['remote_work'].map({
    'Yes': True,
    'No': False,
    'Hybrid': True
})
print(df.dtypes)

# 2.1 Загальна кількість рядків
print("--------------------загальна кількість рядків--------------------")
print(len(df))      # кількість рядків
print(df.shape)     # розмірність DataFrame

# 2.2 Робота з числовими показниками

print("--------------------skills_count > 3--------------------")
top_skills = df[df['skills_count'] > 3]
print(top_skills)

print("--------------------кількість записів, де skills_count > 3--------------------")
print(len(top_skills))

print("--------------------середнє значення salary для вибірки--------------------")
print(top_skills['salary'].mean())

# Виводимо ідентифікатори 10 записів з найбільшим значенням salary
# Ідентифікаторами тут є стандартні індекси DataFrame
print("--------------------ідентифікатори 10 записів з найбільшим salary--------------------")
top10 = df.sort_values(by='salary', ascending=False).head(10)
print(top10.index.tolist())

# За бажанням можна ще подивитися самі ці записи
print("--------------------топ 10 записів за salary--------------------")
print(top10[['job_title', 'salary']])

# 2.3 Дослідження категорій та тексту
print("-------------------- Відфільтруйте дані за точним збігом у категоріальному стовпчику (education_level == PhD) та порахуйте кількість таких записів--------------------")
education_filter = df[df['education_level'] == 'PhD']
print(education_filter)
print(len(education_filter))

print("--------------------Відфільтруйте дані за частковим збігом (наявність підрядка) у іншому категоріальному стовпчику та запишіть у нову змінну--------------------")
job_filter = df[df['job_title'].str.startswith('AI')]
print(job_filter)

print("-------------------- кількість записів, які одночасно PhD і AI --------------------")
both_filter = df[
    (df['education_level'] == 'PhD') &
    (df['job_title'].str.startswith('AI'))
]
print(both_filter)
print("Кількість записів:", len(both_filter))

print("-------------------- Частка записів, які PhD, але не AI --------------------")
only_education = df[
    (df['education_level'] == 'PhD') &
    (~df['job_title'].str.startswith('AI'))
]
share_only_education = len(only_education) / len(df)
print("Кількість записів:", len(only_education))
print("Частка:", share_only_education)

print("-------------------- Кількість записів, які не PhD і не AI --------------------")
neither_filter = df[
    (df['education_level'] != 'PhD') &
    (~df['job_title'].str.startswith('AI'))
]
print(neither_filter)
print("Кількість записів:", len(neither_filter))

# 2.4 Дослідження числових діапазонів та зрізів даних
print("-------------------- experience=7 --------------------")
experience_7 = df[df['experience_years'] == 7]
print(experience_7)

print("-------------------- діапазон значень experience=7-10 --------------------")
print(len(df[(df['experience_years'] >= 7) & (df['experience_years'] <= 10)]))

print("-------------------- кого більше з досвідом 7 років чи 10 --------------------")
experience_7_10 = df[(7 >= df['experience_years']) & (df['experience_years'] <= 10)]
print(len(experience_7) > len(experience_7_10))

# 2.5 Комбіновані фільтри та оцінки
print("-------------------- зп більша 250к --------------------")
best_salary = df[df['salary'] > 250000]
print(best_salary)

print("-------------------- зп більша 250к і найбільший експіріенс --------------------")
print(best_salary.sort_values('experience_years', ascending=False).head())
# best_salary.nlargest(5, 'experience_years') - альтернатива

print("-------------------- середнє та медіана experience_years для топ-10 за salary --------------------")
# top10 - це ідентифікатори 10 записів з найбільшим salary
print("Середнє experience_years:", top10['experience_years'].mean())
print("Медіана experience_years:", top10['experience_years'].median())

# 2.6 Порівняння груп

print("-------------------- створили 2 нових датасета з job_title: df_ai_engin i df_cloud_engin--------------------")
df_ai_engin = df[df['job_title'] == 'AI Engineer']
print(df_ai_engin)
df_cloud_engin = df[df['job_title'] == 'Cloud Engineer']
print(df_cloud_engin)

# Створіть новий DataFrame, який базується на цих двох змінних. Він повинен мати стовпчик category_name із назвою категорії, та стовпчик total_records з підрахованою кількістю записів для кожної групи
print("-------------------- новий DataFrame, на AI Engineer', 'Cloud Engineer. category_name - назва категорії, стовпчик total_records - підрахована кількість записів для кожної групи--------------------")
summary_data = pd.DataFrame({
    'category_name': ['AI Engineer', 'Cloud Engineer'],
    'total_records': [len(df_ai_engin), len(df_cloud_engin)]
})
print(summary_data)

print("-------------------- додали average_experiance, де середнє значення experience_years --------------------")
# зробили змінні з середніми значеннями в категорії
avg_ai = df_ai_engin['experience_years'].mean()
avg_cloud = df_cloud_engin['experience_years'].mean()

# додали average_experiance
summary_data['average_experiance'] = [avg_ai, avg_cloud]
print(summary_data)

# 2.7 Комплексні завдання
# 2.6 Складний фільтр з AND, OR, NOT
print("-------------------- складний фільтр: salary > 200000  + education_level == 'PhD' + remote_work == True + не Small комп--------------------")

complex_filter = df[
    (df['salary'] > 200000) &
    (
        (df['education_level'] == 'PhD') |
        (df['remote_work'] == True)
    ) &
    ~(df['company_size'] == 'Small')
]

print(complex_filter)
print("Кількість записів за складним фільтром:", len(complex_filter))

print("-------------------- графік salary від experience --------------------")
df.plot(x='experience_years', y='salary', kind='scatter', alpha=0.3)
plt.show()