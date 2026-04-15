import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1. Загрузка данных
file_path = r"Automobile.csv"
df = pd.read_csv(file_path)

# Удалим пропуски (важно!)
df = df.dropna()


# 2. Признаки и целевая переменная
X = df[['displacement']]   # вместо ENGINESIZE
Y = df[['mpg']]            # вместо CO2EMISSIONS


# 3. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# 4. Полиномиальные признаки
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)


# 5. Обучение модели
model = LinearRegression()
model.fit(X_train_poly, y_train)


# 6. Прогноз
X_test_poly = poly.transform(X_test)
Y_pred = model.predict(X_test_poly)

print(f"R2 Score: {r2_score(y_test, Y_pred):.2f}")


# 7. Визуализация
plt.scatter(X_train, y_train, color='blue', label='Обучающие данные')

# Сортировка для красивой линии
sorted_idx = X_test['displacement'].values.argsort()

plt.plot(
    X_test['displacement'].values[sorted_idx],
    Y_pred.flatten()[sorted_idx],
    color='red',
    linewidth=2,
    label='Полиномиальная регрессия (степень = 3)'
)

plt.xlabel('Объем двигателя (displacement)')
plt.ylabel('Расход топлива (mpg)')
plt.legend()
plt.show()