# Случайное число с использованием модуля random
import random

rand_list = []
n = 10
for i in range(n):
    rand_list.append(random.randint(3, 9))
print(rand_list)

# использование random.sample()
# Python3 code to demonstrate
# to generate random number list
# using random.sample()
import random

# using random.sample()
# to generate random number list
res = random.sample(range(1, 50), 7)

# printing result
print("Random number list is : " + str(res))

# использование понимания списков + randrange()
# Python3 code to demonstrate
# to generate random number list
# using list comprehension + randrange()
import random

# using list comprehension + randrange()
# to generate random number list
res = [random.randrange(1, 50, 1) for i in range(7)]

# printing result
print("Random number list is : " + str(res))

# использование цикла + randint()
# Method 3: For Loop Random Int List [0, 51]
import random

lis = []
for _ in range(10):
    lis.append(random.randint(0, 51))
print(lis)

# Способ 1: Создание списка случайных целых чисел с помощью функции numpy.random.randint
# importing numpy module
import numpy as np

# print the list of 10 integers from 3 to 7
print(list(np.random.randint(low=3, high=8, size=10)))

# print the list of 5 integers from 0 to 2
# if high parameter is not passed during
# function call then results are from [0, low)
print(list(np.random.randint(low=3, size=5)))

# Способ 2. Создание списка случайных чисел с плавающей точкой с помощью функции numpy.random.random_sample
import numpy as np

# generates list of 4 float values
print(np.random.random_sample(size=4))

# generates 2d list of 4*4
print(np.random.random_sample(size=(4, 4)))

# 1 метод КУБИК

import random
import matplotlib.pyplot as plt


def kubik(n: int) -> list:
    """

    :param n: Количество подбрасываний
    :return:  Список слкучайных подюрасываний кубика
    """
    data = []
    while len(data) < n:
        data.append(random.randint(1, 6))
    return data


print(kubik(100))
print(kubik(1000))
print(kubik(10000))
print(kubik(1000000))


# 2 метод
def count_rate(kub_data: list):
    """
    Возвращает частоту выпадания значений кубика,
    согласно полученным данным
    :param kub_data: данные эксперимента
    :return:
    """
    kub_rate = {}
    for i in kub_data:
        if i in kub_rate:
            continue
        else:
            kub_rate[i] = kub_data.count(i)
    for i in range(1, 7):
        if i not in kub_rate:
            kub_rate[i] = 0
    return kub_rate

print(count_rate([100, 1000, 10000, 1000000]))




# 3 метод
def crate_dataframe(sorted_date: dict):
    """
    Создание и преобразование данных в Pandas dataframe
    :param sorted_date: dict
    :return: pd.Dataframe
    """
    df = pd.DataFrame(sorted_date, index=[0])
    df = df.T
    df = df.rename(columns={0: 'Частота'})
    df.insert(0, 'Количество выпаданий', range(1, 1 + len(df)))
    return df

f = {100, 1000, 10000, 1000000}


def crate_dataframe(f):
    pass


print(crate_dataframe(f))




# 4 метод
import pandas as pd
def probability_solving(dataframe: pd.DataFrame):
    """
    Вычисление вероятности полученных результатов
    :param dataframe:
    :return:
    """
    sum_rate = dataframe['Частота'].sum()
    probability = []
    for i in dataframe['Частота']:
        probability.append(i / sum_rate)
    dataframe['Вероятность'] = probability
    return dataframe

data_values = [100, 1000, 10000, 1000000]
dataframe = pd.DataFrame({'Частота': data_values})

result = probability_solving(dataframe)
print(result)

# Гистограмма 1 метод

import random
import matplotlib.pyplot as plt


def kubik(n: int) -> list:
    """
    :param n: Количество подбрасываний
    :return: Список случайных подбрасываний кубика
    """
    data = []
    while len(data) < n:
        data.append(random.randint(1, 6))
    return data


def plot_histogram(n):
    data = kubik(n)  # Генерация данных
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=np.arange(1, 8) - 0.5, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Гистограмма подбрасываний кубика для n = {n}')
    plt.xlabel('Значения (грань кубика)')
    plt.ylabel('Частота')
    plt.xticks(range(1, 7))  # Подписываем ось X
    plt.grid(axis='y')
    plt.show()


# Значения n
n_values = [100, 1000, 10000, 1000000]

# Построение гистограмм
for n in n_values:
    plot_histogram(n)

# Гистограмма 2 метод
import numpy as np
import matplotlib.pyplot as plt


def count_rate(kub_data: np.ndarray):
    # Use np.unique to count occurrences and get the corresponding values
    values, counts = np.unique(kub_data, return_counts=True)
    kub_rate = dict(zip(values, counts))

    # Ensure all values from 1 to 6 are represented
    for i in range(1, 7):
        if i not in kub_rate:
            kub_rate[i] = 0

    return kub_rate


# Параметры для тестирования
n_values = [100, 1000, 10000, 1000000]
fig, axs = plt.subplots(len(n_values), 1, figsize=(10, 20))

for idx, n in enumerate(n_values):
    # Генерация данных кубика
    kub_data = np.random.randint(1, 7, n)

    # Подсчет частоты
    kub_rate = count_rate(kub_data)

    # Подготовка данных для гистограммы
    values = list(kub_rate.keys())
    frequencies = list(kub_rate.values())

    # Построение гистограммы
    axs[idx].bar(values, frequencies, color='skyblue')
    axs[idx].set_title(f'Гистограмма для n = {n}')
    axs[idx].set_xlabel('Значения кубика')
    axs[idx].set_ylabel('Частота')
    axs[idx].set_xticks(values)

plt.tight_layout()
plt.show()

# Гистограмма 3 метод
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_rate(kub_data: np.ndarray):
    # Use np.unique to count occurrences and get the corresponding values
    values, counts = np.unique(kub_data, return_counts=True)
    kub_rate = dict(zip(values, counts))

    # Ensure all values from 1 to 6 are represented
    for i in range(1, 7):
        if i not in kub_rate:
            kub_rate[i] = 0

    return kub_rate


def crate_dataframe(sorted_data: dict):
    df = pd.DataFrame(sorted_data, index=[0])
    df = df.T
    df = df.rename(columns={0: 'Частота'})
    df.insert(0, 'Количество выпаданий', range(1, 1 + len(df)))
    return df


# Параметры для тестирования
n_values = [100, 1000, 10000, 1000000]
fig, axs = plt.subplots(len(n_values), 1, figsize=(10, 20))

for idx, n in enumerate(n_values):
    # Генерация данных кубика
    kub_data = np.random.randint(1, 7, n)

    # Подсчет частоты
    kub_rate = count_rate(kub_data)

    # Преобразование в DataFrame
    df = crate_dataframe(kub_rate)

    # Подготовка данных для гистограммы
    values = df['Количество выпаданий']
    frequencies = df['Частота']

    # Построение гистограммы
    axs[idx].bar(values, frequencies, color='skyblue')
    axs[idx].set_title(f'Гистограмма для n = {n}')
    axs[idx].set_xlabel('Количество выпаданий')
    axs[idx].set_ylabel('Частота')
    axs[idx].set_xticks(values)

plt.tight_layout()
plt.show()

# 4 метод
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_rate(kub_data: np.ndarray):
    # Use np.unique to count occurrences and get the corresponding values
    values, counts = np.unique(kub_data, return_counts=True)
    kub_rate = dict(zip(values, counts))

    # Ensure all values from 1 to 6 are represented
    for i in range(1, 7):
        if i not in kub_rate:
            kub_rate[i] = 0

    return kub_rate


def crate_dataframe(sorted_data: dict):
    df = pd.DataFrame(sorted_data, index=[0])
    df = df.T
    df = df.rename(columns={0: 'Частота'})
    df.insert(0, 'Количество выпаданий', range(1, 1 + len(df)))
    return df


def probability_solving(dataframe: pd.DataFrame):
    sum_rate = dataframe['Частота'].sum()
    probability = [i / sum_rate for i in dataframe['Частота']]
    dataframe['Вероятность'] = probability
    return dataframe


# Параметры для тестирования
n_values = [100, 1000, 10000, 1000000]
fig, axs = plt.subplots(len(n_values), 1, figsize=(10, 20))

for idx, n in enumerate(n_values):
    # Генерация данных кубика
    kub_data = np.random.randint(1, 7, n)

    # Подсчет частоты
    kub_rate = count_rate(kub_data)

    # Преобразование в DataFrame
    df = crate_dataframe(kub_rate)

    # Вычисление вероятностей
    df = probability_solving(df)

    # Подготовка данных для гистограммы
    values = df['Количество выпаданий']
    probabilities = df['Вероятность']

    # Построение гистограммы вероятностей
    axs[idx].bar(values, probabilities, color='skyblue')
    axs[idx].set_title(f'Гистограмма вероятностей для n = {n}')
    axs[idx].set_xlabel('Количество выпаданий')
    axs[idx].set_ylabel('Вероятность')
    axs[idx].set_xticks(values)

plt.tight_layout()
plt.show()
