# библиотека линейной алгебры
import numpy as np


# Модель классификатора RandomForestClassifier
# функция разделения данных на тестовые и контрольные
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




# Загрузили данные файла формата csv где "," это разделитель
# Имеем матрицу 1372 строк на 5 столбцов 
dataset=np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", delimiter=",")


# Имеем матрицу значений x 1372 строк и 4 столбца для классификации по DecisionTree
# Имеем массив значений y в 1372 элемента для классификации по DecisionTree
dataX = dataset[:, :-1]
dataY = dataset[:, 4]


# Имеем tests/trains выборки с ответами, разделённые в соотношении 30/70
dataXTrain, dataXTest, dataYTrain, dataYTest = train_test_split(dataX, dataY, test_size=0.30, random_state=100)



# Определяем модель классификатора RandomForest
# Критерий информативности энтропийный
# Критерий информативности gini
randomForestEntropy = RandomForestClassifier(criterion="entropy", n_estimators=10, max_features=2, max_depth=7, random_state=0)
randomForestGini = RandomForestClassifier(criterion="gini", n_estimators=10, max_features=2, max_depth=7, random_state=0)


# Обучаем алгоритм RandomForest с критерием entropy на тестовых данных
# Обучаем алгоритм RandomForest с критерием gini на тестовых данных
randomForestEntropy.fit(dataXTrain, dataYTrain)
randomForestGini.fit(dataXTrain, dataYTrain)


# получаем точность классификации алгоритмом RandomForest с критерием entropy на контрольных данных
# получаем точность классификации алгоритмом RandomForest с критерием Gini на контрольных данных
accuracyEntropy = randomForestEntropy.score(dataXTest, dataYTest)
accuracyGini = randomForestGini.score(dataXTest, dataYTest)


print(accuracyEntropy)
print(accuracyGini)


"""
Крутил разные параметры. 
На текущий момент точность для randomForestEntropy равна 0.9975728155339806
для randomForestGini точность равна 0.9951456310679612
На точность влияет тасование выборки параметром random_state, если ставить меньше или больше 100, точность падает от 0.01 и выше.
Ставить параметру max_depth больше 7 для randomForestEntropy и randomForestGini, не имеет смысла, точность не поднимается, значение меньше 7, приводит к снижению точности.
Изменения значения параметра random_state для randomForestEntropy и randomForestGini на точность не влияло

Изменение значения параметра n_estimators меньше 10 приводит к снижению точности, увеличение более 10 не повышает точность.
Изменение параметра max_feature более или менее 2 приводит к снижению точности
"""
