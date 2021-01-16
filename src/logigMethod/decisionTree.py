# библиотека линейной алгебры
import numpy as np


# Модель классификатора DecisionTree
# функция разделения данных на тестовые и контрольные
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




# Загрузили данные файла формата csv где "," это разделитель
# Имеем матрицу 1372 строк на 5 столбцов 
dataset=np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", delimiter=",")


# Имеем матрицу значений x 1372 строк и 4 столбца для классификации по DecisionTree
# Имеем массив значений y в 1372 элемента для классификации по DecisionTree
dataX = dataset[:, :-1]
dataY = dataset[:, 4]


# Имеем tests/trains выборки с ответами, разделённые в соотношении 30/70
dataXTrain, dataXTest, dataYTrain, dataYTest = train_test_split(dataX, dataY, test_size=0.30, random_state=100)



# Определяем модель классификатора DecisionTree
# Критерий информативности энтропийный
# Критерий информативности gini
decisionTreeEntropy = DecisionTreeClassifier(criterion="entropy", max_depth=7, random_state=0)
decisionTreeGini = DecisionTreeClassifier(criterion="gini", max_depth=7, random_state=0)


# Обучаем алгоритм DecisionTree с критерием entropy на тестовых данных
# Обучаем алгоритм DecisionTree с критерием gini на тестовых данных
decisionTreeEntropy.fit(dataXTrain, dataYTrain)
decisionTreeGini.fit(dataXTrain, dataYTrain)


# получаем точность классификации алгоритмом DecisionTree с критерием entropy на контрольных данных
# получаем точность классификации алгоритмом DecisionTree с критерием Gini на контрольных данных
accuracyEntropy = decisionTreeEntropy.score(dataXTest, dataYTest)
accuracyGini = decisionTreeGini.score(dataXTest, dataYTest)


print(accuracyEntropy)
print(accuracyGini)


"""
Крутил разные параметры. 
На текущий момент точность для DecisionTreeEntropy равна 0.9951456310679612
для decisionTreeGini точность равна 0.9927184466019418
На точность влияет тасование выборки параметром random_state, если ставить меньше или больше 100, точность падает от 0.01 и выше.
Ставить параметру max_depth больше 7 для DecisionTreeEntropy и decisionTreeGini, не имеет смысла, точность не поднимается, значение меньше 7, приводит к снижению точности.
Изменения значения параметра random_state для decisionTreeEntropy и decisionTreeGini на точность не влияло

"""
