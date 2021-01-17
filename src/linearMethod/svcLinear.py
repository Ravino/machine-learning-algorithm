# Библиотека линейной алгебры
import numpy as np


from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split




# Загрузили данные файла формата csv где "," это разделитель
# Имеем матрицу 1372 строк на 5 столбцов 
dataset=np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", delimiter=",")


# Имеем матрицу значений x 1372 строк и 4 столбца для классификации по SVC
# Имеем массив значений y в 1372 элемента для классификации по SVC
dataX = dataset[:, :-1]
dataY = dataset[:, 4]


# Имеем tests/trains выборки с ответами, разделённые в соотношении 30/70
dataXTrain, dataXTest, dataYTrain, dataYTest = train_test_split(dataX, dataY, test_size=0.30, random_state=100)



# Определяем модель классификатора SVC с ядром linear
svcLinear = SVC(kernel="linear", C=0.31)


# Обучаем алгоритм SVC с ядром linear на тренеровачных данных
svcLinear.fit(dataXTrain, dataYTrain);


# получаем точность классификации алгоритмом SVC с ядром linear  на тестовых данных
accuracy = svcLinear.score(dataXTest, dataYTest)
print(accuracy)


"""
Текущая точность равна 0.9902912621359223
Покрутив параметр C, поднимать значение выше 0.31 не имеет смысла, точность не меняется до значения параметра 0.81
Значение параметра C меньше 0.51  или больше 0.75  приводит к снижению точности от 0.01 до 0.02 единиц
"""
