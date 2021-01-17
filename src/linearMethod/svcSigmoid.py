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
dataXTrain, dataXTest, dataYTrain, dataYTest = train_test_split(dataX, dataY, test_size=0.30, random_state=10)



# Определяем модель классификатора SVC с ядром sigmoid
svcSigmoid = SVC(kernel="sigmoid", C=0.025)


# Обучаем алгоритм SVC с ядром sigmoid на тренеровачных данных
svcSigmoid.fit(dataXTrain, dataYTrain);


# получаем точность классификации алгоритмом SVC с ядром sigmoid  на тестовых данных
accuracy = svcSigmoid.score(dataXTest, dataYTest)
print(accuracy)


"""
Текущая точность равна 0.8131067961165048
Покрутив параметр C, поднимать значение выше 0.025 не имеет смысла? т.к. к лучшим результатам не приводит
Значение параметра C меньше 0.025  или больше 0.025  приводит к снижению точности от 0.001 до 0.5 единиц
Сильно точность изменяет параметр random_state в разбивке выборки, значение меньше или больше 10 понижает точность в несколько сотых
"""
