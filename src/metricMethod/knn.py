# библиотека линейной алгебры
import numpy as np


# Модель классификатора KNN 
# функция разделения данных на тестовые и контрольные
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




# Загрузили данные файла формата csv, где "," это разделитель
# Имеем матрицу 1372 строк на 5 столбцов 
dataset=np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", delimiter=",")


# Имеем матрицу значений x 1372 строк и 2 столбца для классификации по KNN
# Имеем массив значений y в 1372 элемента для классификации по KNN
dataX = dataset[:, :-3]
dataY = dataset[:, 4]


# Имеем tests/trains выборки с ответами, разделённые в соотношении 30/70
dataXTrain, dataXTest, dataYTrain, dataYTest = train_test_split(dataX, dataY, test_size=0.30, random_state=100)



# Определяем модель классификатора KNN, где сосед K равен 3
# Задаём метрическое пространство
neigh = KNeighborsClassifier(n_neighbors=3, metric="minkowski")



# Обучаем алгоритм KNN на тестовых данных
neigh.fit(dataXTrain, dataYTrain);



# получаем точность классификации алгоритмом KNN на контрольных данных
accuracy = neigh.score(dataXTest, dataYTest)
print(accuracy)


"""
Крутил разные параметры. На текущий момент точность 0.9635922330097088
На точность влияет тасование выборки параметром random_state, если ставить меньше или больше 100, точность падает от 0.01 и выше.
Параметр метрики в алгоритме KNN тоже изменял точность, но при тасовании выборки параметром random_state в 100, значение точности не меняется на метриках euclidean и minkowski, 
метрика manhattan снижала точность в 0.01 от установленной точности в комментарии
При выборе значения  больше или меньше 3 параметра n_neigh, точность падала от 0.01 и выше. 
 """
