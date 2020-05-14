#импорт необходимых компонент библиотеки
import os
import numpy as np
import tensorflow as tf
# тип нейронной сети, слои которой соединены друг с другом 
from keras.models import Sequential 

#тип слоев, все предыдущие слои соединены со следующими
from keras.layers import Dense 

#утилиты для работы с массивами
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#задаем seed для повторяемости результатов
np.random.seed(42)

#загружаем данные из MNIST автоматически x_train - картинки, y_train лэйблы(правильные ответы)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
#Преобразуем данные
x_train = x_train.reshape(60000, 784)

# #Нормализуем данные
x_train = x_train.astype('float32')
x_train /= 255.0
#преобразование меток
#5 [0,0,0,0,0,1,0,0,0,0]
y_train = np_utils.to_categorical(y_train, 10)

#Создаем модель последовательная,слои идут друг за другом
model = Sequential()
#Добавляем уровни сети
model.add(Dense(800, input_dim=784, init="normal", activation="relu"))
model.add(Dense(10, init="normal", activation="softmax"))

#Компиляция модели SGD метод стохастического градиентного спуска, мера ошибки , метрика - точность
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())


#Обучение сети
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

#Тест на данных x_text
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255.0
y_test = np_utils.to_categorical(y_test, 10)
predictions = model.predict(x_test)
predictions = np.argmax(predictions,axis=1)
output = np.column_stack((range(1, predictions.shape[0] + 1), predictions))

#Вывод в файл sub.csv
np.savetxt('sub.csv', output, header=" ID, Label", comments="", fmt="%d,%d")
