import matplotlib.pyplot as plt
import math
import numpy as np
def f(x):
    return max(x, 0)
def f_2(x):
    return math.log(1 + math.exp(x))
x = np.arange(-100,50,0.1)
y1 = [f(i) for i in x]
y2 = [f_2(i) for i in x]
plt.title('Функция активации')
plt.grid()
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()

