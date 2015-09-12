__author__ = 'omohamme'
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
x = np.arange(0, np.pi*4, 0.01)
print len(x)
sine_wave = np.sin(x)
new_wave = []
for item in sine_wave:
    if item > 0.5:
        new_wave.append(0.5)
    elif item < -0.5:
        new_wave.append(-0.5)
    else:
        new_wave.append(item)
# tanh_wave = np.tanh(np.sin(x*2*np.pi) * 4)
# x1 = np.arange(0, 10, 0.01)
tanh_wave = np.tanh(10 * np.sin(x*2*np.pi))
# tanh_wave = np.sinh(np.tan(x))
# square_wave = signal.square(x)
# convolved = sine_wave * square_wave
# convolved = sine_wave * tanh_wave
# print len(convolved)

plt.figure()
# plt.plot(x, sine_wave, 'bo', x, new_wave,'r+')#, x, square_wave, 'r*', x, convolved, 'g*')
# plt.plot(x, tanh_wave,'r+',x, sine_wave, 'bo')#, x, square_wave, 'r*', x, convolved, 'g*')
plt.plot(x, tanh_wave,'r+')
plt.plot(x, sine_wave, 'bo')#, x, square_wave, 'r*', x, convolved, 'g*')
# plt.plot(x,sine_wave,'bo')
plt.show()