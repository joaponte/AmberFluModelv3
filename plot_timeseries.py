import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

names = ['time', 'AT', 'AI1', 'AI2', 'AD', 'AV', 'U', 'I1', 'I2', 'D', 'V']
f = np.genfromtxt('AmberFluModel.txt', skip_header=1, delimiter=',', names=names)

T0 = f['AT'][0]
U0 = f['U'][0]

days = 4.0
if f['time'][-1] < days:
    days = f['time'][-1]

plt.figure(figsize=(9.0, 4.5))
plt.subplot(1, 2, 1)
plt.plot(f['time'], f['AT'] / T0, '--', linewidth=2.0, color='blue')
plt.plot(f['time'], f['AI1'] / T0, '--', linewidth=2.0, color='orange')
plt.plot(f['time'], f['AI2'] / T0, '--', linewidth=2.0, color='red')
plt.plot(f['time'], f['AD'] / T0, '--', linewidth=2.0, color='purple')

plt.plot(f['time'], f['U'] / U0, linewidth=2.0, color="blue")
plt.plot(f['time'], f['I1'] / U0, linewidth=2.0, color='orange')
plt.plot(f['time'], f['I2'] / U0, linewidth=2.0, color='red')
plt.plot(f['time'], f['D'] / U0, linewidth=2.0, color='purple')
plt.xlim([0, days])

l1 = Line2D([0], [0], linestyle='dashed', color='black', linewidth=2.0)
l2 = Line2D([0], [0], color='black', linewidth=2.0)
custom_lines = [l1, l2]
plt.legend(custom_lines, ['Cells ODE model', 'Cells Cellular model'], loc=1)

plt.subplot(1, 2, 2)
plt.plot(f['time'], f['AV'], '--', linewidth=2.0, color='black')
plt.plot(f['time'], f['V'], linewidth=2.0, color='black')
plt.yscale('log')
plt.xlim([0, days])
plt.ylim([10 ** 1, 10 ** 6.5])
plt.legend(['TCID ODE Model', 'TCID Cellular Model'], loc=4)

plt.tight_layout()
plt.savefig('TimeSeries.pdf')
plt.show()