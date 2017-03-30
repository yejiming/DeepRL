from pylab import *
import os
from os.path import join

expdir = "C:/tmp/ref"
dirnames = os.listdir(expdir)

fig, axes = subplots(4)
for dirname in dirnames:
    print(dirname)
    A = np.genfromtxt(join(expdir, dirname, 'log.txt'),delimiter='\t',dtype=None, names=True)
    x = A['TimestepsSoFar']
    axes[0].plot(x, A['EpRewMean'], '-x')
    axes[1].plot(x, A['KLOldNew'], '-x')
    axes[2].plot(x, A['Entropy'], '-x')
    axes[3].plot(x, A['EVBefore'], '-x')
legend(dirnames,loc='best').draggable()
axes[0].set_ylabel("EpRewMean")
axes[1].set_ylabel("KLOldNew")
axes[2].set_ylabel("Entropy")
axes[3].set_ylabel("EVBefore")
axes[3].set_ylim(-1,1)
axes[-1].set_xlabel("Iterations")
show()