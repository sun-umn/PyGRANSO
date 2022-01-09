import matplotlib.pyplot as plt
import os
import time

def plot(log,save_plot=True):

    path = '/home/buyun/Documents/GitHub/PyGRANSO/test/plots'
    os.chdir(path)
    ts = time.strftime('%H:%M:%S_%b_%d_%Y')
    print(ts)
    os.mkdir(ts)
    os.chdir(ts)

    iters = range(0,len(log.f))

    fig, axs = plt.subplots(3)
    # fig.suptitle('Vertically stacked subplots')

    axs[0].plot(iters,log.f,'b-')
    axs[0].set_ylabel('obj val')
    # axs[0].xlabel('iter')

    axs[1].plot(iters,log.alpha,'k-')
    axs[1].set_ylabel('step_size')
    # axs[1].xlabel('iter')

    axs[2].plot(iters,log.stat_val,'r-')
    axs[2].set_ylabel('stationarity value')
    axs[2].set_xlabel('iter')

    fig.tight_layout()

    fig.savefig("output.png", dpi=500)
