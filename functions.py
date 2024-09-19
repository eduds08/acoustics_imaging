import numpy as np
import matplotlib.pyplot as plt


def plot_l2_norm():
    l2_norm = np.load('./TimeReversal/l2_norm.npy')

    plt.figure()
    plt.imshow(l2_norm, aspect='auto')
    plt.colorbar()
    plt.grid()
    plt.title('L2-Norm - Time Reversal')
    plt.show()
