import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ ==  '__main__':
    result = np.loadtxt('task5.result.txt', dtype=np.float32)
    predict = list()
    plt.figure()
    for i in range(3):
        ax = plt.subplot(3, 1, i+1)
        x = list(range(len(result.T[i])))
        y = result.T[i]
        f = np.polyfit(x, y, 10)
        p = np.poly1d(f)
        y_hat = p(x)
        predict.append(y_hat)
        ax.scatter(x, y, s=0.5, color='tab:blue')
        ax.plot(x, y_hat, color='tab:red')
    plt.tight_layout()
    plt.savefig('move2d.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(result.T[0], result.T[1], result.T[2], s=0.5, color='tab:blue')
    ax.plot(predict[0], predict[1], predict[2], color='tab:red')
    ax.set_xlim3d(0, 5000)
    ax.set_ylim3d(0, 5000)
    ax.set_zlim3d(0, 3000)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('move3d.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
