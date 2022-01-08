import os
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from functools import partial


X, Y, Z = 0, 1, 2


def ls(ref, data):
    A = 2 * np.matrix([
        [ref[0][X] - ref[3][X], ref[0][Y] - ref[3][Y], ref[0][Z] - ref[3][Z]],
        [ref[1][X] - ref[3][X], ref[1][Y] - ref[3][Y], ref[1][Z] - ref[3][Z]],
        [ref[2][X] - ref[3][X], ref[2][Y] - ref[3][Y], ref[2][Z] - ref[3][Z]]
    ])
    b = np.matrix([
        [ref[0][X] ** 2 - ref[3][X] ** 2 + ref[0][Y] ** 2 - ref[3][Y] ** 2 + ref[0][Z] ** 2 - ref[3][Z] ** 2 + data[3] ** 2 - data[0] ** 2],
        [ref[1][X] ** 2 - ref[3][X] ** 2 + ref[1][Y] ** 2 - ref[3][Y] ** 2 + ref[1][Z] ** 2 - ref[3][Z] ** 2 + data[3] ** 2 - data[1] ** 2],
        [ref[2][X] ** 2 - ref[3][X] ** 2 + ref[2][Y] ** 2 - ref[3][Y] ** 2 + ref[2][Z] ** 2 - ref[3][Z] ** 2 + data[3] ** 2 - data[2] ** 2]
    ])
    return (A.T * A).I * A.T * b


def phi_meta(x, ref, data):
    dist = 0
    for i in range(len(ref)):
        dx = x[X] - ref[i][X]
        dy = x[Y] - ref[i][Y]
        dz = x[Z] - ref[i][Z]
        dist += ((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5 - data[i]) ** 2
    return dist


def grad_phi_meta(x, ref, data):
    fx, fy, fz = 0, 0, 0
    for i in range(len(ref)):
        dx = x[X] - ref[i][X]
        dy = x[Y] - ref[i][Y]
        dz = x[Z] - ref[i][Z]
        dist = (dx ** 2 + dy ** 2 + dz ** 2) ** -0.5
        fx += 2 * dx - 2 * data[i] * dx * dist
        fy += 2 * dy - 2 * data[i] * dy * dist
        fz += 2 * dz - 2 * data[i] * dz * dist
    return np.array([fx, fy, fz])


if __name__ == '__main__':
    anchors = [
        [0, 0, 1300],
        [5000, 0, 1700],
        [0, 5000, 1700],
        [5000, 5000, 1300]
    ]
    C = '异常'
    tags = np.loadtxt('tags.txt', dtype=np.int)
    states = [0b0000, 0b0001, 0b0010, 0b0100, 0b1000]
    correct, count_all = 0, 0
    for n in tqdm(range(20)):
        mat = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.txt"), dtype=np.int)
        label = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.label.txt"), dtype=np.int)
        for t in range(len(mat)):
            b_init = ls(anchors, mat[t])
            res = list()
            for s in states:
                signal = [0, 0, 0, 0]
                for j in range(4):
                    if (s >> j) & 1 == 1:
                        signal[j] = mat[t][j] - 300
                    else:
                        signal[j] = mat[t][j] + 0
                sols = list()
                for j in range(4):
                    p_anchors = [anchors[k] for k in range(4) if k != j]
                    p_data = [signal[k] for k in range(4) if k != j]
                    phi = partial(phi_meta, ref=p_anchors, data=p_data)
                    grad_phi = partial(grad_phi_meta, ref=p_anchors, data=p_data)
                    sol = opt.root(grad_phi, b_init, method='lm')
                    sols.append(sol.x)
                res.append(np.array(sols).std(axis=0).mean())
            predict = np.argmin(res)
            if predict == label[t]:
                correct += 1
            count_all += 1
    print()
    print(correct / count_all * 100)
