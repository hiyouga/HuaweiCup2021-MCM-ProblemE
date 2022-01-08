import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm
from clean_data import dist
from functools import partial
from method import ls, grad_phi_meta


X, Y, Z = 0, 1, 2


if __name__ == '__main__':
    anchors = [
        [0, 0, 1300],
        [5000, 0, 1700],
        [0, 5000, 1700],
        [5000, 5000, 1300]
    ]
    # G = 'label'
    G = 'pred'
    ori_errs, ls_errs, opt_errs = list(), list(), list()
    xyz_err, xy_err, xz_err, yz_err, x_err, y_err, z_err = list(), list(), list(), list(), list(), list(), list()
    tags = np.loadtxt('tags.txt', dtype=np.int)
    for C in ['正常', '异常']:
        for n in tqdm(range(324)):
            mat = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.txt"), dtype=np.int)
            target = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.{G}.txt"), dtype=np.int)
            for t in range(len(mat)):
                ''' evaluate original node '''
                b_init = ls(anchors, mat[t])
                grad_phi = partial(grad_phi_meta, ref=anchors, data=mat[t])
                ori_node = opt.root(grad_phi, b_init, method='lm').x
                ori_err = dist(tags[n], ori_node)
                ori_errs.append(ori_err)
                ''' evaluate optimal node '''
                signal = [0, 0, 0, 0]
                for j in range(4):
                    if j == target[t]-1:
                        signal[j] = mat[t][j] - 400
                    else:
                        signal[j] = mat[t][j] + 45
                b_init = ls(anchors, signal)
                ls_err = dist(tags[n], b_init)
                if ls_err < 10000:
                    ls_errs.append(ls_err)
                grad_phi = partial(grad_phi_meta, ref=anchors, data=signal)
                opt_node = opt.root(grad_phi, b_init, method='lm').x
                opt_err = dist(tags[n], opt_node)
                opt_errs.append(opt_err)
                xyz_err.append(opt_err)
                xy_err.append(((tags[n][X]-opt_node[X])**2+(tags[n][Y]-opt_node[Y])**2)**0.5)
                xz_err.append(((tags[n][X]-opt_node[X])**2+(tags[n][Z]-opt_node[Z])**2)**0.5)
                yz_err.append(((tags[n][Y]-opt_node[Y])**2+(tags[n][Z]-opt_node[Z])**2)**0.5)
                x_err.append(abs(tags[n][X]-opt_node[X]))
                y_err.append(abs(tags[n][Y]-opt_node[Y]))
                z_err.append(abs(tags[n][Z]-opt_node[Z]))
    print(f"xyz: {np.mean(xyz_err):.2f}")
    print(f"xy: {np.mean(xy_err):.2f}")
    print(f"xz: {np.mean(xz_err):.2f}")
    print(f"yz: {np.mean(yz_err):.2f}")
    print(f"x: {np.mean(x_err):.2f}")
    print(f"y: {np.mean(y_err):.2f}")
    print(f"z: {np.mean(z_err):.2f}")
    print(f"ori: {np.mean(ori_errs):.2f}")
    print(f"ls: {np.mean(ls_errs):.2f}")
    print(f"opt: {np.mean(opt_errs):.2f}")
    plt.figure()
    plt.hist(ori_errs, bins=20)
    plt.savefig(f"{G}_ori_error.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    plt.figure()
    plt.hist(ls_errs, bins=20)
    plt.savefig(f"{G}_ls_error.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    plt.figure()
    plt.hist(opt_errs, bins=20)
    plt.savefig(f"{G}_opt_error.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
