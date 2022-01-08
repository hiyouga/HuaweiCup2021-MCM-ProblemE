import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    n = 9
    # C = '正常'
    C = '异常'
    S = 0 # 0 for raw, 1 for clean
    if S == 0:
        mat = list()
        with open(os.path.join(f"{C}数据", f"{n}.{C}.txt"), 'r', encoding='utf-8') as fin:
            data = fin.read().strip().split('\n')[1:]
        for i in range(int(round(len(data) / 4))):
            get_data = lambda x: int(x.strip().split(':')[5])
            mat.append(list(map(get_data, (data[4*i], data[4*i+1], data[4*i+2], data[4*i+3]))))
        mat = np.array(mat)
    else:
        mat = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n}.{C}.txt"), dtype=np.int)
    ''' curves '''
    plt.figure()
    for j in range(4):
        ax = plt.subplot(2, 2, j+1)
        ax.plot(list(range(len(mat))), mat.T[j])
    plt.tight_layout()
    plt.savefig(f"{C}_curve.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    ''' histograms '''
    plt.figure()
    for j in range(4):
        ax = plt.subplot(2, 2, j+1)
        ax.hist(mat.T[j], bins=20)
    plt.tight_layout()
    plt.savefig(f"{C}_hist.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    ''' detect duplicates '''
    dup = dict()
    for i in range(len(mat)):
        if str(mat[i]) in dup:
            dup[str(mat[i])] += 1
        else:
            dup[str(mat[i])] = 1
