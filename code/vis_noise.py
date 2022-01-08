import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from clean_data import dist


if __name__ == '__main__':

    anchors = [
        [0, 0, 1300],
        [5000, 0, 1700],
        [0, 5000, 1700],
        [5000, 5000, 1300]
    ]

    C = '正常'
    # C = '异常'
    files = range(324) # range(324)
    tags = np.loadtxt('tags.txt', dtype=np.int)
    errs = [[] for i in range(4)]
    measure_err = []
    noise_err = []
    for n in tqdm(files):
        mat = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.txt"), dtype=np.int)
        label = np.loadtxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.label.txt"), dtype=np.int)
        for i in range(len(mat)):
            for j in range(4):
                errs[j].append(mat[i][j] - dist(tags[n], anchors[j]))
                if j+1 == label[i]:
                    noise_err.append(mat[i][j] + 45 - dist(tags[n], anchors[j]))
                else:
                    measure_err.append(mat[i][j] - dist(tags[n], anchors[j]))
    errst = np.array(errs).T
    print(f"measure err: mean {np.mean(measure_err):.2f} std {np.std(measure_err):.2f}")
    if C == '异常':
        print(f"noise err: mean {np.mean(noise_err):.2f} std {np.std(noise_err):.2f}")

    plt.figure()
    for j in range(4):
        ax = plt.subplot(2, 2, j+1)
        ax.hist(errs[j], bins=20)
    plt.savefig(f"{C}_noise_hist.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    covar = np.corrcoef(np.array(errs))
    plt.figure()
    cmap = sns.light_palette((260, 75, 60), input="husl")
    sns.heatmap(covar, annot=True, cmap=cmap, cbar=True)
    plt.savefig(f"{C}_cov.svg", format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
