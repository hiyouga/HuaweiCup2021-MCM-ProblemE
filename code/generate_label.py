import os
import numpy as np
from tqdm import tqdm
from clean_data import dist


if __name__ == '__main__':

    anchors = [
        [0, 0, 1300],
        [5000, 0, 1700],
        [0, 5000, 1700],
        [5000, 5000, 1300]
    ]
    tags = np.loadtxt('tags.txt', dtype=np.int)
    for n in tqdm(range(324)):
        mat = np.loadtxt(os.path.join('正常数据_clean', f"{n+1}.正常.txt"), dtype=np.int)
        pos = [0] * len(mat)
        np.savetxt(os.path.join('正常数据_clean', f"{n+1}.正常.label.txt"), pos, fmt='%d', encoding='utf-8')
        mat = np.loadtxt(os.path.join('异常数据_clean', f"{n+1}.异常.txt"), dtype=np.int)
        pos = list()
        for i in range(len(mat)):
            max_j, max_err = -1, -1
            for j in range(4):
                err = mat[i][j] - dist(tags[n], anchors[j])
                if err > max_err:
                    max_j, max_err = j, err
            pos.append(max_j+1)
        np.savetxt(os.path.join('异常数据_clean', f"{n+1}.异常.label.txt"), pos, fmt='%d', encoding='utf-8')
