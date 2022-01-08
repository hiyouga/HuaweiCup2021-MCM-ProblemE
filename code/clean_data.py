import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


def dist(p, q):
    p, q = np.array(p), np.array(q)
    return ((p - q) ** 2).sum() ** 0.5


if __name__ == '__main__':

    # C = '正常'
    C = '异常'
    n_raw, n_deltime, n_cluster = 0, 0, 0
    for n in tqdm(range(324)):
        mat = list()
        with open(os.path.join(f"{C}数据", f"{n+1}.{C}.txt"), 'r', encoding='utf-8') as fin:
            data = fin.read().strip().split('\n')[1:]
        n_raw += int(round(len(data) / 4))
        for i in range(int(round(len(data) / 4))):
            get_time = lambda x: int(x.strip().split(':')[1])
            get_data = lambda x: int(x.strip().split(':')[5])
            if get_time(data[4*i]) == get_time(data[4*i+1]) == get_time(data[4*i+2]) == get_time(data[4*i+3]):
                mat.append(list(map(get_data, (data[4*i], data[4*i+1], data[4*i+2], data[4*i+3]))))
        mat = np.array(mat)
        n_deltime += len(mat)
        new_mat = list()
        if C == '异常':
            kmeans = KMeans(n_clusters=4, random_state=0).fit(mat)
            centers = kmeans.cluster_centers_
            classes = [[] for i in range(4)]
            for t in range(len(mat)):
                classes[kmeans.labels_[t]].append(mat[t])
            for k in range(4):
                classes[k] = np.array(classes[k])
                anomaly = set()
                for j in range(4):
                    mean, std = np.mean(classes[k].T[j]), np.std(classes[k].T[j])
                    for t in range(len(classes[k])):
                        if classes[k][t][j] < mean - std or classes[k][t][j] > mean + std:
                            anomaly.add(t)
                for t in range(len(classes[k])):
                    if t not in anomaly:
                        new_mat.append(classes[k][t])
        elif C == '正常':
            anomaly = set()
            for j in range(4):
                mean, std = np.mean(mat.T[j]), np.std(mat.T[j])
                for t in range(len(mat)):
                    if mat[t][j] < mean - std or mat[t][j] > mean + std:
                        anomaly.add(t)
            for t in range(len(mat)):
                if t not in anomaly:
                    new_mat.append(mat[t])
        n_cluster += len(new_mat)
        np.savetxt(os.path.join(f"{C}数据_clean", f"{n+1}.{C}.txt"), new_mat, fmt='%d', encoding='utf-8')
    print()
    print('raw', n_raw)
    print('del', n_deltime)
    print('cluster', n_cluster)
