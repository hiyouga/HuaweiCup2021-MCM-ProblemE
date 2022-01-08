import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from clean_data import dist
from sklearn.cluster import KMeans


if __name__ == '__main__':

    dist_n = list()
    for N in range(1, 11):
        dists = list()
        for C in ['正常', '异常']:
            for n in tqdm(range(324)):
                mat = list()
                with open(os.path.join(f"{C}数据", f"{n+1}.{C}.txt"), 'r', encoding='utf-8') as fin:
                    data = fin.read().strip().split('\n')[1:]
                for i in range(int(round(len(data) / 4))):
                    get_time = lambda x: int(x.strip().split(':')[1])
                    get_data = lambda x: int(x.strip().split(':')[5])
                    if get_time(data[4*i]) == get_time(data[4*i+1]) == get_time(data[4*i+2]) == get_time(data[4*i+3]):
                        mat.append(list(map(get_data, (data[4*i], data[4*i+1], data[4*i+2], data[4*i+3]))))
                mat = np.array(mat)
                kmeans = KMeans(n_clusters=N, random_state=0).fit(mat)
                centers = kmeans.cluster_centers_
                classes = [[] for i in range(N)]
                dists.append(np.mean([dist(mat[t], centers[kmeans.labels_[t]]) for t in range(len(mat))]))
        dist_n.append(np.max(dists))
    plt.figure()
    plt.plot(list(range(1, 11)), dist_n)
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.savefig('kmeans_sse.svg', format='svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
