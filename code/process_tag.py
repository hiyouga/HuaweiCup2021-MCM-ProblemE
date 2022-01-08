import numpy as np


if __name__ == '__main__':

    tags = list()
    with open('Tag坐标信息.txt', 'r', encoding='utf-8') as fin:
        data = fin.read().strip().split('\n')[2:]
    for i in range(len(data)):
        line = data[i].strip().split()
        tags.append((int(line[1])*10, int(line[2])*10, int(line[3])*10))
    np.savetxt('tags.txt', tags, fmt='%d', encoding='utf-8')
