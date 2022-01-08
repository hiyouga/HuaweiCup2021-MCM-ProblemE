import torch
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from clean_data import dist
from functools import partial
from neural_network import ffn
from method import ls, phi_meta, grad_phi_meta


if __name__ == '__main__':

    ''' read data '''
    fname = 'task3'
    mat, anchors= list(), list()
    with open(f"{fname}.txt", 'r', encoding='utf-8') as fin:
        fdata = fin.read().strip().split('\n')
    anchors, data = fdata[:4], fdata[4:]
    anchors = [list(map(int, line.strip().split(','))) for line in anchors]
    for i in range(int(round(len(data) / 4))):
        get_data = lambda x: int(x.strip().split(':')[5])
        mat.append(list(map(get_data, (data[4*i], data[4*i+1], data[4*i+2], data[4*i+3]))))
    mat = np.array(mat)
    ''' generate feature '''
    states = [0b0000, 0b0001, 0b0010, 0b0100, 0b1000]
    features = list()
    for t in tqdm(range(len(mat))):
        b_init = ls(anchors, mat[t])
        feat = list()
        for s in states:
            signal = [0, 0, 0, 0]
            for j in range(4):
                if (s >> j) & 1 == 1:
                    signal[j] = mat[t][j] - 400
                else:
                    signal[j] = mat[t][j] + 45
            sols = list()
            ''' compute reference node '''
            phi = partial(phi_meta, ref=anchors, data=signal)
            grad_phi = partial(grad_phi_meta, ref=anchors, data=signal)
            ref_node = opt.root(grad_phi, b_init, method='lm').x
            feat.append(phi(ref_node)) # ref node error
            ''' compute three test nodes '''
            test_node_err, node_dist = list(), list()
            for j in range(4):
                p_anchors = [anchors[k] for k in range(4) if k != j]
                p_data = [signal[k] for k in range(4) if k != j]
                phi = partial(phi_meta, ref=p_anchors, data=p_data)
                grad_phi = partial(grad_phi_meta, ref=p_anchors, data=p_data)
                sol = opt.root(grad_phi, b_init, method='lm').x
                sols.append(sol)
                test_node_err.append(phi(sol)) # test node error
                node_dist.append(dist(sol, ref_node)) # node distence
            node_var = (np.array(sols).std(axis=0) ** 2).tolist()
            feat = feat + test_node_err + node_dist + node_var
        features.append(np.array(feat))
    features = np.array(features)
    ''' predict '''
    state_dict = torch.load('best_model.pt', map_location='cpu')
    ffn.load_state_dict(state_dict['model'])
    with torch.no_grad():
        inputs = torch.from_numpy(features).float()
        inputs = (inputs - state_dict['mean']) / state_dict['std']
        outputs = ffn(inputs)
        predicts = torch.argmax(outputs, dim=-1).numpy()
    if fname == 'task2' or fname == 'task3':
        predicts[0:5] = 0
    res = list()
    for t in tqdm(range(len(mat))):
        signal = [0, 0, 0, 0]
        for j in range(4):
            if j == predicts[t]-1:
                signal[j] = mat[t][j] - 400
            else:
                signal[j] = mat[t][j] + 45
        b_init = ls(anchors, signal)
        grad_phi = partial(grad_phi_meta, ref=anchors, data=signal)
        opt_node = opt.root(grad_phi, b_init, method='lm').x
        res.append(opt_node)
    res = np.array(res)
    np.savetxt(f"{fname}.predict.txt", predicts, fmt='%.4f', encoding='utf-8')
    np.savetxt(f"{fname}.result.txt", res, fmt='%.4f', encoding='utf-8')
