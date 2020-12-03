import torch as T
import numpy as np
import argparse
import time
import model
from matplotlib import pyplot as pt
import data
           
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--out')
    parser.add_argument('--norm_constant', type=float, default=3.)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--cg_window', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    return args

def visualize(args):
    start = time.time()
    
    device_name = 'cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    device = T.device(device_name)
    print('Using device', device_name)
    
    _, features, labels, _, _, _ = data.get_node_classification_data('cora', args.norm_constant, args.num_hops, large_split=False)    
    features = T.from_numpy(features).float()
    print('Built {}-hop aggregated features with k={}'.format(args.num_hops, args.norm_constant))
    
    T.set_grad_enabled(False)
    
    saved_model = T.load(args.model)
    cg = model.CountingGrid(features.shape[1], saved_model['wcg'].shape[-1], args.cg_window, 0).to(device)
    cg.wcg[:] = saved_model['wcg']
    print('Loaded saved model', args.model)
    
    # compute posterior
    logit_posterior = cg(features.to(device)).view(-1,cg.size**2)
    posterior = logit_posterior.softmax(1)

    # compute p(c|s)
    pcs = T.matmul(posterior.T, T.from_numpy(labels).float().to(device))
    pcs /= pcs.sum(1).unsqueeze(1)
        
    cats = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    pt.figure(figsize=(10,6))
    for i in range(7):
        pt.subplot(2,4,i+1)
        pt.title(cats[i])
        pt.imshow(pcs[:,i].cpu().numpy().reshape(cg.size,cg.size))
    pt.savefig(args.out)
    
    print('Saved image', args.out)
    
    print('Total execution time', time.time()-start)
    
if __name__ == '__main__':
    visualize(parse_args())