import torch as T
import numpy as np
import argparse
import time
import model
import data
           
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--norm_constant', type=float, default=3.)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=4.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--cg_window', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    return args

def evaluate(args):
    start = time.time()
    
    device_name = 'cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    device = T.device(device_name)
    print('Using device', device_name)

    _, features, labels, train_mask, val_mask, test_mask = data.get_node_classification_data('cora', args.norm_constant, args.num_hops, large_split=False)    
    features = T.from_numpy(features).float()
    print('Built {}-hop aggregated features with k={}'.format(args.num_hops, args.norm_constant))
    
    T.set_grad_enabled(False)
    
    saved_model = T.load(args.model)
    cg = model.CountingGrid(features.shape[1], saved_model['wcg'].shape[-1], args.cg_window, 0).to(device)
    cg.wcg[:] = saved_model['wcg']
    print('Loaded saved model', args.model)
    
    # compute posterior
    logit_posterior = cg(features.to(device)).view(-1,cg.size**2)
    posterior_pred = (logit_posterior / args.alpha).softmax(1)
    posterior_emb = (logit_posterior / args.beta).softmax(1)

    # compute p(c|s)
    pcs = T.matmul(posterior_emb[train_mask].T, T.from_numpy(labels[train_mask]).float().to(device))
    pcs /= pcs.sum(1).unsqueeze(1)
        
    # compute predictions
    probs = T.matmul(posterior_pred,pcs)
    preds = probs.argmax(1).cpu().numpy()

    # compute accuracy
    val_acc = ((preds[val_mask]==labels[val_mask].argmax(1)).sum()) / val_mask.sum()
    test_acc = ((preds[test_mask]==labels[test_mask].argmax(1)).sum()) / test_mask.sum()
    
    print('Val accuracy', val_acc)
    print('Test accuracy', test_acc)
    
    print('Total execution time', time.time()-start)
    
if __name__ == '__main__':
    evaluate(parse_args())