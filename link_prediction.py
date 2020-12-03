import torch as T
import numpy as np
import argparse
import time
import model
import data
from sklearn.metrics import roc_auc_score, average_precision_score

T.manual_seed(0)
np.random.seed(0)
           
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cg_size', type=int, default=31)
    parser.add_argument('--cg_window', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_batches', type=int, default=8192)
    parser.add_argument('--norm_constant', type=float, default=3.)
    parser.add_argument('--clamp_constant', type=float, default=3.)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--print_interval', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    return args
    
def link_prediction(args):
    
    start = time.time()
    
    device_name = 'cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    device = T.device(device_name)
    print('Using device', device_name)
    
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, features = data.get_link_prediction_data('cora', args.norm_constant, args.num_hops)
    features = T.from_numpy(features).float()
    print('Built {}-hop aggregated features from damaged graph with k={}'.format(args.num_hops, args.norm_constant))
    
    cg = model.CountingGrid(features.shape[1], args.cg_size, args.cg_window, args.clamp_constant).to(device)
    optimizer = optimizer = T.optim.Adam(lr=args.learning_rate, params=cg.parameters())
    
    print('Training {} / {} CG for {} batches of size {}'.format(args.cg_size, args.cg_window, args.num_batches, args.batch_size))
            
    cum_loss = 0.
    for i in range(args.num_batches): 
        indices = np.random.randint(features.shape[0],size=(args.batch_size,))
        
        optimizer.zero_grad()
        lp = cg(features[indices].to(device))
        loss = -lp.logsumexp((1,2)).mean() # the log likelihood (up to +const.)
        loss.backward()
        cum_loss += loss.item()
        optimizer.step()
        with T.no_grad(): cg.clamp()
    
        if i%args.print_interval==0:
            # average log likelihood of node
            print('Batch {} of {}: logP = '.format(i, args.num_batches), -cum_loss / (1 if i==0 else args.print_interval) - np.log(cg.size**2))
            cum_loss = 0.
        
    T.set_grad_enabled(False)
        
    print('Evaluating trained model')
    
    best_val_mean = 0.
    best_test_auc, best_test_ap = 0., 0.
    for c in range(1,11):
            
        logit_posterior = cg(features.to(device)).view(-1,cg.size**2)
        posterior = (logit_posterior*c).softmax(1)

        latent_link_prob = T.einsum('is,ij,jt->st', posterior, T.from_numpy(adj_train).float().to(device), posterior)
        latent_link_denom = T.einsum('is,jt->st', posterior, posterior)
        latent_link_prob /= latent_link_denom

        link_prob = T.einsum('is,st,jt->ij', posterior, latent_link_prob, posterior).cpu().numpy()
                         
        val_auc, val_ap = evaluate(link_prob, val_edges, val_edges_false)
        test_auc, test_ap = evaluate(link_prob, test_edges, test_edges_false)
            
        if (val_auc+val_ap)/2 > best_val_mean:
            best_val_mean = (val_auc+val_ap)/2
            best_test_auc, best_test_ap = test_auc, test_ap
            
        print('Hardening constant c={}: validation AUC={}, AP={}'.format(c, val_auc, val_ap))
    
    print('Test AUC', best_test_auc)
    print('Test AP', best_test_ap)
              
    print('Total execution time', time.time()-start)

def evaluate(all_scores, pos, neg):
    s = np.hstack((pos,neg))

    scores = np.zeros((len(pos)+len(neg)))
              
    assert len(pos)==len(neg)
              
    for i in range(len(pos)): scores[i] = all_scores[pos[i][0],pos[i][1]]
    for i in range(len(neg)): scores[len(pos)+i] = all_scores[neg[i][0],neg[i][1]]
              
    labels = np.concatenate([np.ones((len(pos),)), np.zeros((len(neg),))])

    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
              
    return roc, ap

if __name__ == '__main__':
    link_prediction(parse_args())
