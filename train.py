import torch as T
import numpy as np
import argparse
import time
import model
import data

T.manual_seed(0)
np.random.seed(0)
        
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
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
    
def train(args):
    
    start = time.time()
    
    device_name = 'cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    device = T.device(device_name)
    print('Using device', device_name)
    
    _, features, _, _, _, _ = data.get_node_classification_data('cora', args.norm_constant, args.num_hops, large_split=False)    
    features = T.from_numpy(features).float()
    print('Built {}-hop aggregated features with k={}'.format(args.num_hops, args.norm_constant))
    
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
        
    T.save(cg.state_dict(), args.out)
    print('Saved model', args.out)
    
    print('Total execution time', time.time()-start)

if __name__ == '__main__':
    train(parse_args())
