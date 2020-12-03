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
    parser.add_argument('--model')
    parser.add_argument('--norm_constant', type=float, default=3.)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=3.5)
    parser.add_argument('--beta', type=float, default=2.5)
    parser.add_argument('--cg_window', type=int, default=5)
    parser.add_argument('--clamp_constant', type=float, default=3.)
    parser.add_argument('--finetune_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=10**4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    return args

def finetune_evaluate(args):
    start = time.time()
    
    device_name = 'cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    device = T.device(device_name)
    print('Using device', device_name)

    _, features, labels, train_mask, val_mask, test_mask = data.get_node_classification_data('cora', args.norm_constant, args.num_hops, large_split=True)    
    features = T.from_numpy(features).float()
    print('Built {}-hop aggregated features with k={}'.format(args.num_hops, args.norm_constant))
    
    T.set_grad_enabled(False)
    
    saved_model = T.load(args.model)
    cg = model.CountingGrid(features.shape[1], saved_model['wcg'].shape[-1], args.cg_window, args.clamp_constant).to(device)
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
    
    print('Initial val accuracy', val_acc)
    print('Initial test accuracy', test_acc)
    
    pcs.log_() # use log-domain parametrization of p(c|s)
    
    # make CG and p(c|s) matrix trainable
    T.set_grad_enabled(True)
    cg.wcg.requires_grad = True
    pcs.requires_grad = True
    
    opt = T.optim.SGD([cg.wcg,pcs], lr=args.learning_rate, momentum=args.momentum)
    
    nll = T.nn.NLLLoss().to(device)

    best_val_acc, best_test_acc = 0., 0.
    
    print('Made p(c|s) and CG parameters trainable, finetuning for {} epochs'.format(args.finetune_steps))
    
    for i in range(args.finetune_steps):

        opt.zero_grad()

        logit_posterior = cg(features.to(device)).view(-1,cg.size**2)
        posterior_pred = (logit_posterior / args.alpha).softmax(1)

        # compute predictions
        probs = T.matmul(posterior_pred, pcs.softmax(1))
        preds = probs.argmax(1).cpu().numpy()
        
        # get the current val/test accuracy
        with T.no_grad():
            valacc = ((preds[val_mask]==labels[val_mask].argmax(1)).sum())/val_mask.sum()
            testacc = ((preds[test_mask]==labels[test_mask].argmax(1)).sum())/test_mask.sum()

            if valacc > best_val_acc:
                best_val_acc, best_test_acc = valacc, testacc
                
            print('Epoch {} of {}: validation accuracy {}'.format(i, args.finetune_steps, valacc))

        loss = nll(T.log(probs[train_mask]), T.from_numpy(labels[train_mask].argmax(1)).to(device))
        loss.backward()
        opt.step()
        with T.no_grad(): cg.clamp()

    print('Best val accuracy', best_val_acc)
    print('Test accuracy at best epoch', best_test_acc)
    
    print('Total execution time', time.time()-start)
    
if __name__ == '__main__':
    finetune_evaluate(parse_args())
