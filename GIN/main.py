import torch
import torch.nn as nn


import dgl
import net_gin as net_gin
from args import parser_loader
import utils
from sklearn.metrics import f1_score
import copy
import warnings

from pruning import FastScheduler
import torch.nn.functional as F
import time
###################



warnings.filterwarnings('ignore')

def run_get_mask(args):
    rewind_weight=None
    device = args['device']
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'])  # adj: csr_matrix
    adj = utils.load_adj_raw(args['dataset'])

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1
    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)

    g = g.to(device)
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net_gin.GINNet(args['embedding_dim'], g, args['device'], args['spar_wei'], args['spar_adj'],remain=args["remain"], coef=args['coef'])
    net_gcn = net_gcn.to(device)
     
    # Load Teacher Model
    teacher_model = net_gin.GINNet(args['embedding_dim'], g, args['device'], args['spar_wei'], args['spar_adj'],remain=args["remain"], coef=args['coef'])
    ##teacher_model.load_state_dict(torch.load(args['teacher_path'], map_location=device))
    teacher_model.eval()

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar":0}
    best_mask = None
    mask_ls= []
        
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    pruner = lambda: False
    last_acc_val = None
    
    start_time = time.time()  # start time
    
    for epoch in range(args['total_epoch']+args['pretrain_epoch']+args['retrain_epoch']):
        if epoch == args['pretrain_epoch']:
            print("========================================================")
            print("pretrain finish")
            print("========================================================")
            pruner = lambda: True
            if args["remain"] is not None:
                T_end = args["total_epoch"]
                # utils.add_trainable_mask_noise(net_gcn,c=1e-4)
                pruner = FastScheduler(
                  net_gcn, optimizer,
                  weight_sparsity=0.90,
                  edge_sparsity=0.10,
    ignore_linear_layers=not args['spar_wei'],
    ignore_parameters=not args['spar_adj'],
    delta=args['delta'],
    T_end=args['total_epoch'],
    pretrain=args['pretrain_epoch'],
    
)
               
                
        if epoch == args["total_epoch"]+args['pretrain_epoch']:
            print("========================================================")
            print("retrain start")
            print("========================================================")
            with torch.no_grad():
                net_gcn.load_state_dict(rewind_weight)
                pruner.backward_masks = best_mask
                
        net_gcn.train()
        optimizer.zero_grad()
        output = net_gcn(g, features, pretrain=(epoch < args['pretrain_epoch']))
        
        loss = loss_func(output[idx_train], labels[idx_train])
        add_loss = 0
        if args["spar_adj"]:
            if args["dataset"] != "pubmed":
                add_loss += utils.exp_diversity_loss(net_gcn.adj_mask1_train) * 0.1# pubmed don't use this
            add_loss += utils.log_variance_loss(net_gcn.adj_mask1_train)*0.1

        loss += add_loss
        # Knowledge distillation from teacher model during denoising phase
        if epoch >= args['pretrain_epoch'] and epoch < args['pretrain_epoch'] + args['total_epoch']:
            with torch.no_grad():
                teacher_output = teacher_model(g, features, val_test=True)

            distill_loss = F.kl_div(
                F.log_softmax(output / args['temp'], dim=1),
                F.softmax(teacher_output / args['temp'], dim=1),
                reduction='batchmean'
            )
            loss += args['kd_lambda'] * distill_loss
        loss.backward()
        if epoch >= args['pretrain_epoch'] and epoch <args['pretrain_epoch'] + args['total_epoch']:
            if args["spar_adj"]:
                net_gcn.adj_mask1_train.grad.data.add_(1e-4* torch.sign(net_gcn.adj_mask1_train.data))
                pass
            if pruner():  # grow and cut step
                optimizer.step()
        else:
            optimizer.step()

        with torch.no_grad():
            net_gcn.eval()
            
            output = net_gcn( g,features, val_test=True, pretrain=(epoch < args['pretrain_epoch']))
            
            acc_val = f1_score(labels[idx_val].cpu().numpy( # type: ignore
            ), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(
            ), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy( # type: ignore
            ), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')
            
            if epoch<args['pretrain_epoch']:
                wspar_here,aspar_here = (0,0)
            else:
                wspar_here,aspar_here = pruner.getS()
            
            
            meet = ((args['spar_adj'] and aspar_here >= ((1-args["remain"])*100-1)) or not args['spar_adj']) and \
                    ((args['spar_wei'] and wspar_here >= ((1-args["remain"])*100-1)) or not args['spar_wei']) and \
                        epoch >= args['pretrain_epoch']
            
            if acc_val > best_val_acc['val_acc'] and meet:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_mask = copy.deepcopy(pruner.backward_masks)
                best_val_acc['wei_spar'] = wspar_here 
                best_val_acc['adj_spar'] = aspar_here
                if epoch <args['pretrain_epoch'] + args['total_epoch']:
                    best_val_acc['time'] = time.time()-start_time

            print("Epoch:[{}] L:[{:.3f}] AddL:[{:.3f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] time:[{:.2f}s]|"
                  .format(epoch, loss.item(), float(add_loss), acc_train * 100, acc_val * 100, acc_test * 100, wspar_here, aspar_here,time.time()-start_time), end=" ")
            if meet:
                print("Best Val:[{:.2f}] Test:[{:.2f}] AS:[{:.2f}%] WS:[{:.2f}%] at Epoch:[{}] time:[{:.2f}s]"
                      .format(
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['adj_spar'],
                          best_val_acc['wei_spar'],
                          best_val_acc['epoch'],
                          round(best_val_acc['time'],2)))
            else:
                print("")
        
    end_time = time.time()
    print("total time:",round(end_time-start_time,2))
    
    return best_mask


if __name__ == "__main__":
    args = parser_loader()
    print(args)
    print("args[dataset]",args['dataset'])
    args['teacher_path'] = 'teacher_model_cora.pth'  # Path to pre-trained teacher
    args['temp'] = 1.0  # Temperature for KD
    args['kd_lambda'] = 0.6  # Weight for KD loss
    utils.fix_seed(args['seed'])
    best_mask = run_get_mask(args)

   
