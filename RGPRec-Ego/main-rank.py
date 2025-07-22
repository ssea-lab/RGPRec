import os
import sys
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn
import operator

from utils import collate_fn
from model import RGPRec
from dataloader import RGPDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='../RGPRec-Ability/dataA_result/Tensorflow/',
                    help='dataset directory path: Tensorflow/SourceForge')
parser.add_argument('--data', default='../RGPRec-Ability/dataA_result/Tensorflow', help='corresponding to datapath')
parser.add_argument('--sigma', type=str, default='0', help='social strength definition')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=256, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=200, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=1, help='the number of steps after which the learning rate decay')
parser.add_argument('--keep_prob', nargs='?', default='[0.8, 0.8]', help=' 1: no dropout.')
parser.add_argument('--test', default=False, help='test')  # False/True

args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    print('Loading data...')
    with open(args.dataset_path + 'dataset_' + args.sigma + '.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(args.dataset_path + 'list_' + args.sigma + '.pkl', 'rb') as f:
        u_items_divlist = pickle.load(f)
        u_items_list = pickle.load(f)
        u_avg_list = pickle.load(f)
        u_users_similar = pickle.load(f)
        u_users_items_list = pickle.load(f)
        u_users_items_divlist = pickle.load(f)
        i_avg_list = pickle.load(f)
        i_users_list = pickle.load(f)
        i_users_divlist = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)

    train_data = RGPDataset(train_set, u_items_divlist, u_avg_list, u_users_similar, u_users_items_divlist,
                           i_users_divlist, i_avg_list)
    valid_data = RGPDataset(valid_set, u_items_divlist, u_avg_list, u_users_similar, u_users_items_divlist,
                           i_users_divlist, i_avg_list)
    test_data = RGPDataset(test_set, u_items_divlist, u_avg_list, u_users_similar, u_users_items_divlist,
                          i_users_divlist, i_avg_list)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = RGPRec(user_count + 1, item_count + 1, rate_count + 1, args.embed_dim).to(device)

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load(args.data + '/best_checkpoint_' + args.sigma + '.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        mae, rmse, recall_results, precision_results, ndcg_results = validate(test_loader, model)
        print("\nTest Results:")
        print("MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
        print("\nRecall@K:")
        for k, value in recall_results.items():
            print(f"Recall@{k}: {value:.4f}")
        print("\nPrecision@K:")
        for k, value in precision_results.items():
            print(f"Precision@{k}: {value:.4f}")
        print("\nNDCG@K:")
        for k, value in ndcg_results.items():
            print(f"NDCG@{k}: {value:.4f}")
        return

    optimizer = optim.RMSprop(model.parameters(), args.lr)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    sum_dv_list = []
    pre_sum = 0
    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch=epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr=100)

        mae, rmse, recall_results, precision_results, ndcg_results = validate(valid_loader, model)

        if epoch == 0:
            pre_sum = rmse + mae
            sum_dv_list.append(0)
        else:
            if rmse + mae > pre_sum:
                sum_dv_list.append(1)
            else:
                pre_sum = rmse + mae
                sum_dv_list.append(0)

        if sum(sum_dv_list[-10:]) == 10:
            break

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, args.data + '/latest_checkpoint_' + args.sigma + '.pth.tar')

        if epoch == 0:
            best_sum = rmse + mae
            best_recall = recall_results
            best_precision = precision_results
            best_ndcg = ndcg_results
            torch.save(ckpt_dict, args.data + '/best_checkpoint_' + args.sigma + '.pth.tar')
        elif rmse + mae < best_sum:
            best_sum = rmse + mae
            best_recall = recall_results
            best_precision = precision_results
            best_ndcg = ndcg_results
            torch.save(ckpt_dict, args.data + '/best_checkpoint_' + args.sigma + '.pth.tar')

        print('\nEpoch {} validation:'.format(epoch))
        print("MAE: {:.4f}, RMSE: {:.4f}, Best Sum: {:.4f}".format(mae, rmse, best_sum))
        print("\nRecall@K:")
        for k, value in recall_results.items():
            print(f"Recall@{k}: {value:.4f}")
        print("\nPrecision@K:")
        for k, value in precision_results.items():
            print(f"Precision@{k}: {value:.4f}")
        print("\nNDCG@K:")
        for k, value in ndcg_results.items():
            print(f"NDCG@{k}: {value:.4f}")


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (uids, iids, labels, u_itemsdiv, u_avg, u_users, u_users_items, i_users, i_avg) in tqdm(
            enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        u_itemsdiv = u_itemsdiv.to(device)
        # u_items = u_items.to(device)
        u_avg = u_avg.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        i_avg = i_avg.to(device)

        optimizer.zero_grad()
        outputs = model(uids, iids, u_itemsdiv, u_avg, u_users, u_users_items, i_users, i_avg).to(device)

        loss = criterion(outputs, labels.unsqueeze(1)).to(device)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                  % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                     len(uids) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    errors = []
    result_dict = {}
    all_labels = {}
    
    with torch.no_grad():
        for uids, iids, labels, u_itemsdiv, u_avg, u_users, u_users_items, i_users, i_avg in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            u_itemsdiv = u_itemsdiv.to(device)
            u_avg = u_avg.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)
            i_users = i_users.to(device)
            i_avg = i_avg.to(device)
            preds = model(uids, iids, u_itemsdiv, u_avg, u_users, u_users_items, i_users, i_avg).to(device)

            for i, uid in enumerate(uids):
                uid_item = uid.item()
                iid_item = iids[i].item()
                if uid_item not in result_dict:
                    result_dict[uid_item] = {}
                    all_labels[uid_item] = {}
                
                result_dict[uid_item][iid_item] = preds[i].item()
                all_labels[uid_item][iid_item] = labels[i].item()

            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))

    k_list = [5, 10, 20]
    recall_dict = {k: [] for k in k_list}
    precision_dict = {k: [] for k in k_list}
    ndcg_dict = {k: [] for k in k_list}
    
    for uid, item_scores in result_dict.items():
        items = list(item_scores.items())
        items.sort(key=lambda x: x[1], reverse=True)
        
        actual_items = set(iid for iid, score in all_labels[uid].items() if score >= 4.0)
        
        if not actual_items:
            continue
            
        for k in k_list:
            pred_items = set(iid for iid, _ in items[:k])
            
            recall = len(pred_items & actual_items) / len(actual_items)
            recall_dict[k].append(recall)
            
            precision = len(pred_items & actual_items) / k
            precision_dict[k].append(precision)
            
            dcg = 0
            idcg = 0
            for i, (iid, _) in enumerate(items[:k]):
                if iid in actual_items:
                    dcg += 1 / np.log2(i + 2)
            
            for i in range(min(k, len(actual_items))):
                idcg += 1 / np.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_dict[k].append(ndcg)
    
    recall_results = {k: np.mean(values) if values else 0.0 for k, values in recall_dict.items()}
    precision_results = {k: np.mean(values) if values else 0.0 for k, values in precision_dict.items()}
    ndcg_results = {k: np.mean(values) if values else 0.0 for k, values in ndcg_dict.items()}
    
    print("\nRecall@K:")
    for k in k_list:
        print(f"Recall@{k}: {recall_results[k]:.4f}")
    
    print("\nPrecision@K:")
    for k in k_list:
        print(f"Precision@{k}: {precision_results[k]:.4f}")
    
    print("\nNDCG@K:")
    for k in k_list:
        print(f"NDCG@{k}: {ndcg_results[k]:.4f}")

    df = pd.DataFrame(columns=('devid', 'score_list'))
    i = 0
    for uid, value1 in result_dict.items():
        sorted_iid_score = sorted(value1.items(), key=operator.itemgetter(1), reverse=True)
        df.loc[i] = [uid, sorted_iid_score]
        i += 1
    # df.to_csv('result.csv', index=False)
    
    return mae, rmse, recall_results, precision_results, ndcg_results


if __name__ == '__main__':
    main()
