import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error
from time import time
import argparse
import LoadData as DATA
import warnings

warnings.filterwarnings("ignore")

import pandas as pd


#################### Arguments ####################
def parse_args():
    print("RGPRec-Ability")
    parser = argparse.ArgumentParser(description="Run RGPRecA.")
    parser.add_argument('--path', nargs='?', default='../RGPRec-RAG/dataR_result/',
                        help='Input dataA_result path.')
    parser.add_argument('--dataset', nargs='?', default='10/',
                        help='Choose a dataset with given training ratio.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (128-512).')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[128]',
                        help="Size of each layer.")
    parser.add_argument('--layers_r1', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--layers_r2', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.8, 0.8]',
                        help='Keep probability for each deep layer and the Interaction layer. 1: no dropout.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='MSELoss',
                        help='Specify a loss type (MSELoss or BCELLoss).')
    parser.add_argument('--optimizer', nargs='?', default='Adagrad',
                        help='Specify an optimizer type (Adam, Adagrad, SGD, Momentum).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normalization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--alpha', type=float, default=0.95,
                        help='The weight between two loss functions (0.95-0.99)')
    return parser.parse_args()


class TaskDataset(Dataset):
    def __init__(self, dev_features, task_features, labels1, labels2):
        self.dev_features = torch.LongTensor(dev_features)
        self.task_features = torch.LongTensor(task_features)
        self.labels1 = torch.FloatTensor(labels1)
        self.labels2 = torch.FloatTensor(labels2)

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        return (self.dev_features[idx], self.task_features[idx],
                self.labels1[idx], self.labels2[idx])


class RGPRecA(nn.Module):
    def __init__(self, features_M_dev, features_M_task, hidden_factor, layers, layers_r1, layers_r2,
                 loss_type, epoch, batch_size, learning_rate, lamda_bilinear, keep_prob, optimizer_type,
                 batch_norm, activation_function, verbose, alpha, random_seed=2022):
        super(RGPRecA, self).__init__()

        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Store parameters
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.layers_r1 = layers_r1
        self.layers_r2 = layers_r2
        self.loss_type = loss_type
        self.features_M_dev = features_M_dev
        self.features_M_task = features_M_task
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.keep_prob = keep_prob
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.alpha = alpha

        #  Data Embedding Layer 
        self.left_embeddings = nn.Embedding(features_M_dev, hidden_factor)
        self.right_embeddings = nn.Embedding(features_M_task.shape[0], hidden_factor)

        #  Interactive Perception Layer
        self.deep_layers = nn.ModuleList()
        prev_dim = hidden_factor
        for layer_size in layers:
            self.deep_layers.append(nn.Linear(prev_dim, layer_size))
            if batch_norm:
                self.deep_layers.append(nn.BatchNorm1d(layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(1 - keep_prob[0]))
            prev_dim = layer_size

        # Multi‐label rating layer

        # Initialize R1 layers
        self.r1_layers = nn.ModuleList()
        prev_dim = layers[-1]
        for layer_size in layers_r1:
            self.r1_layers.append(nn.Linear(prev_dim, layer_size))
            if batch_norm:
                self.r1_layers.append(nn.BatchNorm1d(layer_size))
            self.r1_layers.append(nn.ReLU())
            self.r1_layers.append(nn.Dropout(1 - keep_prob[0]))
            prev_dim = layer_size
        self.r1_prediction = nn.Linear(layers_r1[-1], 1)

        # Initialize R2 layers
        self.r2_layers = nn.ModuleList()
        prev_dim = layers[-1]
        for layer_size in layers_r2:
            self.r2_layers.append(nn.Linear(prev_dim, layer_size))
            if batch_norm:
                self.r2_layers.append(nn.BatchNorm1d(layer_size))
            self.r2_layers.append(nn.ReLU())
            self.r2_layers.append(nn.Dropout(1 - keep_prob[0]))
            prev_dim = layer_size
        self.r2_prediction = nn.Linear(layers_r2[-1], 1)

        # Initialize optimizer
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'Adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'Momentum':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.95)

        # Initialize loss function
        if loss_type == 'MSELoss':
            self.criterion = nn.MSELoss()
        else:  # L2_loss
            self.criterion = nn.BCELLoss()

        # Performance tracking
        self.train_mae = []
        self.test_mae, self.test_nmae, self.test_mae_all = [], [], []

    def forward(self, dev_features, task_features):
        # Embedding lookup
        left_emb = self.left_embeddings(dev_features)    #dev_embeddings
        right_emb = self.right_embeddings(task_features) #task_embeddings

        # Sum embeddings
        summed_left = torch.sum(left_emb, dim=1)
        summed_right = torch.sum(right_emb, dim=1)
        summed_all = torch.sum(torch.cat([left_emb, right_emb], dim=1), dim=1)

        # Interactive Perception
        csmf = torch.mul(summed_left, summed_right)
        csmf = torch.cat([summed_all.unsqueeze(1), csmf.unsqueeze(1)], dim=1)
        csmf = torch.sum(csmf, dim=1)

        x = csmf
        for layer in self.deep_layers:
            x = layer(x)

        # R1 layers
        r1 = x
        for layer in self.r1_layers:
            r1 = layer(r1)
        r1 = self.r1_prediction(r1)

        # R2 layers
        r2 = x
        for layer in self.r2_layers:
            r2 = layer(r2)
        r2 = self.r2_prediction(r2)

        return r1, r2

    def train_model(self, train_data):
        """训练模型但不输出预测结果"""
        train_dataset = TaskDataset(train_data['X_U'], train_data['X_I'],
                                  train_data['Y1'], train_data['Y2'])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch):
            t1 = time()
            self.train()
            total_loss = 0

            for batch in train_loader:
                dev_features, task_features, labels1, labels2 = batch

                self.optimizer.zero_grad()
                r1_pred, r2_pred = self(dev_features, task_features)

                loss1 = self.criterion(r1_pred, labels1)
                loss2 = self.criterion(r2_pred, labels2)
                loss = self.alpha * loss1 + (1 - self.alpha) * loss2

                if self.lamda_bilinear > 0:
                    l2_reg = torch.tensor(0., requires_grad=True)
                    for param in self.parameters():
                        l2_reg += torch.norm(param)
                    loss += self.lamda_bilinear * l2_reg

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # if self.verbose > 0:
            #     print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Time: {time() - t1:.2f}s")

    def predict_all(self, test_data):
        self.eval()
        with torch.no_grad():

            dev_features = torch.LongTensor(test_data['X_U'])
            task_features = torch.LongTensor(test_data['X_I'])

            predictions = []
            batch_size = 1024
            for i in range(0, len(dev_features), batch_size):
                batch_dev = dev_features[i:i+batch_size]
                batch_task = task_features[i:i+batch_size]
                
                r1_pred, r2_pred = self(batch_dev, batch_task)
                

                for j in range(len(batch_dev)):
                    pred_dict = {
                        'dev_id': int(batch_dev[j][0]),
                        'task_id': int(batch_task[j][0]),
                        'r1': float(r1_pred[j][0]),
                        'r2': float(r2_pred[j][0]),
                        'r_a': math.ceil(((r1_pred[j][0] + r2_pred[j][0])/2).float())


                    }
                    predictions.append(pred_dict)
            
            # DataFrame
            return pd.DataFrame(predictions)


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    data = DATA.LoadData(args.path, args.dataset)
    
    # print("Training data size:", len(data.Train_data['X_U']))
    # print("All data size:", len(data.Test_data['X_U']))

    # Initialize model
    features_M_task = np.random.normal(0.0, 0.01, [data.features_M_task, args.hidden_factor])
    model = RGPRecA(data.features_M_dev, features_M_task, args.hidden_factor,
                    eval(args.layers), eval(args.layers_r1), eval(args.layers_r2),
                    args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
                    eval(args.keep_prob), args.optimizer, args.batch_norm,
                    nn.ReLU(), args.verbose, args.alpha)

    print("Training model...")
    t1 = time()
    model.train_model(data.Train_data)
    print(f"Training completed in {time() - t1:.2f}s")

    print("Generating ability for all data...")
    predictions_df = model.predict_all(data.Test_data)

    output_file = f'dataA_result/SourceForge/Ability.csv'
    # output_file = f'Ability_{args.dataset.replace("/", "_")}.csv'
    predictions_df.to_csv(output_file, index=False)
    # print(f"Ability saved to {output_file}")


