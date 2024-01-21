# %%
import os
from torch_geometric.data import HeteroData
import torch
import pyreadr
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd

# %%
SEED = 666

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# %%
def load_lncRNA_data():
    lncRNA_info = torch.load(current_path + '/../handled_data/lncRNA_info.pth')

    lncRNADis_edge = lncRNA_info['lncRNADis_edge']
    lncRNA_edge = lncRNA_info['lncRNA_edge']
    lncRNA_feature = lncRNA_info['lncRNA_feature'].to(torch.float)

    disease_edge = torch.load(current_path + '/../handled_data/disease_info.pth')

    return lncRNADis_edge, lncRNA_edge, lncRNA_feature, disease_edge

def load_miRNA_data():
    miRNA_info = torch.load(current_path + '/../handled_data/miRNA_info_HMDD.pth')

    miRNADis_edge = miRNA_info['miRNADis_edge']
    miRNA_edge = miRNA_info['miRNA_edge']
    miRNA_feature = miRNA_info['miRNA_feature'].to(torch.float)

    disease_edge = torch.load(current_path + '/../handled_data/disease_info.pth')

    return miRNADis_edge, miRNA_edge, miRNA_feature, disease_edge

def load_mRNA_data():
    mRNA_info = torch.load(current_path + '/../handled_data/mRNA_info.pth')

    mRNADis_edge = mRNA_info['mRNADis_edge']
    mRNA_edge = mRNA_info['mRNA_edge']
    mRNA_feature = mRNA_info['mRNA_feature'].to(torch.float)

    disease_edge = torch.load(current_path + '/../handled_data/disease_info.pth')

    return mRNADis_edge, mRNA_edge, mRNA_feature, disease_edge

def load_snoRNA_data():
    snoRNA_info = torch.load(current_path + '/../handled_data/snoRNA_info.pth')

    snoRNADis_edge = snoRNA_info['snoRNADis_edge']
    snoRNA_edge = snoRNA_info['snoRNA_edge']
    snoRNA_feature = snoRNA_info['snoRNA_feature'].to(torch.float)

    disease_edge = torch.load(current_path + '/../handled_data/disease_info.pth')

    return snoRNADis_edge, snoRNA_edge, snoRNA_feature, disease_edge

def load_lncRNA_miRNA_mRNA_snoRNA_data():
    #
    lncRNA_miRNA_info = torch.load(current_path + '/../handled_data/lncRNA_miRNA_info.pth')

    LncMi_edge = lncRNA_miRNA_info['LncMi_edge']
    MiLnc_edge = lncRNA_miRNA_info['MiLnc_edge']

    #
    lncRNA_mRNA_info = torch.load(current_path + '/../handled_data/lncRNA_mRNA_info.pth')

    LncM_edge = lncRNA_mRNA_info['LncM_edge']
    MLnc_edge = lncRNA_mRNA_info['MLnc_edge']

    #
    lncRNA_snoRNA_info = torch.load(current_path + '/../handled_data/lncRNA_snoRNA_info.pth')

    LncSno_edge = lncRNA_snoRNA_info['LncSno_edge']
    SnoLnc_edge = lncRNA_snoRNA_info['SnoLnc_edge']

    #
    miRNA_mRNA_info = torch.load(current_path + '/../handled_data/miRNA_mRNA_info.pth')

    MiM_edge = miRNA_mRNA_info['MiM_edge']
    MMi_edge = miRNA_mRNA_info['MMi_edge']

    #
    miRNA_snoRNA_info = torch.load(current_path + '/../handled_data/miRNA_snoRNA_info.pth')

    MiSno_edge = miRNA_snoRNA_info['MiSno_edge']
    SnoMi_edge = miRNA_snoRNA_info['SnoMi_edge']

    #
    snoRNA_mRNA_info = torch.load(current_path + '/../handled_data/snoRNA_mRNA_info.pth')

    SnoM_edge = snoRNA_mRNA_info['SnoM_edge']
    MSno_edge = snoRNA_mRNA_info['MSno_edge']

    return LncMi_edge, MiLnc_edge, \
        LncM_edge, MLnc_edge, \
            LncSno_edge, SnoLnc_edge, \
                MiM_edge, MMi_edge, \
                    MiSno_edge, SnoMi_edge, \
                        SnoM_edge, MSno_edge

# %%
def construct_heterogeneous_network():
    data = HeteroData()

    data['lncRNA'].node_id = torch.arange(len(lncRNA_names))
    data['miRNA'].node_id = torch.arange(len(miRNA_names))
    data['snoRNA'].node_id = torch.arange(len(snoRNA_names))
    data['mRNA'].node_id = torch.arange(len(mRNA_names))
    data['disease'].node_id = torch.arange(len(disease_names))

    data['lncRNA'].x = (lncRNA_feature - torch.min(lncRNA_feature, dim=1, keepdim=True).values) / (torch.max(lncRNA_feature, dim=1, keepdim=True).values - torch.min(lncRNA_feature, dim=1, keepdim=True).values)
    data['miRNA'].x = (miRNA_feature - torch.min(miRNA_feature, dim=1, keepdim=True).values) / (torch.max(miRNA_feature, dim=1, keepdim=True).values - torch.min(miRNA_feature, dim=1, keepdim=True).values)
    data['snoRNA'].x = (snoRNA_feature - torch.min(snoRNA_feature, dim=1, keepdim=True).values) / (torch.max(snoRNA_feature, dim=1, keepdim=True).values - torch.min(snoRNA_feature, dim=1, keepdim=True).values)
    data['mRNA'].x = (mRNA_feature - torch.min(mRNA_feature, dim=1, keepdim=True).values) / (torch.max(mRNA_feature, dim=1, keepdim=True).values - torch.min(mRNA_feature, dim=1, keepdim=True).values)

    data['lncRNA', 'LncLnc', 'lncRNA'].edge_index = lncRNA_edge
    data['lncRNA', 'LncDis', 'disease'].edge_index = lncRNADis_edge
    
    data['miRNA', 'MiMi', 'miRNA'].edge_index = miRNA_edge
    data['miRNA', 'MiDis', 'disease'].edge_index = miRNADis_edge

    data['snoRNA', 'SnoSno', 'snoRNA'].edge_index = snoRNA_edge
    data['snoRNA', 'SnoDis', 'disease'].edge_index = snoRNADis_edge

    data['mRNA', 'MM', 'mRNA'].edge_index = mRNA_edge
    data['mRNA', 'MDis', 'disease'].edge_index = mRNADis_edge

    data['disease', 'DisDis', 'disease'].edge_index = disease_edge

    data['lncRNA', 'LncMi', 'miRNA'].edge_index = LncMi_edge
    data['lncRNA', 'LncM', 'mRNA'].edge_index = LncM_edge
    data['lncRNA', 'LncSno', 'snoRNA'].edge_index = LncSno_edge
    data['miRNA', 'MiM', 'mRNA'].edge_index = MiM_edge
    data['miRNA', 'MiSno', 'snoRNA'].edge_index = MiSno_edge
    data['mRNA', 'SnoM', 'snoRNA'].edge_index = MSno_edge

    return data
# %%
class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# %%
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# %%
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        # self.dropout = torch.nn.Dropout(p=0.5) 

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[RNA_type][row], z_dict['disease'][col]], dim=-1)

        z = self.lin1(z).relu()
        # z = self.dropout(z)
        z = self.lin2(z)

        z = z.view(-1).sigmoid()

        return z

# %%
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=70, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, fold):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, fold)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, fold)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # path = os.path.join(self.save_path, 'best_network.pth')
        pathtmp = 'best_network_' + str(fold) + '_' + RNA_type + '.pt'
        path = os.path.join(self.save_path, pathtmp)
        # torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        torch.save(model, path)	# 这里会存储迄今最优模型的参数

        self.val_loss_min = val_loss

# %%
def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data[RNA_type, 'disease'].edge_label_index)
    target = train_data[RNA_type, 'disease'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)

# %%
@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data[RNA_type, 'disease'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data[RNA_type, 'disease'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), pred.cpu().numpy(), target.cpu().numpy()

# %%
def load_prediction(edge_type, RNA_type):
    load_file = current_path + '/' + edge_type + '_model/' + RNA_type + '.npz'
    
    pred_all = np.load(load_file, allow_pickle=True)

    pred_score = pred_all['score_matrix']
    label_matrix = pred_all['label_matrix']

    return pred_score, label_matrix

# %%
if __name__ == '__main__':
    current_path = os.path.dirname(__file__)

    # %%
    device = torch.device('cuda')

    # %%
    # load data
    lncRNADis_edge, lncRNA_edge, lncRNA_feature, disease_edge = load_lncRNA_data()
    miRNADis_edge, miRNA_edge, miRNA_feature, disease_edge = load_miRNA_data()
    mRNADis_edge, mRNA_edge, mRNA_feature, disease_edge = load_mRNA_data()
    snoRNADis_edge, snoRNA_edge, snoRNA_feature, disease_edge = load_snoRNA_data()

    LncMi_edge, MiLnc_edge, \
        LncM_edge, MLnc_edge, \
            LncSno_edge, SnoLnc_edge, \
                MiM_edge, MMi_edge, \
                    MiSno_edge, SnoMi_edge, \
                        SnoM_edge, MSno_edge \
                            = load_lncRNA_miRNA_mRNA_snoRNA_data()

    lncRNA_names = pyreadr.read_r(current_path + '/../data/lncRNA_names.RData')['lncRNA_names'].to_numpy().reshape(-1)
    snoRNA_names = pyreadr.read_r(current_path + '/../data/snoRNA_names.RData')['snoRNA_names'].to_numpy().reshape(-1)
    miRNA_names = pyreadr.read_r(current_path + '/../data/miRNA_names.RData')['miRNA_names'].to_numpy().reshape(-1)
    mRNA_names = pyreadr.read_r(current_path + '/../data/mRNA_names.RData')['mRNA_names'].to_numpy().reshape(-1)    
    disease_names = pyreadr.read_r(current_path + '/../data/disease_names.RData')['disease_names'].to_numpy().reshape(-1)

    # %%
    data = construct_heterogeneous_network()

    data['disease'].x = (torch.rand(len(disease_names), 32) - torch.min(torch.rand(len(disease_names), 32), dim=1, keepdim=True).values) / (torch.max(torch.rand(len(disease_names), 32), dim=1, keepdim=True).values - torch.min(torch.rand(len(disease_names), 32), dim=1, keepdim=True).values)

    # %%
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)

    # %%
    data.to(device=device)

    # %%
    RNA_type_all = [['snoRNA', 'SnoDis'], ['lncRNA', 'LncDis'], ['miRNA', 'MiDis'], ['mRNA', 'MDis']]

    for RNA_type, edge_type in RNA_type_all:
        print(RNA_type)
        print(edge_type)

        # %%
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=(RNA_type, edge_type, 'disease'),
            rev_edge_types=('disease', 'rev_' + edge_type, RNA_type),
        )
        train_data, val_data, test_data = transform(data)
        
        # %%
        model = Model(hidden_channels=32).to(device)

        learning_rate = 0.0001
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        # %%
        early_stopping = EarlyStopping(current_path + '/'+edge_type+'_model/')

        # %%
        for epoch in range(1, 1001):
            loss = train()

            train_rmse, train_pred, train_true = test(train_data)

            val_rmse, val_pred, val_true = test(val_data)
            val_AUC = roc_auc_score(val_true, val_pred)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
                f'Val: {val_rmse:.4f}, Val:{val_AUC:.4f}')

            scheduler.step()
            early_stopping(1-val_AUC, model, 1)
            #达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break #跳出迭代，结束训练

        # %%
        # construct test dataset
        score_matrix = np.zeros((data[RNA_type].node_id.shape[0], data['disease'].node_id.shape[0]))

        label_matrix = np.zeros((data[RNA_type].node_id.shape[0], data['disease'].node_id.shape[0]))
        label_matrix[(data[RNA_type, edge_type, 'disease'].edge_index.cpu().numpy()[0], 
                        data[RNA_type, edge_type, 'disease'].edge_index.cpu().numpy()[1])] = 1

        edge_label = torch.tensor(label_matrix.reshape(-1), dtype=torch.float32).to(device)

        edge_label_index = torch.tensor(np.where(score_matrix == 0)).to(device)

        all_data = test_data
        all_data[RNA_type, edge_type, 'disease'].edge_label = edge_label
        all_data[RNA_type, edge_type, 'disease'].edge_label_index = edge_label_index

        all_rmse, all_pred, all_true = test(all_data)

        score_matrix = all_pred.reshape(score_matrix.shape)

        auc = roc_auc_score(all_true, all_pred)

        np.savez(file=current_path + '/'+edge_type+'_model/'+RNA_type+'.npz',
                all_pred = all_pred, score_matrix = score_matrix, label_matrix = label_matrix)

    # %%
    disease_name = pyreadr.read_r(current_path + '/../data/disease_names.RData')['disease_names'].to_numpy().reshape(-1)

    # DOID: 684 LIHC
    # DOID:3007 BRCA
    # DOID:1612 breast cancer
    HCC_id = np.where(disease_name == 'DOID:684')[0]

    project_names = 'LIHC'

    # %%
    lncRNA_pred_score, lncRNA_label_matrix = load_prediction('LncDis', 'lncRNA')
    miRNA_pred_score, miRNA_label_matrix = load_prediction('MiDis', 'miRNA')
    mRNA_pred_score, mRNA_label_matrix = load_prediction('MDis', 'mRNA')
    snoRNA_pred_score, snoRNA_label_matrix = load_prediction('SnoDis', 'snoRNA')

    # %%
    lncRNA_HCC_score = lncRNA_pred_score[:, HCC_id].reshape(-1)
    miRNA_HCC_score = miRNA_pred_score[:, HCC_id].reshape(-1)
    snoRNA_HCC_score = snoRNA_pred_score[:, HCC_id].reshape(-1)
    mRNA_HCC_score = mRNA_pred_score[:, HCC_id].reshape(-1)

    lncRNA_HCC_label = lncRNA_label_matrix[:, HCC_id].reshape(-1)
    miRNA_HCC_label = miRNA_label_matrix[:, HCC_id].reshape(-1)
    snoRNA_HCC_label = snoRNA_label_matrix[:, HCC_id].reshape(-1)
    mRNA_HCC_label = mRNA_label_matrix[:, HCC_id].reshape(-1)

    # %%
    lncRNA_HCC_idx = np.argsort(-lncRNA_HCC_score)
    miRNA_HCC_idx = np.argsort(-miRNA_HCC_score)
    snoRNA_HCC_idx = np.argsort(-snoRNA_HCC_score)
    mRNA_HCC_idx = np.argsort(-mRNA_HCC_score)

    # %%
    lncRNA_HCC_case = {'lncRNA_name': lncRNA_names[lncRNA_HCC_idx].tolist(),
                   'lncRNA_score': lncRNA_HCC_score[lncRNA_HCC_idx].tolist(),
                   'lncRNA_label': lncRNA_HCC_label[lncRNA_HCC_idx].tolist()}
    lncRNA_HCC_case = pd.DataFrame(lncRNA_HCC_case)
    lncRNA_HCC_case.to_csv(current_path + '/'+project_names+'_result/lncRNA_'+project_names+'_case.csv')

    miRNA_HCC_case = {'miRNA_name': miRNA_names[miRNA_HCC_idx].tolist(),
                   'miRNA_score': miRNA_HCC_score[miRNA_HCC_idx].tolist(),
                   'miRNA_label': miRNA_HCC_label[miRNA_HCC_idx].tolist()}
    miRNA_HCC_case = pd.DataFrame(miRNA_HCC_case)
    miRNA_HCC_case.to_csv(current_path + '/'+project_names+'_result/miRNA_'+project_names+'_case.csv')

    mRNA_HCC_case = {'mRNA_name': mRNA_names[mRNA_HCC_idx].tolist(),
                   'mRNA_score': mRNA_HCC_score[mRNA_HCC_idx].tolist(),
                   'mRNA_label': mRNA_HCC_label[mRNA_HCC_idx].tolist()}
    mRNA_HCC_case = pd.DataFrame(mRNA_HCC_case)    
    mRNA_HCC_case.to_csv(current_path + '/'+project_names+'_result/mRNA_'+project_names+'_case.csv')

    snoRNA_HCC_case = {'snoRNA_name': snoRNA_names[snoRNA_HCC_idx].tolist(),
                   'snoRNA_score': snoRNA_HCC_score[snoRNA_HCC_idx].tolist(),
                   'snoRNA_label': snoRNA_HCC_label[snoRNA_HCC_idx].tolist()}
    snoRNA_HCC_case = pd.DataFrame(snoRNA_HCC_case)
    snoRNA_HCC_case.to_csv(current_path + '/'+project_names+'_result/snoRNA_'+project_names+'_case.csv')

# %%
