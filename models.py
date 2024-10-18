import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F

from typing import Optional
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric_signed_directed.nn.directed import MagNetConv, complex_relu

class GAT_Net(nn.Module):
    def __init__(self, in_channels=17, out_channels=1, emb_hidden_dim=32,  hidden_dim = 8, heads = 2, drop_out=0.6,
                num_birthyear_disc=4, replace_missing_value='replace', rm_feature=-999):
        super().__init__()
        
        self.rm_feature = rm_feature
        self.by_disc = num_birthyear_disc if rm_feature!=0 else 0
        
        # self.conv1 = GATv2Conv(in_channels, 8, heads=8, dropout=0.6)
        self.conv1 = GATv2Conv(emb_hidden_dim, hidden_dim, heads)
        # On the Pubmed dataset, use heads=8 in conv2.
        # self.conv2 = GATv2Conv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim * heads, heads=1, concat=False)
        self.linear = nn.Linear(hidden_dim * heads, 1)

        ## birthyear embedding
        if rm_feature != 0:
            self.lookup_birth = nn.Embedding(self.by_disc, emb_hidden_dim).to('cuda:0')
        
        ## gender embedding
        if rm_feature != 1:
            self.lookup_gender = nn.Embedding(2, emb_hidden_dim).to('cuda:0')

        num_options = 3 if replace_missing_value == 'replace' else 2
        
        ## symptoms
        num_options = 3 if replace_missing_value == 'replace' else 2
        self.num_symp = 14 if rm_feature in range(2,17) else 15
        symp_embeddings = nn.ModuleList([nn.Embedding(num_options, emb_hidden_dim).to('cuda:0') for _ in range(self.num_symp)])
        for idx, emb in enumerate(symp_embeddings):
            setattr(self, f"lookup_a{idx+1}", emb)

        ## Dropout layer
        self.drop_out = drop_out



    def forward(self, x, edge_index):
        ## birthyear
        if self.rm_feature != 0:
            birth_embs = self.lookup_birth((x[:,:self.by_disc] == 1).nonzero(as_tuple=True)[1])
        
        ## gender
        if self.rm_feature != 1:
            gender_embs = self.lookup_gender(x[:,self.by_disc].to(torch.int))
        
        ## symptoms
        feature_embs = []
        for i in range(self.num_symp):
            emb = getattr(self, f"lookup_a{i+1}")
            # a_embs = emb(x[:, self.by_disc + i + 1].to(torch.int))
            a_embs = emb(x[:, -self.num_symp+i].to(torch.int))
            feature_embs.append(a_embs)
        feature_embs = torch.mean(torch.stack(feature_embs), axis=0)
        
        ## Averaging birthyear, gender, symptoms embedding
        if self.rm_feature == 0:
            x = torch.mean(torch.stack([gender_embs, feature_embs]), axis=0)    
        elif self.rm_feature == 1:
            x = torch.mean(torch.stack([birth_embs, feature_embs]), axis=0)
        else:
            x = torch.mean(torch.stack([birth_embs, gender_embs, feature_embs]), axis=0)
        
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        # return F.log_softmax(x, dim=-1)
        return x



class GCN_Net(nn.Module):
    def __init__(self, in_channels=17, out_channels=1, emb_hidden_dim=32, hidden_dim = 16, drop_out=0.6,
                num_birthyear_disc=4, replace_missing_value='replace', rm_feature=-999):
        super().__init__()
        
        self.rm_feature = rm_feature
        self.by_disc = num_birthyear_disc if rm_feature!=0 else 0
        
        self.conv1 = GCNConv(emb_hidden_dim, hidden_dim) # GCNConv 
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        
        ## birthyear embedding
        if rm_feature != 0:
            self.lookup_birth = nn.Embedding(self.by_disc, emb_hidden_dim).to('cuda:0')
        
        ## gender embedding
        if rm_feature != 1:
            self.lookup_gender = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
            
        ## symptoms
        num_options = 3 if replace_missing_value == 'replace' else 2
        self.num_symp = 14 if rm_feature in range(2,17) else 15
        symp_embeddings = nn.ModuleList([nn.Embedding(num_options, emb_hidden_dim).to('cuda:0') for _ in range(self.num_symp)])
        for idx, emb in enumerate(symp_embeddings):
            setattr(self, f"lookup_a{idx+1}", emb)

        ## dropout
        self.drop_out = drop_out



    def forward(self, x, edge_index):
        ## birthyear
        if self.rm_feature != 0:
            birth_embs = self.lookup_birth((x[:,:self.by_disc] == 1).nonzero(as_tuple=True)[1])
        
        ## gender
        if self.rm_feature != 1:
            gender_embs = self.lookup_gender(x[:,self.by_disc].to(torch.int))
        
        ## symptoms
        feature_embs = []
        for i in range(self.num_symp):
            emb = getattr(self, f"lookup_a{i+1}")
            # a_embs = emb(x[:, self.by_disc + i + 1].to(torch.int))
            a_embs = emb(x[:, -self.num_symp+i].to(torch.int))
            feature_embs.append(a_embs)
        feature_embs = torch.mean(torch.stack(feature_embs), axis=0)
        
        ## Averaging birthyear, gender, symptoms embedding
        if self.rm_feature == 0:
            x = torch.mean(torch.stack([gender_embs, feature_embs]), axis=0)    
        elif self.rm_feature == 1:
            x = torch.mean(torch.stack([birth_embs, feature_embs]), axis=0)
        else:
            x = torch.mean(torch.stack([birth_embs, gender_embs, feature_embs]), axis=0)
        
        
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        # return F.log_softmax(x, dim=-1)
        return x
    

class MagNet(nn.Module):
    def __init__(self, in_channels: int = 17, emb_hidden_dim=32, hidden: int = 16, q: float = 0.25, K: int = 2, 
                 activation: bool = False, trainable_q: bool = False, layer: int = 2, 
                 drop_out: float = False, normalization: str = 'sym', cached: bool = False, num_birthyear_disc=4, replace_missing_value='replace',
                 rm_feature=-999):
        super(MagNet, self).__init__()
        
        self.rm_feature = rm_feature
        self.by_disc = num_birthyear_disc if rm_feature!=0 else 0
        
        chebs = nn.ModuleList()
        chebs.append(MagNetConv(in_channels=emb_hidden_dim, out_channels=hidden, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu.complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MagNetConv(in_channels=hidden, out_channels=hidden, K=K,
                                    q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))

        self.Chebs = chebs

        self.Conv = nn.Conv1d(2* hidden, 2 * hidden, kernel_size=1)
        self.linear = nn.Linear(2 * hidden, 1)

        ## birthyear embedding
        if self.rm_feature != 0:
            self.lookup_birth = nn.Embedding(self.by_disc, emb_hidden_dim).to('cuda:0')
        
        ## gender embedding
        if self.rm_feature != 1:
            self.lookup_gender = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        
        ## symptoms
        num_options = 3 if replace_missing_value == 'replace' else 2
        self.num_symp = 14 if rm_feature in range(2,17) else 15
        symp_embeddings = nn.ModuleList([nn.Embedding(num_options, emb_hidden_dim).to('cuda:0') for _ in range(self.num_symp)])
        for idx, emb in enumerate(symp_embeddings):
            setattr(self, f"lookup_a{idx+1}", emb)
        
        ## dropout layer
        self.drop_out = drop_out


    def forward(self, x, edge_index: torch.LongTensor,
                edge_weight: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        ## birthyear
        if self.rm_feature!=0:
            birth_embs = self.lookup_birth((x[:,:self.by_disc] == 1).nonzero(as_tuple=True)[1])
        
        ## gender
        if self.rm_feature != 1:
            gender_embs = self.lookup_gender(x[:,self.by_disc].to(torch.int))
        
        ## symptoms
        feature_embs = []
        for i in range(self.num_symp):
            emb = getattr(self, f"lookup_a{i+1}")
            # a_embs = emb(x[:, self.by_disc + i + 1].to(torch.int))
            a_embs = emb(x[:, -self.num_symp+i].to(torch.int))
            feature_embs.append(a_embs)
        feature_embs = torch.mean(torch.stack(feature_embs), axis=0)
        
        ## Averaging birthyear, gender, symptoms embedding
        if self.rm_feature == 0 :
            x = torch.mean(torch.stack([gender_embs, feature_embs]), axis=0)
        elif self.rm_feature == 1:
            x = torch.mean(torch.stack([birth_embs, feature_embs]), axis=0)
        else:
            x = torch.mean(torch.stack([birth_embs, gender_embs, feature_embs]), axis=0)
        
        x = F.dropout(x, self.drop_out, training=self.training)
        real = x
        imag = x
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real, imag), dim=-1)

        if self.drop_out > 0:
            x = F.dropout(x, self.drop_out, training=self.training)
        
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute((0, 2, 1))
        x = self.linear(x)
        return x.squeeze()



class MLPRegressor(nn.Module):
    def __init__(self, input_size=17, emb_hidden_dim=32, hidden_size=64, output_size=1, drop_out=0.0, num_birthyear_disc=4, replace_missing_value='replace', rm_feature=-999):
        super().__init__()
        
        self.rm_feature = rm_feature
        self.by_disc = num_birthyear_disc if rm_feature!=0 else 0
        
        self.fc1 = nn.Linear(emb_hidden_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)

        ## birthyear embedding
        if rm_feature != 0:
            self.lookup_birth = nn.Embedding(self.by_disc, emb_hidden_dim).to('cuda:0')
            
        ## gender embedding
        if rm_feature != 1:
            self.lookup_gender = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        
        ## symptoms
        num_options = 3 if replace_missing_value == 'replace' else 2
        self.num_symp = 14 if rm_feature in range(2,17) else 15
        symp_embeddings = nn.ModuleList([nn.Embedding(num_options, emb_hidden_dim).to('cuda:0') for _ in range(self.num_symp)])
        for idx, emb in enumerate(symp_embeddings):
            setattr(self, f"lookup_a{idx+1}", emb)

        ### Dropout layer
        self.drop_out = nn.Dropout(drop_out)


    def forward(self, x):
        ## birthyear
        if self.rm_feature!=0:
            birth_embs = self.lookup_birth((x[:,:self.by_disc] == 1).nonzero(as_tuple=True)[1])
        
        ## gender
        if self.rm_feature!=1:
            gender_embs = self.lookup_gender(x[:,self.by_disc].to(torch.int))
            
        ## symptoms
        feature_embs = []
        for i in range(self.num_symp):
            emb = getattr(self, f"lookup_a{i+1}")
            # a_embs = emb(x[:, self.by_disc + i + 1].to(torch.int))
            a_embs = emb(x[:, -self.num_symp+i].to(torch.int))
            feature_embs.append(a_embs)
        feature_embs = torch.mean(torch.stack(feature_embs), axis=0)
        
        ## Averaging birthyear, gender, symptoms embedding
        if self.rm_feature == 0:
            x = torch.mean(torch.stack([gender_embs, feature_embs]), axis=0)    
        elif self.rm_feature == 1:
            x = torch.mean(torch.stack([birth_embs, feature_embs]), axis=0)
        else:
            x = torch.mean(torch.stack([birth_embs, gender_embs, feature_embs]), axis=0)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return x



class LinearRegression(torch.nn.Module):
    def __init__(self, out_channels=1, num_birthyear_disc=4, rm_feature=-999):
        super(LinearRegression, self).__init__()
        if rm_feature == 0:
            input_size = 16
        elif rm_feature in range(1, 17):
            input_size = 16 + num_birthyear_disc - 1
        else:
            input_size = 16 + num_birthyear_disc
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        return x