import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data as Data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
################################################################
train = np.load("./data_nature/new_cell_data.npy")
train = train[:1000,:]
label = np.load("./data_nature/label_chirp.npy")
label = label.T[:1000,:]

X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.3, random_state=50)

X_train_final = torch.from_numpy(X_train.astype(np.float32))
y_train_final = torch.from_numpy(y_train.astype(np.float32))
X_test_final = torch.from_numpy(X_test.astype(np.float32))
y_test_final = torch.from_numpy(y_test.astype(np.float32))

train_data = Data.TensorDataset(X_train_final, y_train_final)
test_data = Data.TensorDataset(X_test_final, y_test_final)
train_loader = Data.DataLoader(dataset=train_data, batch_size=16,
                               shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=16,
                               shuffle=True, num_workers=0)


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(input_size+1000, hidden_size)
        self.position_encoding = nn.Embedding(input_size+1000, hidden_size)
        
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, tgt):
        src_embedded = self.embedding(src.long()) + self.position_encoding(src.long())
        tgt_embedded = self.embedding(tgt.long()) + self.position_encoding(tgt.long())
        
        src_embedded = src_embedded.permute(1, 0, 2)  # Shape: (src_len, batch_size, hidden_size)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)  # Shape: (tgt_len, batch_size, hidden_size)
        
        output = self.transformer(src_embedded, tgt_embedded)
        
        output = output.permute(1, 0, 2)  # Shape: (batch_size, tgt_len, hidden_size)
        output = self.fc(output)
        
        return output
    
########################################################################
input_dim = 1503 
output_dim = 249  
# seq_length = 1503 
hidden_dim = 512
num_layers = 4 
num_heads = 8  
# max_seq_length = seq_length
dropout = 0.2

########################################################################
def dis_train_trans(rank, world_size):
    # 设置分布式训练环境
    dist.init_process_group("nccl",init_method='env://', rank=rank, world_size=world_size)
    
    model_test = TransformerModel(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=0)
    ### use multi-GPU training
    model_test = DDP(model_test.to(rank))
    optimizer = torch.optim.Adam(model_test.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses = []
    num_epochs = 100
    
    # training model
    
    for epoch in range(num_epochs):
        model_test.train()
        epoch_loss = 0.0
    
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(rank), targets.to(rank)
            optimizer.zero_grad()
            output = model_test(inputs, inputs)
            # print("a:", output[:,0,:].size(), "b:", targets.size())
            # print("out:", output[:,0,:], "target:", targets)
            loss = criterion(output[:,0,:], targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
