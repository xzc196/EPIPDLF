from torch import nn
import torch
import numpy as np



NUMBER_WORDS = 4097
NUMBER_POS = 70


#网络参数
EMBEDDING_DIM = 768
CNN_KERNEL_SIZE = 40
POOL_KERNEL_SIZE = 20
OUT_CHANNELs = 64
LEARNING_RATE = 1e-3


embedding_matrix = torch.tensor(np.load('dnabert1_matrix.npy'), dtype=torch.float32)

class EPIModel(nn.Module):
    def __init__(self):
        super(EPIModel, self).__init__()

        #embedding
        self.embedding_en = nn.Embedding(4097, 768)
        self.embedding_pr = nn.Embedding(4097, 768)

        self.embedding_en.weight = nn.Parameter(embedding_matrix)
        self.embedding_pr.weight = nn.Parameter(embedding_matrix)

        self.embedding_en.requires_grad = True
        self.embedding_pr.requires_grad = True

        
        #使用一维卷积操作
        self.enhancer_sequential = nn.Sequential(nn.Conv1d(in_channels=768, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
                                                 nn.BatchNorm1d(64),
                                                 nn.Dropout(p=0.5)
                                                 )
        self.promoter_sequential = nn.Sequential(nn.Conv1d(in_channels=768, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
                                                 nn.BatchNorm1d(64),
                                                 nn.Dropout(p=0.5)
                                                 )
        

        self.linear_layer = nn.Linear(50, 64)
        self.l1GRU = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, num_layers=2)
        self.l2GRU = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, num_layers=2)

        self.MTHEAD = nn.MultiheadAttention(embed_dim=64,num_heads=8)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4)
        # self.transformerencoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.batchnorm1d = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(nn.Linear(245 * 64, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(128, 1),           
               )
   

        
        
        
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
        
        
        
        #使用二维度卷积操作
        '''
        self.enhancer_sequential = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=40),
                                                 nn.MaxPool2d(20)
                )
        
        self.promoter_sequential = nn.Sequential(
                )
        '''

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight)
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
        
        
    def forward(self, enhancer_ids, promoter_ids, gene_data):
        SAMPLE_SIZE = enhancer_ids.size(0)

        enhancer_embedding = self.embedding_en(enhancer_ids)
        promoter_embedding = self.embedding_pr(promoter_ids)

        gene_data = self.linear_layer(gene_data)
        gene_data = torch.unsqueeze(gene_data, dim=0)
 
        #一维卷积需要对数据变形
        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))

        
        enhancers_output, _ = self.l1GRU(enhancers_output.permute(2, 0, 1))
        promoters_output, _ = self.l2GRU(promoters_output.permute(2, 0, 1))
        
        # stacked_tensor = torch.cat((enhancers_output, promoters_output), dim=1).transpose(0, 1)
        # output = self.transformerencoder(stacked_tensor).transpose(0, 1).flatten(start_dim=1)
        # output = self.batchnorm1d(output)
        # output = self.dropout(output)

        enhancers_output, _ = self.MTHEAD(enhancers_output, enhancers_output, enhancers_output)
        promoters_output, _ = self.MTHEAD(promoters_output, promoters_output, promoters_output)
        gene_data, _ = self.MTHEAD(gene_data, gene_data, gene_data)

        stacked_tensor = torch.cat((enhancers_output, promoters_output), dim=0)
        stacked_tensor = torch.cat((stacked_tensor, gene_data), dim=0).permute(1, 2, 0)
        output = self.batchnorm1d(stacked_tensor)
        output = self.dropout(output)

        
        #print(enhancers_output.shape)
        #加上CLS
        result = self.fc(output.flatten(start_dim=1))
        # result = self.fc(output)
        
        
        return torch.sigmoid(result), output.flatten(start_dim=1)