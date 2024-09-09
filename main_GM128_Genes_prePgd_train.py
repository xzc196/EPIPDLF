from torch.utils.data import DataLoader
from dataset_embedding import MyDataset
from dataset_embedding import IDSDataset
from model_dnabert2_embedding_small import EPIModel
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR
from epi_dataset_new import EPIGeneDataset
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset



BATCH_SIZE = 512
epoch = 50
pre_train_epoch = 10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_num_correct(preds, labels):
    predictions = (preds >= 0.5).float()
    
    return (predictions == labels).sum().item()



def load_dna_sequences(file_path):
    # 读取DNA序列文件
    with open(file_path, 'r') as file:
        # 逐行读取文件并保存为字符串列表
        sequences = [line.strip() for line in file]

    # 将字符串列表转换为NumPy数组
    sequences_array = np.array(sequences)

    return sequences_array

def load_labels(file_path):
    # 读取标签文件
    with open(file_path, 'r') as file:
        # 逐行读取文件并保存为整数列表
        labels = [int(line.strip()) for line in file]

    # 将整数列表转换为NumPy数组
    labels_array = np.array(labels)

    return labels_array

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        # 获取对应索引的样本
        enhancer_sequence, promoter_sequence, label = self.dataset1[index]
        gene_data  = self.dataset2[index]

        # 返回两个数据集样本的组合
        return enhancer_sequence, promoter_sequence, gene_data, label

    def __len__(self):
        # 返回数据集的长度（假设两个数据集长度相同）
        return len(self.dataset1)


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name1='embedding_en.', emb_name2='embedding_pr', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)


    def restore(self, emb_name1='embedding_en', emb_name2='embedding_pr'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='embedding_en', emb_name2='embedding_pr'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name1='embedding_en', emb_name2='embedding_pr'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def visualize_with_tsne(self_attention_output, labels, file_path):
    # 将 self_attention_output 转换为 NumPy 数组
    self_attention_np = self_attention_output.cpu().detach().numpy()

    # 将 labels 转换为 NumPy 数组，并将其调整为一维数组
    labels_np = labels.view(-1).cpu().detach().numpy()

    # 创建 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)

    # 使用 t-SNE 对 self_attention_output 进行降维
    embedded_points = tsne.fit_transform(self_attention_np)

    # 根据标签分割正样本和负样本的坐标
    positive_points = embedded_points[labels_np == 1]
    negative_points = embedded_points[labels_np == 0]

    # 绘制散点图，使用不同颜色表示正样本和负样本
    plt.scatter(positive_points[:, 0], positive_points[:, 1], c='red', label='Positive')
    plt.scatter(negative_points[:, 0], negative_points[:, 1], c='blue', label='Negative')

    # 添加图例和标题
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig(file_path)
    # 显示图形
    plt.show()

def val_forwrd(model, dataloader):
    model.eval()
    test_epoch_loss = 0.0
    test_epoch_correct = 0
    test_epoch_preds = torch.tensor([]).to(device)
    test_epoch_target = torch.tensor([]).to(device)
    all_attention_outputs = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            enhancer_ids, promoter_ids, gene_data, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            gene_data = gene_data.to(device)
            labels = labels.to(device)
                
                
            outputs, _ = model(enhancer_ids, promoter_ids, gene_data)
            labels = labels.unsqueeze(1).float()
            test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

            if labels.shape == torch.Size([1, 1]):
                labels = torch.reshape(labels, (1,))
                
            loss = model.criterion(outputs, labels)
            test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
            test_epoch_loss += loss.item()
            test_epoch_correct += get_num_correct(outputs, labels)
                
        
    #visualize_with_tsne(torch.cat(all_attention_outputs, dim=0), torch.cat(all_labels, dim=0), "test_" + str(i) + ".png")
    test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    return test_epoch_loss, test_epoch_aupr, test_epoch_auc


#得到的是序列数据
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

'''
enhaners = load_dna_sequences('enhancers/NHEK_enhancers_nochr.fasta')
promoters = load_dna_sequences('promoters/NHEK_promoters_nochr.fasta')
Labels = load_labels('labels/NHEKlabels.txt')
'''


'''
enhaners = load_dna_sequences('enhancers/enhancers_test.fasta')
promoters = load_dna_sequences('promoters/promoters_test.fasta')
Labels = load_labels('labels/labels_test.txt')
'''

EPIGeneDataset
enhancers_file =  'HUVEC_enhancer_nochr_train.fasta'
enhancers_train = load_dna_sequences('enhancer2/' + enhancers_file)

enhancers_test_GM12878 = load_dna_sequences('enhancer2/GM12878_enhancer_nochr_test.fasta')
enhancers_test_IMR90 = load_dna_sequences('enhancer2/IMR90_enhancer_nochr_test.fasta')
enhancers_test_HeLa = load_dna_sequences('enhancer2/HeLa_enhancer_nochr_test.fasta')
enhancers_test_HUVEC = load_dna_sequences('enhancer2/HUVEC_enhancer_nochr_test.fasta')
enhancers_test_K562 = load_dna_sequences('enhancer2/K562_enhancer_nochr_test.fasta')
enhancers_test_NHEK = load_dna_sequences('enhancer2/NHEK_enhancer_nochr_test.fasta')

promoters_file =  'HUVEC_promoter_nochr_train.fasta'
promoters_train = load_dna_sequences('promoter2/' + promoters_file)

promoters_test_GM12878 = load_dna_sequences('promoter2/GM12878_promoter_nochr_test.fasta')
promoters_test_IMR90 = load_dna_sequences('promoter2/IMR90_promoter_nochr_test.fasta')
promoters_test_HeLa = load_dna_sequences('promoter2/HeLa_promoter_nochr_test.fasta')
promoters_test_HUVEC = load_dna_sequences('promoter2/HUVEC_promoter_nochr_test.fasta')
promoters_test_K562 = load_dna_sequences('promoter2/K562_promoter_nochr_test.fasta')
promoters_test_NHEK = load_dna_sequences('promoter2/NHEK_promoter_nochr_test.fasta')

Labels_file = 'HUVEC_label_train.txt'
Labels_train = load_labels('label2/' + Labels_file)

Labels_test_GM12878 = load_labels('label2/GM12878_label_test.txt')
Labels_test_IMR90 = load_labels('label2/IMR90_label_test.txt')
Labels_test_HeLa = load_labels('label2/HeLa_label_test.txt')
Labels_test_HUVEC = load_labels('label2/HUVEC_label_test.txt')
Labels_test_K562 = load_labels('label2/K562_label_test.txt')
Labels_test_NHEK = load_labels('label2/NHEK_label_test.txt')

val_fold_GM12878 = MyDataset(enhancers_test_GM12878, promoters_test_GM12878, Labels_test_GM12878)
val_fold_GM12878 = IDSDataset(val_fold_GM12878)
val_fold_IMR90 = MyDataset(enhancers_test_IMR90, promoters_test_IMR90, Labels_test_IMR90)
val_fold_IMR90 = IDSDataset(val_fold_IMR90)
val_fold_HeLa = MyDataset(enhancers_test_HeLa, promoters_test_HeLa, Labels_test_HeLa)
val_fold_HeLa = IDSDataset(val_fold_HeLa)
val_fold_HUVEC = MyDataset(enhancers_test_HUVEC, promoters_test_HUVEC, Labels_test_HUVEC)
val_fold_HUVEC = IDSDataset(val_fold_HUVEC)
val_fold_K562 = MyDataset(enhancers_test_K562, promoters_test_K562, Labels_test_K562)
val_fold_K562 = IDSDataset(val_fold_K562)
val_fold_NHEK = MyDataset(enhancers_test_NHEK, promoters_test_NHEK, Labels_test_NHEK)
val_fold_NHEK = IDSDataset(val_fold_NHEK)

file_path = "./"
#checkpoint = torch.load(file_path)

#epimodel = torch.load(file_path)

#checkpoint = torch.load(file_path)
#epimodel.load_state_dict(checkpoint)

epimodel_file = "premodel_DNABERT1_model_pgd_gene_new_19.pt"
epimodel = torch.load(file_path + epimodel_file)
print(epimodel_file)
epimodel.to(device)

for param_group in epimodel.optimizer.param_groups:
        param_group['lr'] = 1e-3

scheduler = MultiStepLR(epimodel.optimizer, milestones=[25], gamma=0.1)
#fgm = FGM(epimodel)
pgd = PGD(epimodel)
K = 3
#epimodel.init_weights()

all_params_on_gpu = all(param.is_cuda for param in epimodel.parameters())
print("gpu是否可用：",torch.cuda.is_available())
print("模型是否在GPU上：:",all_params_on_gpu)

print("增强子训练文件：", enhancers_file)
print("启动子训练文件：", promoters_file)
print("标签训练文件：", Labels_file)
    

train_fold = MyDataset(enhancers_train, promoters_train, Labels_train)
train_fold = IDSDataset(train_fold) #IDSDataset转化为token_id
train_fold_gene = EPIGeneDataset(
            enh_datasets="../data/BENGI/HUVEC_enhancer.bed",
            pro_datasets="../data/BENGI/HUVEC_promoter.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "HUVEC",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        ) 
val_fold_gene_GM12878 = EPIGeneDataset(
            enh_datasets="../data/BENGI/GM12878_enhancer_test.bed",
            pro_datasets="../data/BENGI/GM12878_promoter_test.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "GM12878",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        )
val_fold_gene_IMR90 = EPIGeneDataset(
            enh_datasets="../data/BENGI/IMR90_enhancer_test.bed",
            pro_datasets="../data/BENGI/IMR90_promoter_test.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "IMR90",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        )
val_fold_gene_HeLa = EPIGeneDataset(
            enh_datasets="../data/BENGI/HeLa_enhancer_test.bed",
            pro_datasets="../data/BENGI/HeLa_promoter_test.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "HeLa",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        )
val_fold_gene_HUVEC = EPIGeneDataset(
            enh_datasets="../data/BENGI/HUVEC_enhancer_test.bed",
            pro_datasets="../data/BENGI/HUVEC_promoter_test.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "HUVEC",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        )
val_fold_gene_K562 = EPIGeneDataset(
            enh_datasets="../data/BENGI/K562_enhancer_test.bed",
            pro_datasets="../data/BENGI/K562_promoter_test.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "K562",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        )
val_fold_gene_NHEK = EPIGeneDataset(
            enh_datasets="../data/BENGI/NHEK_enhancer_test.bed",
            pro_datasets="../data/BENGI/NHEK_promoter_test.bed",
            feats_config="../data/genomic_data/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            cell = "NHEK",
            enh_seq_len=3000,
            pro_seq_len=2000,
            bin_size=500
        )




train_fold = CombinedDataset(train_fold, train_fold_gene)

val_fold_GM12878 = CombinedDataset(val_fold_GM12878, val_fold_gene_GM12878)
val_fold_IMR90 = CombinedDataset(val_fold_IMR90, val_fold_gene_IMR90)
val_fold_HeLa = CombinedDataset(val_fold_HeLa, val_fold_gene_HeLa)
val_fold_HUVEC = CombinedDataset(val_fold_HUVEC, val_fold_gene_HUVEC)
val_fold_K562 = CombinedDataset(val_fold_K562, val_fold_gene_K562)
val_fold_NHEK = CombinedDataset(val_fold_NHEK, val_fold_gene_NHEK)



'''
val_fold_IMR90 = MyDataset(enhancers_test_IMR90, promoters_test_IMR90, Labels_test_IMR90)
val_fold_IMR90 = IDSDataset(val_fold_IMR90)
val_fold_HeLa = MyDataset(enhancers_test_HeLa, promoters_test_HeLa, Labels_test_HeLa)
val_fold_HeLa = IDSDataset(val_fold_HeLa)
val_fold_HUVEC = MyDataset(enhancers_test_HUVEC, promoters_test_HUVEC, Labels_test_HUVEC)
val_fold_HUVEC = IDSDataset(val_fold_HUVEC)
val_fold_K562 = MyDataset(enhancers_test_K562, promoters_test_K562, Labels_test_K562)
val_fold_K562 = IDSDataset(val_fold_K562)
val_fold_NHEK = MyDataset(enhancers_test_NHEK, promoters_test_NHEK, Labels_test_NHEK)
val_fold_NHEK = IDSDataset(val_fold_NHEK)
'''

train_num = len(train_fold)
train_loader = DataLoader(dataset=train_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)

val_num = len(val_fold_GM12878)
val_fold_GM12878 = DataLoader(dataset=val_fold_GM12878, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)

val_num = len(val_fold_IMR90)
val_fold_IMR90 = DataLoader(dataset=val_fold_IMR90, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)

val_num = len(val_fold_HeLa)
val_fold_HeLa = DataLoader(dataset=val_fold_HeLa, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)

val_num = len(val_fold_HUVEC)
val_fold_HUVEC = DataLoader(dataset=val_fold_HUVEC, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)

val_num = len(val_fold_K562)
val_fold_K562 = DataLoader(dataset=val_fold_K562, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)

val_num = len(val_fold_NHEK)
val_fold_NHEK = DataLoader(dataset=val_fold_NHEK, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)


'''
enhancers_pretrain = load_dna_sequences('enhancers/enhancers_test.fasta')
promoters_pretrain = load_dna_sequences('promoters/promoters_test.fasta')
Labels_pretrain = load_labels('labels/labels_test.txt')
'''


'''
prepoch = 15

enhancers_pretrain = load_dna_sequences('enhancer2/all_enhancer_train.fasta')
promoters_pretrain = load_dna_sequences('promoter2/all_promoter_train.fasta')
Labels_pretrain = load_labels('label2/all_labels_train.txt')




train_prefold = MyDataset(enhancers_pretrain, promoters_pretrain, Labels_pretrain)
train_prefold = IDSDataset(train_prefold) #IDSDataset转化为token_id
train_preloader = DataLoader(dataset=train_prefold, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=32)
'''


'''
for i in range(prepoch):
        
        epimodel.train()
        train_epoch_loss = 0.0
        train_epoch_correct = 0
        train_epoch_preds = torch.tensor([]).to(device)
        train_epoch_target = torch.tensor([]).to(device)
        for data in train_preloader:
            enhancer_ids, promoter_ids, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            labels = labels.to(device)
            
            outputs, _ = epimodel(enhancer_ids, promoter_ids)
            labels = labels.unsqueeze(1).float()
            
            train_epoch_preds = torch.cat((train_epoch_preds, outputs.view(-1)))
            train_epoch_target = torch.cat((train_epoch_target, labels.view(-1)))
            
            if labels.shape == torch.Size([1, 1]):
              labels = torch.reshape(labels, (1,))

            loss = epimodel.criterion(outputs, labels)
            train_epoch_loss += loss.item()
            train_epoch_correct += get_num_correct(outputs, labels)


            loss.backward()
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    epimodel.zero_grad()
                else:
                    pgd.restore_grad()
            
                outputs_adv, _ = epimodel(enhancer_ids, promoter_ids)
                loss_adv = epimodel.criterion(outputs_adv, labels)
                loss_adv.backward()
            
            pgd.restore() # 恢复embedding参数
            
            epimodel.optimizer.step()
            epimodel.optimizer.zero_grad()
             
            
        scheduler.step()
        print("learning_rate:", scheduler.get_last_lr())
        train_epoch_aupr = average_precision_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
        train_epoch_auc = roc_auc_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
            
        torch.save(epimodel, file_path + 'premodel_PGD_model_' + str(i) + '.pt')
        print("pre_train epoch: {}, train_loss：{}, train_aupr：{}, train_auc: {}".format(i + 1, train_epoch_loss, train_epoch_aupr, train_epoch_auc))


        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_IMR90)
        print("pre_train IMR90 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_GM12878)
        print("pre_train GM12878 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_HeLa)
        print("pre_train HeLa epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_HUVEC)
        print("pre_train HUVEC epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_K562)
        print("pre_train K562 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_NHEK)
        print("pre_train NHEK epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))


'''





for i in range(epoch):
        
        epimodel.train()
        train_epoch_loss = 0.0
        train_epoch_correct = 0
        train_epoch_preds = torch.tensor([]).to(device)
        train_epoch_target = torch.tensor([]).to(device)
        all_attention_outputs = []
        all_labels = []
        for data in train_loader:
            enhancer_ids, promoter_ids, gene_data, labels = data
            enhancer_ids = enhancer_ids.to(device)
            gene_data = gene_data.to(device)
            promoter_ids = promoter_ids.to(device)
            labels = labels.to(device)
            
            outputs, emd = epimodel(enhancer_ids, promoter_ids, gene_data)
            
            all_attention_outputs.append(emd)

            labels = labels.unsqueeze(1).float()
            all_labels.append(labels)
            
            train_epoch_preds = torch.cat((train_epoch_preds, outputs.view(-1)))
            train_epoch_target = torch.cat((train_epoch_target, labels.view(-1)))
            
            if labels.shape == torch.Size([1, 1]):
              labels = torch.reshape(labels, (1,))

            loss = epimodel.criterion(outputs, labels)
            train_epoch_loss += loss.item()
            train_epoch_correct += get_num_correct(outputs, labels)

            
            loss.backward()
            
            '''
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    epimodel.zero_grad()
                else:
                    pgd.restore_grad()
            
                #outputs_adv = epimodel(enhancer_ids, promoter_ids)
                outputs_adv, emd = epimodel(enhancer_ids, promoter_ids, gene_data)
                loss_adv = epimodel.criterion(outputs_adv, labels)
                loss_adv.backward()
            
            pgd.restore() # 恢复embedding参数
            '''


            epimodel.optimizer.step()
            epimodel.optimizer.zero_grad()
            
        
        #visualize_with_tsne(torch.cat(all_attention_outputs, dim=0), torch.cat(all_labels, dim=0), "train_" + str(i) + ".png") 
        scheduler.step()
        print("learning_rate:", scheduler.get_last_lr())
        train_epoch_aupr = average_precision_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
        train_epoch_auc = roc_auc_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
          
        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_GM12878)
        torch.save(epimodel, file_path + 'DNABERT1_pgd_genes_HUVEC_train_model_lr1e-3' + str(i) + '.pt')
        print("fine_tining GM12878 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))
        
        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_IMR90)
        print("fine_tining IMR90 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_HeLa)
        print("fine_tining HeLa epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_HUVEC)
        print("fine_tining HUVEC epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_K562)
        print("fine_tining K562 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_fold_NHEK)
        print("fine_tining NHEK epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))
        
   
        
        

'''
        epimodel.eval()
        test_epoch_loss = 0.0
        test_epoch_correct = 0
        test_epoch_preds = torch.tensor([]).to(device)
        test_epoch_target = torch.tensor([]).to(device)
        with torch.no_grad():
            for data in val_fold_GM12878:
                enhancer_ids, promoter_ids, labels = data
                enhancer_ids = enhancer_ids.to(device)
                promoter_ids = promoter_ids.to(device)
                labels = labels.to(device)
                
                
                outputs = epimodel(enhancer_ids, promoter_ids)
                labels = labels.unsqueeze(1).float()
                test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

                if labels.shape == torch.Size([1, 1]):
                  labels = torch.reshape(labels, (1,))
                
                loss = epimodel.criterion(outputs, labels)
                test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
                test_epoch_loss += loss.item()
                test_epoch_correct += get_num_correct(outputs, labels)
        

        test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        
        print("GM12878 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))


        epimodel.eval()
        test_epoch_loss = 0.0
        test_epoch_correct = 0
        test_epoch_preds = torch.tensor([]).to(device)
        test_epoch_target = torch.tensor([]).to(device)
        with torch.no_grad():
            for data in val_fold_HeLa:
                enhancer_ids, promoter_ids, labels = data
                enhancer_ids = enhancer_ids.to(device)
                promoter_ids = promoter_ids.to(device)
                labels = labels.to(device)
                
                
                outputs = epimodel(enhancer_ids, promoter_ids)
                labels = labels.unsqueeze(1).float()
                test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

                if labels.shape == torch.Size([1, 1]):
                  labels = torch.reshape(labels, (1,))
                
                loss = epimodel.criterion(outputs, labels)
                test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
                test_epoch_loss += loss.item()
                test_epoch_correct += get_num_correct(outputs, labels)
        

        test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        
        print("HeLa epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))



        epimodel.eval()
        test_epoch_loss = 0.0
        test_epoch_correct = 0
        test_epoch_preds = torch.tensor([]).to(device)
        test_epoch_target = torch.tensor([]).to(device)
        with torch.no_grad():
            for data in val_fold_HUVEC:
                enhancer_ids, promoter_ids, labels = data
                enhancer_ids = enhancer_ids.to(device)
                promoter_ids = promoter_ids.to(device)
                labels = labels.to(device)
                
                
                outputs = epimodel(enhancer_ids, promoter_ids)
                labels = labels.unsqueeze(1).float()
                test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

                if labels.shape == torch.Size([1, 1]):
                  labels = torch.reshape(labels, (1,))
                
                loss = epimodel.criterion(outputs, labels)
                test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
                test_epoch_loss += loss.item()
                test_epoch_correct += get_num_correct(outputs, labels)
        

        test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        
        print("HUVEC epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))



        epimodel.eval()
        test_epoch_loss = 0.0
        test_epoch_correct = 0
        test_epoch_preds = torch.tensor([]).to(device)
        test_epoch_target = torch.tensor([]).to(device)
        with torch.no_grad():
            for data in val_fold_K562:
                enhancer_ids, promoter_ids, labels = data
                enhancer_ids = enhancer_ids.to(device)
                promoter_ids = promoter_ids.to(device)
                labels = labels.to(device)
                
                
                outputs = epimodel(enhancer_ids, promoter_ids)
                labels = labels.unsqueeze(1).float()
                test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

                if labels.shape == torch.Size([1, 1]):
                  labels = torch.reshape(labels, (1,))
                
                loss = epimodel.criterion(outputs, labels)
                test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
                test_epoch_loss += loss.item()
                test_epoch_correct += get_num_correct(outputs, labels)
        

        test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        
        print("K562 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))



        epimodel.eval()
        test_epoch_loss = 0.0
        test_epoch_correct = 0
        test_epoch_preds = torch.tensor([]).to(device)
        test_epoch_target = torch.tensor([]).to(device)
        with torch.no_grad():
            for data in val_fold_NHEK:
                enhancer_ids, promoter_ids, labels = data
                enhancer_ids = enhancer_ids.to(device)
                promoter_ids = promoter_ids.to(device)
                labels = labels.to(device)
                
                
                outputs = epimodel(enhancer_ids, promoter_ids)
                labels = labels.unsqueeze(1).float()
                test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

                if labels.shape == torch.Size([1, 1]):
                  labels = torch.reshape(labels, (1,))
                
                loss = epimodel.criterion(outputs, labels)
                test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
                test_epoch_loss += loss.item()
                test_epoch_correct += get_num_correct(outputs, labels)
        

        test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
        
        print("NHEK epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

print(device)
'''