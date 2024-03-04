import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from numpy import mean
import pandas as pd
import numpy as np
# customerized modules
from output import BertRepeatModel
from gu import ItemData

data_dir = r"code/data/"    # 数据文件夹

# 数据集的处理
# 训练数据集的处理，得到S_u,T_u,C_u序列
def convert_unique_idx(df, column_name,cla_dic=False):
    if cla_dic:
        column_dict = cla_dic
    else:
        column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    #assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

# 非常恶劣的代码风格，将绝对路径嵌在函数里面
# def Get_user_gp(cla_dic=False,file_dir = r"D:\学习\研究生任务\子璐论文\code\code&data\数据集切割\sampledata" ,test = 10):
def Get_user_gp(cla_dic=False,file_dir = r"../../data/sampledata" ,test = 10):
    df_train=pd.DataFrame()
    data_num = 10
    for i in range(data_num):  #合并数据子集形成训练数据集
        if i+1 != test:
            file_home=file_dir + str(i + 1) + ".csv"
            df = pd.read_csv(file_home, encoding="gbk")  # 读取数据
            df_train=df_train._append(df,ignore_index=True)
    df_train_sort = df_train.sort_values(by ='createtime')
    df_train_sort,cla_mapping = convert_unique_idx(df_train_sort,'classify',cla_dic)
    print(cla_mapping)
    df_train_sort['Time']= df_train_sort.createtime.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp())
    df_train_sort['createtime'] = pd.to_datetime(df_train_sort.createtime)#time stamp
    df_train_sort['month'] = df_train_sort.createtime.dt.month #apply(lambda row:row.month,1)
    df_train_sort['day'] = df_train_sort.createtime.dt.day #apply(lambda row:row.day,1)
    df_train_sort['weekday'] = df_train_sort.createtime.dt.weekday #.apply(lambda row:row.weekday(),1)
    df_train_sort['hour'] = df_train_sort.createtime.dt.hour #.apply(lambda row:row.hour,1)
    user_gp_item = df_train_sort.groupby(['user'])['item'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_t = df_train_sort.groupby(['user'])['Time'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_month = df_train_sort.groupby(['user'])['month'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_day = df_train_sort.groupby(['user'])['day'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_weekday = df_train_sort.groupby(['user'])['weekday'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_hour = df_train_sort.groupby(['user'])['hour'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_cla = df_train_sort.groupby(['user'])['classify'].agg({list}).reset_index()['list'].values.tolist()
    return user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla#,user_gp_hour

def Test_user_gp(cla_dic=False,file_dir = r"../../data/sampledata",data_num = 10):
    print('Testing user group on test-set:',data_num)
    df_train=pd.DataFrame()
    file_home=file_dir + str(data_num) + ".csv"
    df = pd.read_csv(file_home, encoding="gbk")  # 读取数据
    df_train=df_train._append(df,ignore_index=True)
    df_train_sort = df_train.sort_values(by ='createtime')
    df_train_sort,cla_mapping = convert_unique_idx(df_train_sort,'classify',cla_dic)
    print(cla_mapping)
    df_train_sort['Time']= df_train_sort.createtime.apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp())
    df_train_sort['createtime'] = pd.to_datetime(df_train_sort.createtime)
    df_train_sort['month'] = df_train_sort.createtime.dt.month #apply(lambda row:row.month,1)
    df_train_sort['day'] = df_train_sort.createtime.dt.day #.apply(lambda row:row.day,1)
    df_train_sort['weekday'] = df_train_sort.createtime.dt.weekday #apply(lambda row:row.weekday(),1)
    df_train_sort['hour'] = df_train_sort.createtime.dt.hour #apply(lambda row:row.hour,1)
    user_gp_item = df_train_sort.groupby(['user'])['item'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_t = df_train_sort.groupby(['user'])['Time'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_month = df_train_sort.groupby(['user'])['month'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_day = df_train_sort.groupby(['user'])['day'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_weekday = df_train_sort.groupby(['user'])['weekday'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_hour = df_train_sort.groupby(['user'])['hour'].agg({list}).reset_index()['list'].values.tolist() # 以USER为维度聚合后得到每一用户的商品购买序列
    user_gp_cla = df_train_sort.groupby(['user'])['classify'].agg({list}).reset_index()['list'].values.tolist()
    return user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla


def get_loss(model, batch, device, cfg):  # make sure loss is tensor
    x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch  # masked_pos其实好像没啥用
    x = x.to(torch.int64).to(device)
    stamp = stamp.to(torch.int64).to(device)
    # gap = gap.to(torch.int64)
    masked_ids = masked_ids.to(torch.int64).to(device)
    masked_pos = masked_pos.to(torch.int64).to(device)
    time_matrix = time_matrix.to(torch.int64).to(device)
    time_gap = time_gap.to(torch.int64).to(device)
    cla = cla.to(torch.int64).to(device)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    logits_lm = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, 'train')  # [batch,pre_num,vocab_num] forwawrd
    loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
    loss_lm = (loss_lm * masked_weights.float()).mean()

    return logits_lm, loss_lm

def train(user_gp, user_gp_test, user_gp_stamp, user_gp_stamp_test,user_gp_stamp_t, user_gp_stamp_t_test, user_gp_cla,user_gp_cla_test, vocabulary, cfg):
    dataset = ItemData(user_gp, user_gp_stamp, user_gp_stamp_t, user_gp_cla, vocabulary, cfg, 'train')
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    dataset_test = ItemData(user_gp_test, user_gp_stamp_test, user_gp_stamp_t_test, user_gp_cla_test, vocabulary, cfg,'test')
    loader_test = DataLoader(dataset_test, batch_size=cfg['batch_size'] * 10, shuffle=True)
    torch.manual_seed(cfg['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertRepeatModel(cfg,device)
    if cfg['pre_train']:
        model.transformer = torch.load(cfg['pre_train']).transformer
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    
    
    #if token embed block needs L2 norm
    # model_else = []
    # for name, p in model.named_parameters():
    #     if name != 'transformer.embed.tok_embed.weight':
    #         model_else += [p] 
    # optim = torch.optim.Adam([{'params':model.transformer.embed.tok_embed.parameters(),'weight_decay':cfg['weight_decay']},{'params':model_else}],lr=cfg['lr'])
    
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])  # 也可以换其它的优化器
    loss_total = []
    loss_per_eps = []
    recall_total = []
    t1 = time.time()
    for i in range(cfg['iter_n']):
        model.train()
        count_iter = 0
        # t1 = time.time()
        for batch_idx, batch in enumerate(loader):
            optim.zero_grad()
            logits, loss = get_loss(model, batch, device, cfg)
            loss.backward()
            optim.step()
            loss_total.append(loss.detach().numpy())
            count_iter += 1
            if batch_idx % 50 == 0:
                print(
                    "Epoch: ", i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                )
                # t2 = time.time()-t1
                # t1 = time.time()
                # print('用时：',t2)
        loss_per_eps.append(sum(loss_total[-count_iter:]) / count_iter)
        print("| loss_per_eps: %.3f" % loss_per_eps[-1])
        if Config['test_while_train']:
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.cuda() if torch.cuda.is_available() else model.cpu()
            # if torch.cuda.is_available():
            #     print("GPU test avaliable")
            #     device = torch.device("cuda")
            #     model = model.cuda()
            # else:
            #     device = torch.device("cpu")
            #     model = model.cpu()
            average = 0
            correct_range = cfg['correct_range']
            # new
            for ctt in range(cfg['test_times']):
                ans = test(loader_test, model, correct_range, device)
                average += ans
                ans_test = test(loader_test, model, correct_range, device)
            print('----------------- Average Accuracy:', average / cfg['test_times'], '---------------------')
            recall_total.append(average / cfg['test_times'])
            if cfg['save']:
                torch.save(model, cfg['save_dir'] + 'repeat_res_int_dis' + str(i) + str(ans_test)[2:] + '.model')
                print('Model saved to', cfg['save_dir'] + 'repeat_res_int_dis' + str(i) + str(ans_test)[2:] + '.model')

    print('use time:', time.time() - t1)

    # 画loss图
    plt.plot(loss_total)
    plt.show()
    plt.plot(loss_per_eps)
    plt.show()
    print('Mean Recall after 15 epochs:', mean(recall_total[15:]))
    print(recall_total)

def test(loader_test, model, correct_range, device):
    hits_all = 0
    test_num = 0  # new # new
    for batch_idx, batch in enumerate(loader_test):
        # print('start batch', batch_idx, '----------------------')
        x, stamp, input_mask, masked_ids, masked_pos, masked_weights, time_matrix, time_gap, cla = batch
        x = x.to(torch.int64).to(device)
        stamp = stamp.to(torch.int64).to(device)
        time_matrix = time_matrix.to(torch.int64)
        time_gap = time_gap.to(torch.int64).to(device)
        cla = cla.to(torch.int64).to(device)
        masked_ids = masked_ids.to(torch.int64).to(device)
        masked_pos = masked_pos.to(torch.int64).to(device)
        input_mask = input_mask.to(torch.int64).to(device)
        masked_weights = masked_weights.to(torch.float32).to(device)
        # y_pre = model(x, stamp, gap, input_mask, masked_pos).tolist()  #
        y_pre = model(x, stamp, input_mask, masked_pos, time_matrix, time_gap, cla, 'test')
        # new_start
        _, indices = torch.topk(y_pre, correct_range, -1)
        indices = indices.view(indices.size()[0], -1)
        masked_ids = masked_ids.view(-1, 1).expand_as(indices)
        hits = (masked_ids == indices).nonzero()
        if len(hits) == 0:
            return 0
        n_hits = (masked_ids == indices).nonzero()[:, :-1].size(0)
        hits_all += n_hits
        test_num += masked_ids.size(0)
        # recall = float(n_hits) / masked_ids.size(0)
        # print('这个batch的准确率为：', recall)
        # recall_li.append(recall)
        # new_end
    print('hit:', hits_all, '|total:', test_num)
    return hits_all / test_num

Config = {
    'vocab_size': 259,  # Size of the vocabulary
    'dim': 35,  # Dimension
    'n_layers': 1,  # Number of layers
    'n_heads': 5,  # Number of attention heads
    'dim_ff': 35 * 4,  # Dimension of the feed-forward layer
    'p_drop_hidden': 0.1,  # Dropout probability for hidden layers
    'p_drop_attn': 0.1,  # Dropout probability for attention layers
    'max_len': 11,  # Maximum length of input sequences
    'iter_n': 100,  # Number of iterations
    'half_epoch': 50,  # Half of the epoch size
    'batch_size': 128,  # Size of each training batch
    'save': False,  # Flag indicating whether to save the model
    'max_pred': 1,  # Maximum number of predictions
    'lr': 0.002,  # Learning rate
    'weight_decay': 1e-4,  # Weight decay
    'mask_id': 256,  # ID for masking tokens
    'pad_id': 257,  # ID for padding tokens
    'interest_id': 258,  # ID for interest tokens
    'seed': 2300,  # Random seed
    'test_while_train': True,  # Flag indicating whether to test while training
    'test_times': 1,  # Number of times to run the test
    'correct_range': 3,  # Range for correctness
    'model': 'Repeat',  # Model type
    'pro_add': 'add',  # Pro add type
    'time_span': 65,  # Time span
    'pre_train': False,  # Pre-training flag
    'save_dir': 'code/model/',  # Directory to save the model
    'repeat_proj': False,  # Repeat projection flag
    'explore_proj': False,  # Explore projection flag
    'cla_dic': {'视频': 1, '游戏': 3, '出行': 3, '电商': 0, '音频': 1, '阅读': 2, '美食': 2, '酒店': 4, '医疗': 4, '生活服务': 3, '通信': 4, '工具': 0, '教育': 4, '办公': 2, '快递': 4},  # Classification dictionary
    'res': True,  # Res flag
    'multi_mask': True  # Multi-mask flag
}

if __name__ == "__main__":
    user_gp_item,user_gp_t,user_gp_month,user_gp_day,user_gp_weekday,user_gp_cla \
        = Get_user_gp(cla_dic=Config['cla_dic'],test=10)#获得训练集的用户购买商品序列
    user_gp_stamp = []#把时间戳信息融合在一起
    user_gp_stamp_t = []
    for i in range(len(user_gp_item)):
        insert = []
        insert_t = []
        for j in range(len(user_gp_item[i])):
            #insert.append([user_gp_month[i][j],user_gp_day[i][j],user_gp_weekday[i][j],user_gp_hour[i][j]])
            insert.append([user_gp_month[i][j],user_gp_day[i][j],user_gp_weekday[i][j]])#user_gp_t是s，s/60/60/24向下取整为日期差
            insert_t.append(user_gp_t[i][j])
            #insert.append(user_gp_month[i][j])
        user_gp_stamp.append(insert)
        user_gp_stamp_t.append(insert_t)
    print('Timestamps loaded')

    vocabulary=np.load('../../data/quanyi_train_vocab.npy')
    vocabulary=vocabulary.tolist()
    #dic_item_emb, vocabulary = Dic_item_emb(emb_dim=Config['dim'])#获得item基础向量表示字典和词表
    print(user_gp_item[:5])#,dic_item_emb.keys())
    print(user_gp_stamp[:5])
    print(user_gp_stamp_t[:5])
    print("vocab loaded.")

    # test部分
    # 获得训练集的用户购买商品序列
    # user_gp_item,user_gp_month,user_gp_day,user_gp_weekday,user_gp_hour = Get_user_gp(test=10)#获得训练集的用户购买商品序列
    user_gp_test, user_gp_t_test, user_gp_month_test, user_gp_day_test, user_gp_weekday_test, user_gp_cla_test = \
        Test_user_gp( cla_dic=Config['cla_dic'], data_num=10)  
    user_gp_stamp_test = []  # 把时间戳信息融合在一起
    user_gp_stamp_t_test = []
    print('#Sequences in test-set:', len(user_gp_test))

    # Narm,Stamp diginetica yoochoose(click) movielen rights FM book?

    for i in range(len(user_gp_test)):
        insert_test = []
        insert_t_test = []
        for j in range(len(user_gp_test[i])):
            # insert_test.append([user_gp_month_test[i][j],user_gp_day_test[i][j],user_gp_weekday_test[i][j],user_gp_hour_test[i][j]])
            insert_test.append([user_gp_month_test[i][j], user_gp_day_test[i][j], user_gp_weekday_test[i][j]])
            insert_t_test.append(user_gp_t_test[i][j])
            # insert_test.append(user_gp_month_test[i][j])
        user_gp_stamp_test.append(insert_test)
        user_gp_stamp_t_test.append(insert_t_test)

    # print('we have %s rebuy test' % len(user_gp_stamp_test_re))
    # print('we have %s other test' % len(user_gp_stamp_test_no))

    train(user_gp_item,
          user_gp_test,
          user_gp_stamp,
          user_gp_stamp_test,
          user_gp_stamp_t,
          user_gp_stamp_t_test,
          user_gp_cla,
          user_gp_cla_test,
          vocabulary,
          Config
        )