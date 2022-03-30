import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchtext.legacy import data
import csv

seq_len = 65
batch_size = 8
vocab_size = 2562
embedding_dim = 300
hidden_dim = 256
n_layers = 2
bidirectional = True
dropout  = 0.2

# 构造一个batch的数据
input = torch.randint(low=0,high=100,size=[seq_len,batch_size])  # input形状是个10*20的二阶矩阵，即每个句子有20个词，一个batch有10个句子

# 数据经过embedding处理
embedding = nn.Embedding(vocab_size,embedding_dim=embedding_dim,padding_idx=1)
input_embedding = embedding(input)  #[10,20]->[10,20,30]

# 将embedd传入LSTM
lstm =  nn.LSTM(embedding_dim
                           , hidden_size=hidden_dim
                           , num_layers=n_layers
                           , bidirectional=bidirectional
                           , dropout=dropout)

fc1= nn.Linear(hidden_dim * 2, hidden_dim)
fc2 = nn.Linear(hidden_dim,5)

lstm_out = lstm(input_embedding)[0]
final = lstm_out.sum(dim=0)
y = fc1(final)
y = fc2(y)
y = y.argmax(-1)
# output,(h_n,c_n) = lstm(input_embedding)
#
# # 获取最后一个时间步上的输出
# last_output = output[:,-1,:]
# # 获取最后一次的hidden_state
# last_hidden_state = h_n[-1,:,:]

if __name__ == '__main__':
    # print(input.shape)  # [65,8]
    # print(input_embedding.shape)  # [65,8,300]
    # print(lstm_out)
    # print(y)
    list = []
    for i in range(5):
        list.append(i)
    dataframe = pd.DataFrame({"rating":list})
    dataframe.to_csv("./Mytesting.csv",mode="a",index = False)


