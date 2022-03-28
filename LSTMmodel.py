import torch
from torch import optim
from torchtext.legacy import data
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

# configuration
train_path = "./Mytraining.csv"
valid_path = "./Myvalidation.csv"
test_path = "./Mytesting.csv"
batch_size = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

split_space = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=split_space, lower=False)
LABEL = data.Field(sequential=False, use_vocab=False)


class DrugReviewDataset(data.Dataset):

    # iteration进行排序的关键字,以长度为标准进行排序
    @staticmethod
    def sort_key(input):
        return len(input.review)

    def __init__(self, path, text_field, label_field, test=False):
        fields = [("review", text_field),
                  ("rating", label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        print("Reading data from {}".format(path))
        if test:
            # 测试集中没有标签
            for text in tqdm(csv_data['review']):
                examples.append(data.Example.fromlist([text, None], fields=fields))
        else:
            for text, label in tqdm(zip(csv_data['review'], csv_data['rating'])):
                examples.append(data.Example.fromlist([text, label], fields=fields))
        super(DrugReviewDataset, self).__init__(examples, fields)


# 构建数据集
train = DrugReviewDataset(train_path, text_field=TEXT, label_field=LABEL, test=False)
valid = DrugReviewDataset(valid_path, text_field=TEXT, label_field=LABEL, test=False)
test = DrugReviewDataset(test_path, text_field=TEXT, label_field=None, test=True)

# 构建词向量表
TEXT.build_vocab(train, min_freq=2)
TEXT.build_vocab(valid, min_freq=2)
TEXT.build_vocab(test, min_freq=2)

train_iter = data.BucketIterator(dataset=train
                                 , batch_size=batch_size
                                 , shuffle=True, sort_within_batch=False
                                 , repeat=False
                                 , device=device)  #
valid_iter = data.BucketIterator(dataset=valid
                                 , batch_size=batch_size
                                 , shuffle=True
                                 , sort_within_batch=False
                                 , repeat=False
                                 , device=device)  #
test_iter = data.BucketIterator(dataset=test
                                , batch_size=1
                                , sort_within_batch=False
                                , repeat=False
                                , device=device)  #

# LSTM 超参数
EPOCH = 5

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.4
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # padding在词向量中的序号


class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, pad_idx):
        super(LSTM_net, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim
                           , hidden_size=hidden_dim
                           , num_layers=n_layers
                           , bidirectional=bidirectional
                           , dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 6)

    def forward(self, text):
        embedding = self.embedding(text)
        rnn_out = self.rnn(embedding)[0]
        final = rnn_out.sum(dim=0)
        y = self.fc1(final)
        y = self.fc2(y)
        return y


model = LSTM_net(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
model.to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
cri = torch.nn.CrossEntropyLoss()


# 训练模型
def train(epoch):
    total_loss = 0
    for idx, batch in enumerate(train_iter):
        input = batch.review
        target = batch.rating

        optimizer.zero_grad()
        predicted = model(input)
        loss = cri(predicted, target)
        loss.backward()
        optimizer.step()
        total_loss = loss
        if idx % 8 == 0:
            print('{}th batch, loss is {}'.format(idx, loss.item()))
    print("ep : {}, loss : {}".format(epoch + 1, total_loss))


# 评估模型
def eval(epoch):
    with torch.no_grad():
        true_count = 0
        total_count = 0
        for idx, batch in enumerate(valid_iter):
            input = batch.review
            target = batch.rating
            predicted = model(input).argmax(-1)
            true_count += (predicted == target).sum()
            total_count += batch_size
        print("{}th epoch's accuracy is {}".format(epoch + 1, true_count / total_count))


# 对测试集数据进行预测
def test():
    res = []
    for idx, batch in enumerate(test_iter):
        test_review = batch.review
        test_rating = model(test_review).argmax(-1).item()
        print(test_rating)
        res.append(test_rating)
    return res


if __name__ == '__main__':
    # for idx,batch in enumerate(train_iter):
    #     print(idx)
    #     print(batch.review.shape)
    #     print(type(batch.review))
    #     print(batch.rating.shape)
    #     print("="*80)
    # print(len(TEXT.vocab))
    for i in range(EPOCH):
        train(i)
        eval(i)
    result = test()
    print(len(result))

