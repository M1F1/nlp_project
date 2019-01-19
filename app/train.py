# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app.DataLoader import DataLoader
import os
from tqdm import tqdm
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
# torch.cuda.set_device(args.gpu)
import torch.utils.data as Data


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sent1_batch, sent2_batch):
        embeds = self.word_embeddings(sent1_batch)
        x1 = embeds.view(len(sent1_batch), self.batch_size, -1)
        lstm_out1, self.hidden = self.lstm(x1, self.hidden)

        embeds = self.word_embeddings(sent2_batch)
        x2 = embeds.view(len(sent2_batch), self.batch_size , -1)
        lstm_out2, self.hidden = self.lstm(x2, self.hidden)

        concatenation_lstm_out = torch.cat((lstm_out1, lstm_out2), 0)

        hidden2label_out = self.hidden2label(concatenation_lstm_out)
        hidden2label_out_relu = F.relu(hidden2label_out)
        y = self.second_linear_2label(hidden2label_out_relu)
        log_probs = F.log_softmax(y)
        return log_probs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right/len(truth)


def train():
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 50
    EPOCH = 3
    BATCH_SIZE = 32
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root_dir, 'data')
    train_path = os.path.join(data_path, 'train', 'train_processed_data.pickle')
    dev_path = os.path.join(data_path, 'dev', 'dev_processed_data.pickle')
    test_path = os.path.join(data_path, 'test', 'test_processed_data.pickle')
    embeddings_path = os.path.join(data_path, 'vocab', 'final_embeddings_matrix.npy')
    vocab_path = os.path.join(data_path, 'vocab', 'vocab_dict.pkl')
    data_loader = DataLoader(train_path=train_path,
                             test_path=test_path,
                             dev_path=dev_path,
                             vocab_path=vocab_path,
                             embeddings_path=embeddings_path,
                             batch_size=BATCH_SIZE)
    train_iter, dev_iter, test_iter = data_loader.get_batches()
    print(data_loader.embeddings.shape)

    best_dev_acc = 0.0

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM,
                           vocab_size=len(data_loader.vocab_dict),
                           label_size=len(data_loader.labels_dict),
                           batch_size=BATCH_SIZE)
    model.word_embeddings.weight.data = torch.from_numpy(data_loader.embeddings)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    no_up = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_iter, loss_function, optimizer, i)
        print('now best dev acc:', best_dev_acc)
        dev_acc = evaluate(model, dev_iter, loss_function, 'dev')
        test_acc = evaluate(model, test_iter, loss_function, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm best_models/mr_best_model_minibatch_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(), 'best_models/mr_best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()


def evaluate(model, eval_iter, loss_function,  name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    # tgqm
    print(F'\n{name} phase:\n')
    for i in tqdm(range(len(eval_iter))):
        sent1, sent2, label = eval_iter[i].premises, eval_iter[i].hypothesises, eval_iter[i].hypothesises
        label = torch.IntTensor(label)
        # label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent1, sent2)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x[0] for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc))
    return acc


def train_epoch(model, train_iter, loss_function, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    print('\ntrain phase:\n')
    for i in tqdm(range(len(train_iter))):
        sent1, sent2, label = train_iter[i].premises, train_iter[i].hypothesises, train_iter[i].labels
        label = torch.IntTensor(label)
        # label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(torch.from_numpy(sent1), torch.from_numpy(sent2))
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x[0] for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 100 == 0:
            print('epoch: %d iterations: %d loss :%g' % (epoch, count*model.batch_size, loss.data[0]))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(epoch, avg_loss, get_accuracy(truth_res,pred_res)))


if __name__ == '__main__':
    train()