import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app import DataLoader
import os
from tqdm import tqdm
import random
import numpy as np
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
import time
import matplotlib.pyplot as plt


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden_size = 200
        self.label_size = label_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2linear = nn.Linear(hidden_dim * 2, self.hidden_size)
        self.linear2labels = nn.Linear(self.hidden_size, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sent1_batch, sent2_batch, sent1_batch_lengths, sent2_batch_lengths):
        # check embeddings and pipe maybe sth is wrong with pack padded sequence
        embeds1 = self.word_embeddings(sent1_batch)
        sent1_batch_lengths[::-1].sort()
        pps1 = torch.nn.utils.rnn.pack_padded_sequence(embeds1, sent1_batch_lengths, batch_first=True)
        lstm_out1, self.hidden = self.lstm(pps1.float(), self.hidden)
        lstm_out1_pps, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out1, batch_first=True)
        last_out1 = lstm_out1_pps[:, -1]

        embeds2 = self.word_embeddings(sent2_batch)
        sent2_batch_lengths[::-1].sort()
        pps2 = torch.nn.utils.rnn.pack_padded_sequence(embeds2, sent2_batch_lengths, batch_first=True)
        lstm_out2, self.hidden = self.lstm(pps2.float(), self.hidden)
        lstm_out2_pps, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out2, batch_first=True)
        last_out2 = lstm_out2_pps[:, -1]
        # think about cat dim
        # shape manipulation on tensor to put to linear
        last_out1 = last_out1.contiguous()
        last_out2 = last_out2.contiguous()
        # lstm_out1_to_linear = lstm_out1_pps.view(-1, lstm_out1_pps.shape[2])
        # lstm_out2_to_linear = lstm_out2_pps.view(-1, lstm_out2_pps.shape[2])
        # concatenation_lstm_out = torch.cat((lstm_out1_to_linear, lstm_out2_to_linear), 0)
        concatenation_lstm_out = torch.cat((last_out1, last_out2), 1)

        hidden2linear_out = self.hidden2linear(concatenation_lstm_out)
        hidden2linear_out_relu = F.relu(hidden2linear_out)
        logits = self.linear2labels(hidden2linear_out_relu)
        log_probs = F.log_softmax(logits, dim=1)
        # log_probs = log_probs.view(self.batch_size, self.label_size)

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
    HIDDEN_DIM = 200
    EPOCH = 3
    BATCH_SIZE = 32

    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root_dir, 'data')
    results_path = os.path.join(root_dir, 'results')
    train_path = os.path.join(data_path, 'train', 'train_processed_data.pickle')
    dev_path = os.path.join(data_path, 'dev', 'dev_processed_data.pickle')
    test_path = os.path.join(data_path, 'test', 'test_processed_data.pickle')
    embeddings_path = os.path.join(data_path, 'vocab', 'final_embeddings_matrix.npy')
    vocab_path = os.path.join(data_path, 'vocab', 'vocab_dict.pkl')

    data_loader = DataLoader.DataLoader(train_path=train_path,
                                        test_path=test_path,
                                        dev_path=dev_path,
                                        vocab_path=vocab_path,
                                        embeddings_path=embeddings_path,
                                        batch_size=BATCH_SIZE)
    train_iter, dev_iter, test_iter = data_loader.get_batches()

    best_dev_acc = 0.0

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM,
                           vocab_size=len(data_loader.vocab_dict),
                           label_size=len(data_loader.labels_dict),
                           batch_size=BATCH_SIZE)

    model.word_embeddings.weight.data = torch.from_numpy(data_loader.embeddings)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('\nStart training.\n')
    for i in range(EPOCH):
        print(F'\nEpoch {i}:')
        time.sleep(1)
        train_epoch(model, train_iter, loss_function, optimizer, i, results_path)
        print('best acc at dev set:', best_dev_acc)
        dev_acc = evaluate(model, dev_iter, loss_function, i, 'dev', results_path)
        test_acc = evaluate(model, test_iter, loss_function, i, 'test', results_path)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model_path = os.path.join(results_path,
                                      F"epoch_*_best_model_acc_*.model")
            os.system(F'rm {model_path}')

            torch.save(model.state_dict(),
                       os.path.join(results_path,
                                    F'epoch_{i}_best_model_acc_' +
                                    str(int(test_acc*10000)) + '.model')
                       )

            print('Saving best model.')


def final_evaluation():
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 200
    EPOCH = 3
    BATCH_SIZE = 32

    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root_dir, 'data')
    results_path = os.path.join(root_dir, 'results')
    train_path = os.path.join(data_path, 'train', 'train_processed_data.pickle')
    dev_path = os.path.join(data_path, 'dev', 'dev_processed_data.pickle')
    test_path = os.path.join(data_path, 'test', 'test_processed_data.pickle')
    embeddings_path = os.path.join(data_path, 'vocab', 'final_embeddings_matrix.npy')
    vocab_path = os.path.join(data_path, 'vocab', 'vocab_dict.pkl')

    data_loader = DataLoader.DataLoader(train_path=train_path,
                                        test_path=test_path,
                                        dev_path=dev_path,
                                        vocab_path=vocab_path,
                                        embeddings_path=embeddings_path,
                                        batch_size=BATCH_SIZE)
    train_iter, dev_iter, test_iter = data_loader.get_batches()

    best_model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM,
                                vocab_size=len(data_loader.vocab_dict),
                                label_size=len(data_loader.labels_dict),
                                batch_size=BATCH_SIZE)
    best_model.word_embeddings.weight.data = torch.from_numpy(data_loader.embeddings)
    import glob
    filename = glob.glob(os.path.join(results_path, F"epoch_*_best_model_acc_*.model"))
    print(filename)
    best_model.load_state_dict(torch.load(filename[0]))
    loss_function = nn.CrossEntropyLoss()
    evaluate(best_model, test_iter, loss_function, 0, 'final_evaluation', results_path)


def evaluate(model, eval_iter, loss_function, epoch, name, results_path):
    print(F'\n{name} phase:\n')
    time.sleep(1)
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    loss_list = []
    acc_list = []
    # import visdom
    # vis = visdom.Visdom()
    # loss_window = vis.line(X=np.zeros((1,)),
    #                        Y=np.zeros((1,)),
    #                        opts=dict(xlabel='iteration',
    #                                  ylabel='Loss',
    #                                  title='Iteration loss',
    #                                  ))
    #
    # acc_window = vis.line(X=np.zeros((1,)),
    #                        Y=np.zeros((1,)),
    #                        opts=dict(xlabel='iteration',
    #                                  ylabel='Accuracy',
    #                                  title='Iteration accuracy',
    #                                  ))
    for i in tqdm(range(len(eval_iter))):
        sent1, sent2, label = eval_iter[i].premises, eval_iter[i].hypothesises, eval_iter[i].labels
        label = torch.LongTensor(label)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        sent1_lengths = (sent1 != 0).sum(1)
        sent2_lengths = (sent2 != 0).sum(1)
        pred = model(torch.from_numpy(sent1), torch.from_numpy(sent2), sent1_lengths, sent2_lengths)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        loss_list.append(loss)
        acc_list.append(get_accuracy(truth_res, pred_res))
        # np.random.seed(i)
        # X1 = np.ones((1, 1)) * i
        #
        # np.random.seed(i + 1)
        # Y1 = np.array([avg_loss / (i + 1)]) * np.random.randint(1, 100) - 10
        # Y2 = np.array([acc]) - 10
        # vis.line(
        #     X=np.column_stack(X1),
        #     Y=np.column_stack(Y1),
        #     win=loss_window,
        #     update='append')
        #
        # vis.line(
        #     X=np.column_stack(X1),
        #     Y=np.column_stack(Y2),
        #     win=acc_window,
        #     update='append')



    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print(name + F' avg_loss: {avg_loss} train acc: {acc}')
    create_line_plot(name, 'accuracy', epoch, eval_iter, acc_list, results_path)
    create_line_plot(name, 'loss', epoch, eval_iter, loss_list, results_path)
    return acc


def train_epoch(model, train_iter, loss_function, optimizer, epoch, results_path):
    print('\ntrain phase:\n')
    time.sleep(1)
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    loss_list = []
    acc_list = []
    for i in tqdm(range(len(train_iter))):
        sent1, sent2, label = train_iter[i].premises, train_iter[i].hypothesises, train_iter[i].labels
        sent1_lengths = (sent1 != 0).sum(1)
        sent2_lengths = (sent2 != 0).sum(1)
        label = torch.LongTensor(label)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(torch.from_numpy(sent1), torch.from_numpy(sent2), sent1_lengths, sent2_lengths)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        loss_list.append(loss)
        acc_list.append(get_accuracy(truth_res, pred_res))
        count += 1
        if count % 100 == 0:
            print(F'epoch: {epoch} iterations: {count * model.batch_size} loss :{loss.item()}')
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    print(F'epoch: {epoch} finished.')
    print(F'train avg_loss: {avg_loss}, acc: {get_accuracy(truth_res, pred_res)}')
    create_line_plot('train', 'accuracy', epoch, train_iter, acc_list, results_path)
    create_line_plot('train', 'loss', epoch, train_iter, loss_list, results_path)


def create_line_plot(phase_name, Y_name, epoch, iter, Y, results_path):
    X = np.arange(len(iter))
    fig, ax = plt.subplots()
    ax.plot(X, np.array(Y))
    ax.set(xlabel='iter', ylabel=Y_name)
    ax.grid()
    fig.savefig(os.path.join(results_path, F"{phase_name}_{epoch}_{Y_name}.png"))
    plt.show()


if __name__ == '__main__':
    train()
    final_evaluation()

