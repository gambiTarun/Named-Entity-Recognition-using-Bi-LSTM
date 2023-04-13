import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from BLSTM import train, load_data, Model_Data, enc_tags, makeOutput
from BLSTMGlove import load_glove_casing, enc_case, enc_sent_case_insensitive

class CNN(torch.nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, cnn_win_size, pad_char, outshape):
        super(CNN, self).__init__()
        
        self.emb_char = torch.nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_char)
        torch.nn.init.xavier_uniform_(self.emb_char.weight)
        self.drop = torch.nn.Dropout(0.5)
        self.cd2 = torch.nn.Conv2d(char_emb_dim, out_channels=outshape, kernel_size=(1,cnn_win_size), padding='same')
        self.maxpool = torch.nn.MaxPool2d((68,1))
        
    def forward(self, x):
        
        x = self.emb_char(x)
        x = self.drop(x)
        # (batch, signal, channel) -> (batch, channel, signal) 
        x = torch.permute(x, (0,3,2,1))
        x = self.cd2(x)
        x = self.maxpool(x)
        x = torch.squeeze(x)
        # (batch, channel, signal) -> (batch, signal, channel) 
        x = torch.permute(x, (0,2,1))
    
        return x
    
class BLSTM_CNN(torch.nn.Module):
    def __init__(self, embs_glove, embs_casing, cnn_out_dim, char_vocab_size, char_emb_dim, cnn_win_size, lstm_inp_size, hidden, n_layers, linear_op_size, n_classes, dropout, pad_char, pad_case, pad_word):
        super(BLSTM_CNN, self).__init__()
        self.hidden_dim = hidden
        self.num_layers = n_layers
        self.cnn_outshape = cnn_out_dim
        
        self.cnn_module = CNN(char_vocab_size, char_emb_dim, cnn_win_size, pad_char, outshape=cnn_out_dim)
        self.emb_casing = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_casing).float(), padding_idx=pad_case)
        self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_glove).float(), padding_idx=pad_word)
        self.bilstm = torch.nn.LSTM(input_size=lstm_inp_size, hidden_size=hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.linear1 = torch.nn.Linear(2*hidden, linear_op_size)
        self.elu = torch.nn.ELU()
        self.drop = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(linear_op_size, n_classes)
        
    def forward(self, x, l):
        
        x_word_info = self.emb(x.select(-1,0).select(-1,0))
        x_case_info = self.emb_casing(x.select(-1,1).select(-1,0))
        
        # x_char_info = []

        # batch of sentences -> sentence of words -> word of chars
        # time-distributed cnn module
        # for words in x.select(-1,2):
        #     x_char_info.append(self.cnn_module(words).view(-1,self.cnn_outshape))
        
        # x_char_info = torch.stack(x_char_info)
        
        x_char_info = self.cnn_module(x.select(-1,2))
        
        x = torch.cat([x_word_info,x_case_info,x_char_info], dim=2)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(x_pack)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        x = self.drop(lstm_out)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        return x

def enc_char_info(x,char):

    def enc_char(w,char):
        return np.array([char.get(c, char['<unk>']) for c in w])

    x_pad = []
    for w in x:
        enc_x = torch.from_numpy(enc_char(w,char))
        # padding with "<pad>":0
        x_pad.append(torch.nn.functional.pad(enc_x,(0,longest_word_for_padding-enc_x.shape[0])))
    
    # length info not required
    # x_out = torch.vstack([torch.hstack([x,torch.tensor(len(w))]) for x,w in zip(x_pad,x)])
    
    return torch.vstack(x_pad)


if __name__=='__main__':
    
    # argument parsing
    parser = ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, choices=['train', 'test'], default='test')
    args = parser.parse_args()

    # Data Loading
    print("Loading dataset...")
    loc = ""
    X_train, X_dev, X_test, y_train, y_dev, vocab_enc, tag_enc, tag_dec, tags = load_data(loc)

    print("Loading Glove embeddings...")
    vocab_glove_enc, vocab_glove_dec, embs_glove, casing_enc, embs_casing  = load_glove_casing(loc)

    # character information encoding
    char_enc = {"<pad>":0, "<unk>":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}<>!?:;#'\"/\\%$`&=*+@^~|":
        char_enc[c] = len(char_enc)
        
    longest_word_for_padding = max([len(w) for w in vocab_glove_enc.keys()])

    print("Encoding Data...")
    # how many samples per batch to load
    batch_size = 16

    # encoding and packing the train data
    X_train_sent_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent_case_insensitive(x,vocab_glove_enc)) for x in X_train], batch_first=True, padding_value=vocab_glove_enc['<pad>'])

    X_train_case_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_case(x,casing_enc)) for x in X_train], batch_first=True, padding_value=casing_enc['<pad>'])

    X_train_char_pad = [enc_char_info(x,char_enc) for x in X_train]
    X_train_char_pad = [torch.nn.ConstantPad2d((0,0,0,X_train_sent_pad.shape[1]-x.shape[0]), 0)(x) for x in X_train_char_pad]
    X_train_pad = []
    for x1,x2,x3,l in zip(X_train_sent_pad,X_train_case_pad,X_train_char_pad,X_train):
        x1 = x1.view(-1, 1).expand(x3.shape)
        x2 = x2.view(-1, 1).expand(x3.shape)
        X_train_pad.append([torch.stack([x1,x2,x3],dim=2),len(l)])

    y_train_torch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_tags(y_,tag_enc)) for y_ in y_train], batch_first=True, padding_value=tag_enc['<pad>'])

    # encoding and packing the dev data
    X_dev_sent_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent_case_insensitive(x,vocab_glove_enc)) for x in X_dev], batch_first=True, padding_value=vocab_glove_enc['<pad>'])

    X_dev_case_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_case(x,casing_enc)) for x in X_dev], batch_first=True, padding_value=casing_enc['<pad>'])

    X_dev_char_pad = [enc_char_info(x,char_enc) for x in X_dev]
    X_dev_char_pad = [torch.nn.ConstantPad2d((0,0,0,X_dev_sent_pad.shape[1]-x.shape[0]), 0)(x) for x in X_dev_char_pad]
    X_dev_pad = []
    for x1,x2,x3,l in zip(X_dev_sent_pad,X_dev_case_pad,X_dev_char_pad,X_dev):
        x1 = x1.view(-1, 1).expand(x3.shape)
        x2 = x2.view(-1, 1).expand(x3.shape)
        X_dev_pad.append([torch.stack([x1,x2,x3],dim=2),len(l)])
        
    y_dev_torch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_tags(y_,tag_enc)) for y_ in y_dev], batch_first=True, padding_value=tag_enc['<pad>'])

    # encoding and packing the test data
    X_test_sent_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent_case_insensitive(x,vocab_glove_enc)) for x in X_test], batch_first=True, padding_value=vocab_glove_enc['<pad>'])

    X_test_case_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_case(x,casing_enc)) for x in X_test], batch_first=True, padding_value=casing_enc['<pad>'])

    X_test_char_pad = [enc_char_info(x,char_enc) for x in X_test]
    X_test_char_pad = [torch.nn.ConstantPad2d((0,0,0,X_test_sent_pad.shape[1]-x.shape[0]), 0)(x) for x in X_test_char_pad]
    X_test_pad = []
    for x1,x2,x3,l in zip(X_test_sent_pad,X_test_case_pad,X_test_char_pad,X_test):
        x1 = x1.view(-1, 1).expand(x3.shape)
        x2 = x2.view(-1, 1).expand(x3.shape)
        X_test_pad.append([torch.stack([x1,x2,x3],dim=2),len(l)])

    train_data = Model_Data(X_train_pad,y_train_torch)
    dev_data = Model_Data(X_dev_pad,y_dev_torch)
    test_data = Model_Data(X_test_pad,[])

    # prepare data loaders
    train_loader_blstm_cnn = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    dev_loader_blstm_cnn = torch.utils.data.DataLoader(dev_data, batch_size=batch_size)
    test_loader_blstm_cnn = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    print("Preparing Model...")
    # Hyperparameters
    cnn_win_size = 3
    char_vocab_size = len(char_enc.keys())
    char_emb_dim = 30
    cnn_out_dim = 20
    char_padding = char_enc['<pad>']
    case_padding = casing_enc['<pad>']
    word_padding = vocab_glove_enc['<pad>']
    hidden_size=256 
    num_layers=1 
    linear_op_size=128
    num_classes=len(tag_enc.keys())-1 # exclude the padding tag
    dropout = 0.33
    pad_idx = vocab_enc['<pad>']
    lstm_inp_size = embs_glove.shape[1] + embs_casing.shape[1] + cnn_out_dim
    class_weights = torch.tensor([(1-v/sum(tags.values())) for v in tags.values()]).to(device)

    BLSTM_CNN_classifier = BLSTM_CNN(embs_glove, embs_casing, cnn_out_dim, char_vocab_size, char_emb_dim, cnn_win_size, lstm_inp_size, hidden_size, num_layers, linear_op_size, num_classes, dropout, char_padding, case_padding, word_padding) 
    BLSTM_CNN_classifier.to(device)
    print(BLSTM_CNN_classifier)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=tag_enc['<pad>'])
    optimizer = torch.optim.SGD(BLSTM_CNN_classifier.parameters(), lr=0.1, momentum=0.9) 
    n_epochs = 60

    if args.mode=="train":
        # Train the model
        print("Training Model for {} epochs on device: {}...".format(n_epochs, device))
        train_f1, dev_f1, losses = train(n_epochs,num_classes,BLSTM_CNN_classifier,criterion,optimizer,tag_enc,train_loader_blstm_cnn,dev_loader_blstm_cnn,'blstm_cnn.pt')
    
    elif args.mode=="test":
        print("Inferring from saved model...")
        # Loading the best model and inferring on the dev_data and test_data
        BLSTM_CNN_classifier.load_state_dict(torch.load('blstm_cnn.pt', map_location=torch.device(device)))

        makeOutput(BLSTM_CNN_classifier,dev_loader_blstm_cnn,X_dev,tag_dec,"dev_pred")
        makeOutput(BLSTM_CNN_classifier,test_loader_blstm_cnn,X_test,tag_dec,"pred")

        print("Model predictions on dev_set and test_set saved.")

