import numpy as np
import torch
import gzip
from argparse import ArgumentParser
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from BLSTM import train, load_data, Model_Data, enc_tags, makeOutput

class BLSTMGlove(torch.nn.Module):
    def __init__(self, embs_glove, embs_casing, lstm_inp_size, hidden, n_layers, linear_op_size, n_classes, dropout, pad_case, pad_word):
        super(BLSTMGlove, self).__init__()
        self.hidden_dim = hidden
        self.num_layers = n_layers
        self.emb_casing = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_casing).float(), padding_idx=pad_case)
        self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_glove).float(), padding_idx=pad_word)
        self.bilstm = torch.nn.LSTM(input_size=lstm_inp_size, hidden_size=hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.drop = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(2*hidden, linear_op_size)
        self.elu = torch.nn.ELU()
        self.linear2 = torch.nn.Linear(linear_op_size, n_classes)
        
    def forward(self, x, l):
        x_word_info = self.emb(x.select(1,0))
        x_case_info = self.emb_casing(x.select(1,1))

        x = torch.cat([x_word_info,x_case_info], dim=2)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(x_pack)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        x = self.drop(lstm_out)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        return x

def load_glove_casing(loc):
    # Loading the glove embeddings
    vocab,embeddings = [],[]
    with gzip.open(loc+'glove.6B.100d.gz','rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    # case info encoding
    casing_enc = {'<pad>':0, 'numeric': 1, 'allLower':2, 'allUpper':3, 'initialUpper':4, 'other':5, 'mainly_numeric':6, 'contains_digit': 7}
    embs_casing = np.identity(len(casing_enc), dtype='float32')

    vocab_glove = np.array(vocab)
    embs_glove = np.array(embeddings)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_glove = np.insert(vocab_glove, 0, '<pad>')
    vocab_glove = np.insert(vocab_glove, 1, '<unk>')

    vocab_glove_enc = {k:v for v,k in enumerate(vocab_glove)}
    vocab_glove_dec = list(vocab_glove_enc.keys())

    pad_emb_glove = np.zeros((1,embs_glove.shape[1]))   #embedding for '<pad>' token.
    unk_emb_glove = np.mean(embs_glove,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_glove = np.vstack((pad_emb_glove,unk_emb_glove,embs_glove))

    return vocab_glove_enc, vocab_glove_dec, embs_glove, casing_enc, embs_casing

# Data encoding
def enc_casing(word, caseLookup):   
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
   
    return caseLookup[casing]

# capturing case information
def enc_case(x,case):
    return np.array([enc_casing(w,case) for w in x])

# making word encoding case insensitive
def enc_sent_case_insensitive(x,voc):
    return np.array([voc.get(w.lower(), voc['<unk>']) for w in x])

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

    print("Encoding Data...")
    # how many samples per batch to load
    batch_size = 16

    X_train_sent_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent_case_insensitive(x,vocab_glove_enc)) for x in X_train], batch_first=True, padding_value=vocab_glove_enc['<pad>'])
    X_train_case_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_case(x,casing_enc)) for x in X_train], batch_first=True, padding_value=casing_enc['<pad>'])
    X_train_pad = [[torch.vstack([x1,x2]),len(l)] for x1,x2,l in zip(X_train_sent_pad,X_train_case_pad,X_train)]
    y_train_torch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_tags(y_,tag_enc)) for y_ in y_train], batch_first=True, padding_value=tag_enc['<pad>'])

    X_dev_sent_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent_case_insensitive(x,vocab_glove_enc)) for x in X_dev], batch_first=True, padding_value=vocab_glove_enc['<pad>'])
    X_dev_case_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_case(x,casing_enc)) for x in X_dev], batch_first=True, padding_value=casing_enc['<pad>'])
    X_dev_pad = [[torch.vstack([x1,x2]),len(l)] for x1,x2,l in zip(X_dev_sent_pad,X_dev_case_pad,X_dev)]
    y_dev_torch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_tags(y_,tag_enc)) for y_ in y_dev], batch_first=True, padding_value=tag_enc['<pad>'])

    X_test_sent_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent_case_insensitive(x,vocab_glove_enc)) for x in X_test], batch_first=True, padding_value=vocab_glove_enc['<pad>'])
    X_test_case_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_case(x,casing_enc)) for x in X_test], batch_first=True, padding_value=casing_enc['<pad>'])
    X_test_pad = [[torch.vstack([x1,x2]),len(l)] for x1,x2,l in zip(X_test_sent_pad,X_test_case_pad,X_test)]

    train_data = Model_Data(X_train_pad,y_train_torch)
    dev_data = Model_Data(X_dev_pad,y_dev_torch)
    test_data = Model_Data(X_test_pad,[])

    # prepare data loaders
    train_loader_glove = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    dev_loader_glove = torch.utils.data.DataLoader(dev_data, batch_size=batch_size)
    test_loader_glove = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    print("Preparing Model...")
    # Hyperparameters
    case_padding = casing_enc['<pad>']
    word_padding = vocab_glove_enc['<pad>']
    hidden_size=256 
    num_layers=1 
    linear_op_size=128
    num_classes=len(tag_enc.keys())-1 # exclude the padding tag
    dropout = 0.33
    pad_idx = vocab_enc['<pad>']
    lstm_inp_size = embs_glove.shape[1] + embs_casing.shape[1]
    class_weights = torch.tensor([(1-v/sum(tags.values())) for v in tags.values()]).to(device)

    blstmClassifierGlove = BLSTMGlove(embs_glove, embs_casing, lstm_inp_size, hidden_size, num_layers, linear_op_size, num_classes, dropout, case_padding, word_padding) 
    blstmClassifierGlove.to(device)
    print(blstmClassifierGlove)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=tag_enc['<pad>'])
    optimizer = torch.optim.SGD(blstmClassifierGlove.parameters(), lr=0.1, momentum=0.9) 
    n_epochs = 60

    if args.mode=="train":
        # Train the model
        print("Training Model for {} epochs on device: {}...".format(n_epochs, device))
        train_f1, dev_f1, losses = train(n_epochs,num_classes,blstmClassifierGlove,criterion,optimizer,tag_enc,train_loader_glove,dev_loader_glove,'blstm2.pt')

    elif args.mode=="test":
        print("Inferring from saved model...")
        # Loading the best model and inferring on the dev_data and test_data
        blstmClassifierGlove.load_state_dict(torch.load('blstm2.pt', map_location=torch.device(device)))

        makeOutput(blstmClassifierGlove,dev_loader_glove,X_dev,tag_dec,"dev2.out")
        makeOutput(blstmClassifierGlove,test_loader_glove,X_test,tag_dec,"test2.out")

        print("Model predictions on dev_set and test_set saved.")