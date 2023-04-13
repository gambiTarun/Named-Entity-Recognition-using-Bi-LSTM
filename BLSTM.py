import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from argparse import ArgumentParser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BLSTM(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, n_layers, linear_op_size, n_classes, dropout, pad_idx):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden
        self.num_layers = n_layers
        self.emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.bilstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.drop = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(2*hidden, linear_op_size)
        self.elu = torch.nn.ELU()
        self.linear2 = torch.nn.Linear(linear_op_size, n_classes)
        
    def forward(self, x, l):
        # Initializing hidden state for first input with zeros
        x = self.emb(x)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(x_pack)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = self.drop(lstm_out)

        x = self.linear1(lstm_out)
        x = self.elu(x)
        x = self.linear2(x)
        return x

class Model_Data(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sent = self.X[idx][0].long().to(device)
        ner = self.y[idx].long().to(device) if len(self.y)!=0 else []
        le = self.X[idx][1]
        return sent, ner, le, idx
    
def load_data(loc):
    train_file_data, dev_file_data, test_file_data = None,None,None

    # opening the train,test,dev data to read it
    with open(loc+"data/train",'r',encoding='utf8') as f:
        train_file_data = f.read().split("\n")
        
    with open(loc+'data/dev','r',encoding='utf8') as f:
        dev_file_data = f.read().split("\n")
        
    with open(loc+'data/test','r',encoding='utf8') as f:
        test_file_data = f.read().split("\n")

    # setting threshold for the min frequency of words in the vocab
    thresh = 0
    vocab = {}
    for line in train_file_data:
        if line!='':
            d = line.split(' ')
            # incrementing the frequency in vocab as we encounter the word
            vocab[d[1]] = vocab.get(d[1], 0) + 1
            
    unk = 0
    words=[]
    # creating a words list to account for minimum frequency threshold
    for k,v in vocab.items():
        if v>thresh:
            words.append([v,k])
        else:
            unk+=1

    # creating a vocab dictionary for word frequency
    vocab = {}
    vocab["<unk>"] = unk
    for i,v in enumerate(sorted(words, reverse=True)):
        vocab[v[1]] = v[0]

    # creating a tag frequency dictionary for class weights
    tags = {}
    for w in train_file_data:
        if w!='':
            d = w.split(' ')
            if d[1] in vocab:
                tags[d[2]] = tags.get(d[2], 0) + 1
            
    tag_enc = {k:v for v,k in enumerate(list(tags.keys()))}
    tag_enc['<pad>'] = -1
    vocab_enc = {k:v for v,k in enumerate(['<pad>']+list(vocab.keys()))}
    tag_dec = list(tag_enc.keys())
    vocab_dec = list(vocab_enc.keys())

    sent, label = [], []
    X_train, y_train = [], []
    for w in train_file_data:
        if w=='':
            X_train.append(np.array(sent))
            y_train.append(np.array(label))
            sent,label=[],[]
        else:
            d = w.split(' ')
            sent.append(d[1])
            label.append(d[2])
            
    sent, label = [], []
    X_dev, y_dev = [], []
    for w in dev_file_data:
        if w=='':
            X_dev.append(np.array(sent))
            y_dev.append(np.array(label))
            sent,label=[],[]
        else:
            d = w.split(' ')
            sent.append(d[1])
            label.append(d[2])
            
    sent = []
    X_test = []
    for w in test_file_data:
        if w=='':
            X_test.append(np.array(sent))
            sent=[]
        else:
            d = w.split(' ')
            sent.append(d[1])

    return X_train, X_dev, X_test, y_train, y_dev, vocab_enc, tag_enc, tag_dec, tags

def enc_sent(x,voc):
    return np.array([voc.get(w, voc['<unk>']) for w in x])

def enc_tags(x,tag):
    return np.array([tag.get(w) for w in x])

def train(n_epochs,n_classes,model,criterion,optimizer,tag_enc,training_loader,dev_loader,model_name):

    # initialize tracker for minimum validation loss
    train_f1 = [0]
    dev_f1 = [0]
    losses = [np.Inf]

    train_loss_min = np.Inf # set initial "min" to infinity
        
    for epoch in range(n_epochs):
        # monitor training loss
        
        train_loss = 0
        model.train() # prep model for training

        pred, truth = [], []
        for data, target, length, _ in tqdm(training_loader):
            
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, length)

            predicted = torch.argmax(output, 2)
            [[pred.append(x.item()) for x in p[:t]] for p,t in zip(predicted,length)]
            [[truth.append(x.item()) for x in i[:t]] for i,t in zip(target,length)]

            target = target[:,:max(length)].clone()
            loss = criterion(output.view(-1, n_classes), target.long().view(-1))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
  
        train_f1.append(f1_score(truth,pred,labels=list(tag_enc.values()),average='macro',zero_division=0))

        pred, truth = [], []
        for x, y, l, _ in dev_loader:
            outputs = model(x,l)
            predicted = torch.argmax(outputs, 2)
            [[pred.append(x.item()) for x in p[:t]] for p,t in zip(predicted,l)]
            [[truth.append(x.item()) for x in i[:t]] for i,t in zip(y,l)]

        dev_f1.append(f1_score(truth,pred,labels=list(tag_enc.values()),average='macro',zero_division=0))
        
        train_loss = train_loss/len(training_loader.dataset)
        losses.append(train_loss)

        print('Epoch: {} of {}, \tTraining Loss: {:.6f}'.format(epoch+1,n_epochs,train_loss))

        # save model if validation loss has decreased
        if train_loss <= train_loss_min:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min,
            train_loss))
            torch.save(model.state_dict(), model_name)
            train_loss_min = train_loss

    return train_f1, dev_f1, losses

def makeOutput(model, dataloader, raw_data, tag_dec, filename):
    output = []
    for x, y, l, ids in dataloader:
        outputs = model(x,l)
        pred_labels = torch.argmax(outputs, 2)
        for predicted,siz,id in zip(pred_labels,l,ids):
            words = [w for w in raw_data[id]]
            # golds = [tag[i.item()] for i in true[:siz]]
            preds = [tag_dec[i.item()] for i in predicted[:siz]]
            [output.append(' '.join([str(i+1),w,p,"\n"])) for i,(w,p) in enumerate(zip(words,preds))]
            output.append('\n')
            
    with open(filename,'w') as f:
        f.writelines(output)


if __name__=='__main__':

    # argument parsing
    parser = ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, choices=['train', 'test'], default='test')
    args = parser.parse_args()

    # Data Loading
    print("Loading dataset...")
    loc = ""
    X_train, X_dev, X_test, y_train, y_dev, vocab_enc, tag_enc, tag_dec, tags = load_data(loc)

    # how many samples per batch to load
    batch_size = 16

    X_train_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent(x,vocab_enc)) for x in X_train], batch_first=True, padding_value=vocab_enc['<pad>'])
    X_train_pad = [[x,len(l)] for x,l in zip(X_train_pad,X_train)]
    y_train_torch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_tags(y_,tag_enc)) for y_ in y_train], batch_first=True, padding_value=tag_enc['<pad>'])

    X_dev_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent(x,vocab_enc)) for x in X_dev], batch_first=True, padding_value=vocab_enc['<pad>'])
    X_dev_pad = [[x,len(l)] for x,l in zip(X_dev_pad,X_dev)]
    y_dev_torch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_tags(y_,tag_enc)) for y_ in y_dev], batch_first=True, padding_value=tag_enc['<pad>'])

    X_test_pad = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(enc_sent(x,vocab_enc)) for x in X_test], batch_first=True, padding_value=vocab_enc['<pad>'])
    X_test_pad = [[x,len(l)] for x,l in zip(X_test_pad,X_test)]

    train_data = Model_Data(X_train_pad,y_train_torch)
    dev_data = Model_Data(X_dev_pad,y_dev_torch)
    test_data = Model_Data(X_test_pad,[])

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    print("Preparing Model...")
    # Hyperparameters
    vocab_size=len(vocab_enc.keys())
    pad_idx=vocab_enc['<pad>']
    emb_dim=100 
    hidden_size=256 
    num_layers=1
    linear_op_size=128
    num_classes=len(tag_enc.keys())-1 # exclude the tag class
    dropout = 0.33
    pad_idx = vocab_enc['<pad>']
    class_weights = torch.tensor([(1-v/sum(tags.values())) for v in tags.values()]).to(device)

    blstmClassifier = BLSTM(vocab_size, emb_dim, hidden_size, num_layers, linear_op_size, num_classes, dropout, pad_idx) 
    blstmClassifier.to(device)
    print(blstmClassifier)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=tag_enc['<pad>'])
    optimizer = torch.optim.SGD(blstmClassifier.parameters(), lr=0.1, momentum=0.9) 
    n_epochs = 60

    if args.mode=="train":
        # Train the model
        print("Training Model for {} epochs on device: {}...".format(n_epochs, device))
        train_f1, dev_f1, losses = train(n_epochs,num_classes,blstmClassifier,criterion,optimizer,tag_enc,train_loader,dev_loader,'blstm1.pt')

    elif args.mode=="test":
        print("Inferring from saved model...")
        # Loading the best model and inferring on the dev_data and test_data
        blstmClassifier.load_state_dict(torch.load('blstm1.pt', map_location=torch.device(device)))

        makeOutput(blstmClassifier,dev_loader,X_dev,tag_dec,"dev1.out")
        makeOutput(blstmClassifier,test_loader,X_test,tag_dec,"test1.out")

        print("Model predictions on dev_set and test_set saved.")