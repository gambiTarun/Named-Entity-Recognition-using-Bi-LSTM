# Named-Entity-Recognition-using-Bi-LSTM-CNN
In this repository, I introduce a unique neural network design that seamlessly identifies word and character-level attributes through a combined bidirectional LSTM and CNN structure, significantly reducing the necessity for extensive feature engineering.

# Simple Bidirectional LSTM model

The first model is a simple bidirectional LSTM model for NER. The architecture of the network is:

Embedding → BLSTM → Linear → ELU → classifier

Training this simple BLSTM model with the training data on NER with SGD as the optimizer.

# Using GloVe word embeddings

The second model has the same basic architecture, however, incorporates the use of GloVe word embeddings to improve the simple BLSTM. The way I use the GloVe word embeddings is straight forward: I initialize the embeddings in the neural network with the corresponding vectors in GloVe. Note that GloVe is case-insensitive, but the NER model should be case-sensitive because capitalization is an important information for NER. I come up with a novel way to deal with this conflict and encode the casing information withing the embedding layer.

# Bi-LSTM-CNN model

The model equips the BLSTM model with a CNN module to capture character-level information. The character embedding dimension is set to 30. I tune other hyper-parameters of CNN module, such as the number of CNN layers, the kernel size and output dimension of each CNN layer.

# Evaluation using CoNLL eval script

Use the official evaluation script conll03eval to evaluate the results of the model. To use the script, you need to install perl and prepare your prediction file in the following format:

idx word gold pred

where there is a white space between two columns. gold is the gold-standard NER tag and pred is the model-predicted tag. Then execute the command line:
```
perl conll03eval < {predicted file}
```
where {predicted file} is the prediction file in the prepared format.


# Train/Test the models

## Simple Bidirectional LSTM model

### Training 

To run the training of the model for task1, run the "task1.py" as following:

```
    python BLSTM.py -m train
```


**Make sure to have "data/" (folder with train, dev, and test) in the current directory.**

This will generate "blstm1.pt" for the trained model.

![BLSTM](https://user-images.githubusercontent.com/22619455/232166868-c3380239-491f-44c4-b856-b44bd50b3470.png)

### Testing 

To run the testing on the model for task1, run the "task1.py" as following:

```
    python BLSTM.py -m test
```

**Make sure to have "blstm1.pt", "data/" (folder with train, dev, and test) in the current directory.**

This will generate "dev1.out" (dev_set predictions) and "test1.out" (test_set predictions) using the saved weights.

## Bi-LSTM model with GloVe word embeddings

### Training 

To run the training of the model for task2, run the "task2.py" as following:

```
    python BLSTMGlove.py -m train
```

**Make sure to have "task1.py", "glove.6B.100d.gz", "data/" (folder with train, dev, and test) in the current directory.**

This will generate "blstm2.pt" for the trained model.

![BLSTMGlove](https://user-images.githubusercontent.com/22619455/232166893-f890fa44-fd5b-46e6-af6f-99da24c0d041.png)

### Testing 

To run the testing on the model for task2, run the "task2.py" as following:

```
    python BLSTMGlove.py -m test
```

**Make sure to have "task1.py", "glove.6B.100d.gz", "blstm2.pt", "data/" (folder with train, dev, and test) in the current directory.**

This will generate "dev2.out" (dev_set predictions) and "test2.out" (test_set predictions) using the saved weights.


## Bi-LSTM-CNN model

### Training 

To run the training of the model for task2, run the "bonus.py" as following:
```
    python BLSTM_CNN.py -m train
```

**Make sure to have "task1.py", "task2.py", "glove.6B.100d.gz", "data/" (folder with train, dev, and test) in the current directory.**

This will generate "blstm_cnn.pt" for the trained model.

<img width="995" alt="image" src="https://user-images.githubusercontent.com/22619455/232168270-5119157b-1763-4d61-ae68-3211d66fce41.png">

### Testing 

To run the testing on the model for bonus, run the "bonus.py" as following:
```
    python BLSTM_CNN.py -m test
```

**Make sure to have "task1.py", "task2.py", "glove.6B.100d.gz", "blstm_cnn.pt", "data/" (folder with train, dev, and test) in the current directory.**

This will generate "dev_pred" (dev_set predictions) and "pred" (test_set predictions) using the saved weights.

# References

- [PyTorch Resources](https://pytorch.org/docs/stable/nn.html)
- Jason P. C. Chiu, Eric Nichols: “Named Entity Recognition with Bidirectional LSTM-CNNs”, 2015; [http://arxiv.org/abs/1511.08308 arXiv:1511.08308].
