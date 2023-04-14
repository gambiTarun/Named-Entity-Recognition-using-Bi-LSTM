# Named-Entity-Recognition-using-Bi-LSTM-CNN
In this repository, we introduce a unique neural network design that seamlessly identifies word and character-level attributes through a combined bidirectional LSTM and CNN structure, significantly reducing the necessity for extensive feature engineering.

# Simple Bidirectional LSTM model
### Training 

To run the training of the model for task1, run the "task1.py" as following:

'''
    python task1.py -m train
'''

** Make sure to have "data/" (folder with train, dev, and test) in the current directory. **

This will generate "blstm1.pt" for the trained model.

### Testing 

To run the testing on the model for task1, run the "task1.py" as following:

'''
    python task1.py -m test
'''

** Make sure to have "blstm1.pt", "data/" (folder with train, dev, and test) in the current directory. **

This will generate "dev1.out" (dev_set predictions) and "test1.out" (test_set predictions) using the saved weights.

#######################################################



######################## Task2 ########################

########### Training ###########

To run the training of the model for task2, run the "task2.py" as following:

    python task2.py -m train

*** Make sure to have "task1.py", "glove.6B.100d.gz", "data/" (folder with train, dev, and test) in the current directory. ***

This will generate "blstm2.pt" for the trained model.

########### Testing ###########

To run the testing on the model for task2, run the "task2.py" as following:

    python task2.py -m test

*** Make sure to have "task1.py", "glove.6B.100d.gz", "blstm2.pt", "data/" (folder with train, dev, and test) in the current directory. ***

This will generate "dev2.out" (dev_set predictions) and "test2.out" (test_set predictions) using the saved weights.

#######################################################



######################## Bonus ########################

########### Training ###########

To run the training of the model for task2, run the "bonus.py" as following:

    python bonus.py -m train

*** Make sure to have "task1.py", "task2.py", "glove.6B.100d.gz", "data/" (folder with train, dev, and test) in the current directory. ***

This will generate "blstm_cnn.pt" for the trained model.

########### Testing ###########

To run the testing on the model for bonus, run the "bonus.py" as following:

    python bonus.py -m test

*** Make sure to have "task1.py", "task2.py", "glove.6B.100d.gz", "blstm_cnn.pt", "data/" (folder with train, dev, and test) in the current directory. ***

This will generate "dev_pred" (dev_set predictions) and "pred" (test_set predictions) using the saved weights.

#######################################################

