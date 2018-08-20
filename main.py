####################################   Load the data from file. ####################################
import read_data

train_tokens, train_tags = read_data.read_data('data/train.txt')
validation_tokens, validation_tags = read_data.read_data('data/validation.txt')
test_tokens, test_tags = read_data.read_data('data/test.txt')


#Print the first three to check the current result
for i in range(3):
    for token, tag in zip(train_tokens[i], train_tags[i]):
        print('%s\t%s' % (token, tag))
    print()


####################################  Prepare dictionaries ####################################
import prepare_dictionaries

special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

# Create dictionaries and the word bank based on the data

token2idx, idx2token = prepare_dictionaries.build_dict(train_tokens + validation_tokens, special_tokens)
tag2idx, idx2tag = prepare_dictionaries.build_dict(train_tags, special_tags)


####################################  Generate batches ####################################

def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]


#Help create the mapping between tokens and ids for a sentence.
import numpy as np
"""
    Neural Networks are usually trained with batches. It means that weight updates of the network are based on several sequences at every single time. The tricky part is that all sequences within a batch need to have the same length. So we will pad them with a special <PAD> token. It is also a good practice to provide RNN with sequence lengths, so it can skip computations for padding parts. We provide the batching function batches_generator readily available for you to save time.
    """

def batches_generator(batch_size, tokens, tags, shuffle=True, allow_smaller_last_batch=True):
    """Generates padded batches of tokens and tags."""
    
    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)  #make it as random order
    else:
        order = np.arange(n_samples) #numpy.arange([start, ]stop, [step, ]dtype=None)
    
    
    n_batches = n_samples // batch_size  # // means divide with integral result (discard remainder)
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1
    
    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))
        
        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths



################################################  Evaluation  ################################################

#from recsys.evaluation.decision import PrecisionRecallF1
from evaluation import precision_recall_f1
# no module named evaluation???
from sklearn.metrics import precision_recall_fscore_support

def predict_tags(model, session, token_idxs_batch, lengths):
    """Performs predictions and transforms indices to tokens and tags."""
    
    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)
    
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch

def eval_conll(model, session, tokens, tags, short_report=True):
    """Computes NER quality measures using CONLL shared task script."""
    
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]):
            if token != '<PAD>':
                ground_truth_tags.append(idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)
    
        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])

    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
#    results = precision_recall_fscore_support(y_true, y_pred,average='weighted')

    return results



################################################  LSTM  ################################################
import LSTM

import tensorflow as tf
vocabulary_size = len(idx2token)
n_tags = len(idx2tag)
embedding_dim = 200
n_hidden_rnn = 200
PAD_index = token2idx['<PAD>']

tf.reset_default_graph()

model = LSTM.BiLSTMModel(vocabulary_size,n_tags,embedding_dim,n_hidden_rnn,PAD_index)

#from LSTM import BiLSTMModel
#model = BiLSTMModel(vocabulary_size,n_tags,embedding_dim,n_hidden_rnn,PAD_index)


batch_size = 32
n_epochs = 4
learning_rate = 0.005
learning_rate_decay = 1.414
dropout_keep_probability =  0.5

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, train_tokens, train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)
    
    # Train the model
    for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)
    
    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay

print('...training finished.')


""" When using precision_recall_f1, the result is as following:
    Start training...
    
    -------------------- Epoch 1 of 4 --------------------
    Train data evaluation:
    processed 105778 tokens with 4489 phrases; found: 62528 phrases; correct: 107.
    
    precision:  0.17%; recall:  2.38%; F1:  0.32
    
    Validation data evaluation:
    processed 12836 tokens with 537 phrases; found: 7846 phrases; correct: 19.
    
    precision:  0.24%; recall:  3.54%; F1:  0.45
    
    -------------------- Epoch 2 of 4 --------------------
    Train data evaluation:
    processed 105778 tokens with 4489 phrases; found: 2885 phrases; correct: 516.
    
    precision:  17.89%; recall:  11.49%; F1:  14.00
    
    Validation data evaluation:
    processed 12836 tokens with 537 phrases; found: 239 phrases; correct: 44.
    
    precision:  18.41%; recall:  8.19%; F1:  11.34
    
    -------------------- Epoch 3 of 4 --------------------
    Train data evaluation:
    processed 105778 tokens with 4489 phrases; found: 4686 phrases; correct: 2013.
    
    precision:  42.96%; recall:  44.84%; F1:  43.88
    
    Validation data evaluation:
    processed 12836 tokens with 537 phrases; found: 359 phrases; correct: 139.
    
    precision:  38.72%; recall:  25.88%; F1:  31.03
    
    -------------------- Epoch 4 of 4 --------------------
    Train data evaluation:
    processed 105778 tokens with 4489 phrases; found: 4868 phrases; correct: 2866.
    
    precision:  58.87%; recall:  63.84%; F1:  61.26
    
    Validation data evaluation:
    processed 12836 tokens with 537 phrases; found: 410 phrases; correct: 171.
    
    precision:  41.71%; recall:  31.84%; F1:  36.11
    
    ...training finished.
    
"""
######################################  Set Quality  ########################################

print('-' * 20 + ' Train set quality: ' + '-' * 20)
train_results = eval_conll(model, sess, train_tokens, train_tags, short_report=False)

print('-' * 20 + ' Validation set quality: ' + '-' * 20)
validation_results = eval_conll(model,sess,validation_tokens, validation_tags, short_report=False)

print('-' * 20 + ' Test set quality: ' + '-' * 20)
test_results = eval_conll(model,sess,test_tokens,test_tags, short_report=False)
