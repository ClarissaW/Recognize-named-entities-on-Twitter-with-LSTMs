# Specify the network architecture based on TensorFlow building blocks
# LSTM network will produce probability distribution over tags for each token in a sentence
# Consider both right and left contexts of the token, so will use Bi-Directional LSTM (Bi-LSTM)
# Dense layer will be used on top to perform tag classification

#   Create placeholders for the followings
#    input_batch - sequences of words (the shape equals to [batch_size, sequence_len])
#    ground_truth_tags - sequences of tags (the shape equals to [batch_size, sequence_len])
#    lengths - lengths of not padded sequences (the shape equals to [batch_size])
#    dropout_ph - dropout keep probability; this placeholder has a predefined value 1
#    learning_rate_ph - learning rate; we need this placeholder because we want to change the value during training


"""
    Command:  pip install --user tensorflow
    https://www.tensorflow.org/versions/master/api_docs/python/tf/placeholder
"""
import tensorflow as tf
import numpy as np
class BiLSTMModel():
    pass

    def declare_placeholders(self):
        """Specifies placeholders for the model."""
        # A placeholder is used for feeding external data into a Tensorflow computation
        # tf.variable is used for storing a state
        
        # Placeholders for input and ground truth output.
        self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
        self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')

        # Placeholder for lengths of the sequences.
        self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')
        
        # Placeholder for a dropout keep probability. If we don't feed
        # a value for this placeholder, it will be equal to 1.0.
        # tf.placeholder_with_default to set a default value for placeholder
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
        
        # Placeholder for a learning rate (tf.float32).
        self.learning_rate_ph = tf.placeholder(dtype = tf.float32, shape=[])

        BiLSTMModel.__declare_placeholders = classmethod(declare_placeholders)


    def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
        """Specifies bi-LSTM architecture and computes logits for inputs."""
        
        # Create embedding variable (tf.Variable) with dtype tf.float32
        initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix_variable = tf.Variable(initial_embedding_matrix, dtype = tf.float32, name = 'embedding_matrix_variable')  # tf.Variable(initial_value, dtype, name)
        
        # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units
        # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
        forward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn),
                                                    input_keep_prob=self.dropout_ph,
                                                    output_keep_prob=self.dropout_ph,
                                                    state_keep_prob=self.dropout_ph)
                                                     
                                                     
        backward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn),
                                                    input_keep_prob=self.dropout_ph,
                                                    output_keep_prob=self.dropout_ph,
                                                    state_keep_prob=self.dropout_ph)
        
        # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
        # Shape: [batch_size, sequence_len, embedding_dim].
        embeddings = tf.nn.embedding_lookup(embedding_matrix_variable, self.input_batch)
        
        # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
        # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn].
        # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
        (rnn_output_fw, rnn_output_bw), _ =  tf.nn.bidirectional_dynamic_rnn(
                cell_fw = forward_cell,cell_bw = backward_cell, inputs = embeddings,
                dtype=tf.float32, sequence_length=self.lengths)
        rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
        
        # Dense layer on top.
        # Shape: [batch_size, sequence_len, n_tags].
        self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)
