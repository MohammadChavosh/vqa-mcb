import sys
import numpy as np
import tensorflow as tf

sys.path.append("tensorflow_compact_bilinear_pooling/")
from compact_bilinear_pooling import compact_bilinear_pooling_layer as cbp

# Our implementation of ConvNet layers and Compact Bilinear in TensorFlow
import cnn

def signed_sqrt(t):
    return tf.multiply(tf.sign(t), tf.sqrt(tf.abs(t)))

def vqa_mcb_model(word_indices, glove_vector, seq_length, img_feature,
                  batch_size, num_vocab, embed_dim, glove_dim, max_time,
                  lstm_output_dim, lstm_layers, feat_h, feat_w,
                  img_feat_dim, cbp0_dim, cbp1_dim, num_classes,
                  cbp0_rand={"h1": None, "s1": None, "h2": None, "s2": None},
                  cbp1_rand={"h1": None, "s1": None, "h2": None, "s2": None},
                  apply_dropout=False, keep_prob=0.7,
                  name="vqa_mcb", compute_size=128):

    # ---- VQA Net in TensorFlow ---
    with tf.variable_scope(name):

        # ---- Part 0: LSTM net ----
        with tf.variable_scope('embedding'):
            embed_W = tf.get_variable("weights", [num_vocab, embed_dim])
            embed_b = tf.get_variable("biases", [embed_dim])

        embed_ba = tf.nn.embedding_lookup(embed_W, word_indices) + embed_b
        embed = tf.tanh(embed_ba)
        lstm_input = tf.concat([embed, glove_vector], axis=2)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_output_dim,
            forget_bias=0.0, state_is_tuple=True)
        # Add dropout if specified
        # According to the implementation, dropout is added on the first to
        # second-last LSTM layer, but not added to the last LSTM layer.
        if apply_dropout and keep_prob < 1:
            bottom_lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                output_keep_prob=keep_prob)
        else:
            bottom_lstm_cell = lstm_cell
        cell_list = [bottom_lstm_cell] * (lstm_layers-1) + [lstm_cell]
        cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cell, lstm_input, seq_length,
                                           dtype=tf.float32, time_major=True,
                                           scope="lstm12")
        lstm12 = tf.concat([state[0].h, state[1].h], axis=1)
        # Add dropout if specified
        if apply_dropout and keep_prob < 1:
             lstm12 = tf.nn.dropout(lstm12, keep_prob=keep_prob)

        q = tf.reshape(lstm12, [batch_size, 1, 1, lstm_output_dim*2])
        q_tile = tf.tile(q, (1, feat_h, feat_w, 1))

        # ---- Part 1: Attention ----
        cbp0 = cbp(q_tile, img_feature, cbp0_dim, sum_pool=False,
                   rand_h_1=cbp0_rand['h1'], rand_s_1=cbp0_rand['s1'],
                   rand_h_2=cbp0_rand['h2'], rand_s_2=cbp0_rand['s2'],
                   sequential=True, compute_size=compute_size)
        cbp0_sqrt = signed_sqrt(cbp0)
        cbp0_norm = tf.reshape(tf.nn.l2_normalize(tf.reshape(cbp0_sqrt,
            [batch_size, -1]), 1), [batch_size, feat_h, feat_w, cbp0_dim])
        # Add dropout if specified
        if apply_dropout and keep_prob < 1:
             cbp0_norm = tf.nn.dropout(cbp0_norm, keep_prob=keep_prob)

        # Two-layer ConvNet for attention map
        att_conv1 = cnn.conv_relu_layer("att_conv1", cbp0_norm, kernel_size=1,
                                        stride=1, output_dim=512)
        att_conv2 = cnn.conv_layer("att_conv2", att_conv1, kernel_size=1,
                                   stride=1, output_dim=2)

        # Softmax over conv output
        att_conv2_map0, att_conv2_map1 = tf.split(att_conv2, 2, axis=3)
        att_softmax0 = tf.reshape(tf.nn.softmax(tf.reshape(att_conv2_map0,
            [batch_size, -1])), [batch_size, feat_h, feat_w, 1])
        att_softmax1 = tf.reshape(tf.nn.softmax(tf.reshape(att_conv2_map1,
            [batch_size, -1])), [batch_size, feat_h, feat_w, 1])

        # Get attention feature
        att_feature0 = tf.reduce_sum(tf.multiply(img_feature, att_softmax0),
                                     reduction_indices=[1, 2])
        att_feature1 = tf.reduce_sum(tf.multiply(img_feature, att_softmax1),
                                     reduction_indices=[1, 2])
        att_feature = tf.reshape(tf.concat([att_feature0, att_feature1], axis=1),
                                 [batch_size, 1, 1, img_feat_dim*2])

        # ---- Part 2: Prediction ----
        cbp1 = cbp(att_feature, q, cbp1_dim, sum_pool=True,
                   rand_h_1=cbp1_rand['h1'], rand_s_1=cbp1_rand['s1'],
                   rand_h_2=cbp1_rand['h2'], rand_s_2=cbp1_rand['s2'],
                   sequential=False)
        cbp1_sqrt = signed_sqrt(cbp1)
        cbp1_norm = tf.reshape(tf.nn.l2_normalize(cbp1_sqrt, 1),
                               [batch_size, cbp1_dim])
        # Add dropout if specified
        if apply_dropout and keep_prob < 1:
             cbp1_norm = tf.nn.dropout(cbp1_norm, keep_prob=keep_prob)

        # Class scores
        prediction = cnn.fc_layer("prediction", cbp1_norm,
                                  output_dim=num_classes)

    return prediction, att_softmax0, att_softmax1, q, att_feature
