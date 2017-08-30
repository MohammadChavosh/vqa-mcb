import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss

import vqa_data_provider_layer
from vqa_data_provider_layer import LoadVQADataProvider
from feature_extractor import EXTRACT_LAYER_SIZE
from vqa_mcb_model import vqa_mcb_model

# constants
GPU_ID = 0
VDICT_PATH = "../data/multi_att_2_glove_pretrained/vdict.json"
ADICT_PATH = "../data/multi_att_2_glove_pretrained/adict.json"

# To reproduce exactly the same output, the random vectors
# In Caffe's two CompactBilinear layers are extracted
# For training from scratch, you may use other random vectors
# or leave unspecified
CBP0_RAND_FILE = './tf_vqa_data/cbp0_rand.npz'
CBP1_RAND_FILE = './tf_vqa_data/cbp1_rand.npz'

# Converted from the corresponding Caffe model
PRETRAINED_MODEL = './tf_vqa_data/_iter_190000.tfmodel'

# Set GPU with CUDA_VISIBLE_DEVICES environment
# Effective for both TensorFlow and Caffe
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

# --- VQA Model Params ---
batch_size = 1
num_vocab = 21025
embed_dim = 300
glove_dim = 300
max_time = 15
lstm_output_dim = 1024
lstm_layers = 2
feat_h, feat_w = 14, 14
img_feat_dim = 2048
cbp0_dim = 16000
cbp1_dim = 16000
num_classes = 3000
apply_dropout = False


class VQAModel:
	vqa_data_provider = LoadVQADataProvider(VDICT_PATH, ADICT_PATH, batchsize=1, data_shape=EXTRACT_LAYER_SIZE)
	vqa_data_provider_layer.CURRENT_DATA_SHAPE = EXTRACT_LAYER_SIZE
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	cbp0_rand = dict(np.load(CBP0_RAND_FILE))
	cbp1_rand = dict(np.load(CBP1_RAND_FILE))

	word_indices = tf.placeholder(tf.int32, [max_time, batch_size])
	glove_vector = tf.placeholder(tf.float32, [max_time, batch_size, glove_dim])
	seq_length = tf.placeholder(tf.int32, [batch_size])
	img_feature = tf.placeholder(tf.float32, [batch_size, feat_h, feat_w, img_feat_dim])
	prediction, att_softmax0, _, question_features, att_features = vqa_mcb_model(word_indices, glove_vector,
	                                            seq_length, img_feature, batch_size, num_vocab, embed_dim, glove_dim,
	                                            max_time,
	                                            lstm_output_dim, lstm_layers, feat_h, feat_w, img_feat_dim, cbp0_dim,
	                                            cbp1_dim,
	                                            num_classes, cbp0_rand, cbp1_rand, apply_dropout)
	saver = tf.train.Saver()
	saver.restore(sess, PRETRAINED_MODEL)

	def __init__(self):
		pass

	@staticmethod
	def softmax(arr):
		e = np.exp(arr)
		dist = e / np.sum(e)
		return dist

	@staticmethod
	def get_result(img_feature_val, question, answer):
		question = unicode(question)
		qvec, cvec, avec, glove_matrix = VQAModel.vqa_data_provider.create_batch(question)

		# In TensorFlow, instead of putting the question at the end
		# we put it at the beginning, and perform dynamic computation
		valid_time_idx = qvec.reshape(-1) > 0
		word_indices_val = qvec[:, valid_time_idx].T
		glove_vector_val = glove_matrix[:, valid_time_idx, :].transpose((1, 0, 2))
		seq_length_val = np.sum(valid_time_idx)
		# Put dummy zeros after the sequence length using dynamic_rnn
		word_indices_val = np.concatenate((word_indices_val, np.zeros((max_time - seq_length_val, 1)))).astype(np.int32)
		glove_vector_val = np.concatenate((glove_vector_val, np.zeros((max_time - seq_length_val, 1, glove_dim))))
		img_feature_val = img_feature_val[np.newaxis, ...].transpose((0, 2, 3, 1))

		# Forward Pass
		scores, att_map, q_features, att_features = VQAModel.sess.run([VQAModel.prediction, VQAModel.att_softmax0, VQAModel.question_features, VQAModel.att_features],
		                                    feed_dict={VQAModel.word_indices: word_indices_val,
		                                               VQAModel.glove_vector: glove_vector_val,
		                                               VQAModel.seq_length: [seq_length_val],
		                                               VQAModel.img_feature: img_feature_val})
		scores = VQAModel.softmax(np.squeeze(scores))
		soft_max = VQAModel.softmax(scores)
		prediction = np.argmax(scores)
		correct_answer = VQAModel.vqa_data_provider.answer_to_vec(answer)
		labels = np.zeros(scores.shape)
		labels[correct_answer] = 1.0
		loss = log_loss(labels, soft_max)
		accuracy = float(correct_answer == prediction)
		state = np.concatenate((q_features.reshape((1, -1)), att_features.reshape((1, -1)), soft_max.reshape((1, -1))), axis=1)
		return loss, accuracy, state, prediction
