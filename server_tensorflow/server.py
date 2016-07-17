import os
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
from time import time
import cv2
import hashlib
import numpy as np
import skimage.io
from skimage.transform import resize
import tensorflow as tf

# Our implementation of ConvNet layers and Compact Bilinear in TensorFlow
from vqa_mcb_model import vqa_mcb_model

# For ResNet, we (currently) still use Caffe to extract image features.
# We DO NOT need the vqa Caffe branch. You can use standard Caffe for ResNet.
import caffe

# For loading data, we use the LoadVQADataProvider in original code
import vqa_data_provider_layer
from vqa_data_provider_layer import LoadVQADataProvider

# constants
GPU_ID = 3
RESNET_MEAN_PATH = "../preprocess/ResNet_mean.binaryproto"
RESNET_LARGE_PROTOTXT_PATH = "../preprocess/ResNet-152-448-deploy.prototxt"
RESNET_CAFFEMODEL_PATH = "../data/resnet/ResNet-152-model.caffemodel"
EXTRACT_LAYER = "res5c"
EXTRACT_LAYER_SIZE = (2048, 14, 14)
TARGET_IMG_SIZE = 448
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

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'])
UPLOAD_FOLDER = './uploads/'
VIZ_FOLDER = './viz/'

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

cbp0_rand = dict(np.load(CBP0_RAND_FILE))
cbp1_rand = dict(np.load(CBP1_RAND_FILE))

# global variables
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
feature_cache = {}

# Create a TensorFlow session that allows GPU memory growth.
# Otherwise it's going to take up all available memory
# on the machine.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Input and output Tensors
word_indices = None
glove_vector = None
seq_length = None
img_feature = None
prediction = None
att_softmax0 = None

# Caffe's ResNet for feature extraction
resnet_mean = None
resnet_net = None

# Data Loader
vqa_data_provider = LoadVQADataProvider(VDICT_PATH, ADICT_PATH, batchsize=1,
    mode='test', data_shape=EXTRACT_LAYER_SIZE)

# helpers
def setup():
    global resnet_mean
    global resnet_net

    global word_indices
    global glove_vector
    global seq_length
    global img_feature
    global prediction
    global att_softmax0

    # data provider
    vqa_data_provider_layer.CURRENT_DATA_SHAPE = EXTRACT_LAYER_SIZE

    # mean substraction
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( RESNET_MEAN_PATH , 'rb').read()
    blob.ParseFromString(data)
    resnet_mean = np.array(caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3,224,224)
    resnet_mean = np.transpose(cv2.resize(np.transpose(resnet_mean,(1,2,0)), (448,448)),(2,0,1))

    caffe.set_mode_gpu()

    resnet_net = caffe.Net(RESNET_LARGE_PROTOTXT_PATH, RESNET_CAFFEMODEL_PATH, caffe.TEST)

    # our VQA model in TensorFlow
    word_indices = tf.placeholder(tf.int32, [max_time, batch_size])
    glove_vector = tf.placeholder(tf.float32, [max_time, batch_size, glove_dim])
    seq_length = tf.placeholder(tf.int32, [batch_size])
    img_feature = tf.placeholder(tf.float32, [batch_size, feat_h, feat_w, img_feat_dim])
    prediction, att_softmax0, _ = vqa_mcb_model(word_indices, glove_vector,
        seq_length, img_feature, batch_size, num_vocab, embed_dim, glove_dim, max_time,
        lstm_output_dim, lstm_layers, feat_h, feat_w, img_feat_dim, cbp0_dim, cbp1_dim,
        num_classes, cbp0_rand, cbp1_rand, apply_dropout)

    saver = tf.train.Saver()
    saver.restore(sess, PRETRAINED_MODEL)

    # uploads
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists(VIZ_FOLDER):
        os.makedirs(VIZ_FOLDER)

    print('Finished setup')

def trim_image(img):
    y,x,c = img.shape
    if c != 3:
        raise Exception('Expected 3 channels in the image')
    resized_img = cv2.resize( img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    transposed_img = np.transpose(resized_img,(2,0,1)).astype(np.float32)
    ivec = transposed_img - resnet_mean
    return ivec

def make_rev_adict(adict):
    """
    An adict maps text answers to neuron indices. A reverse adict maps neuron
    indices to text answers.
    """
    rev_adict = {}
    for k,v in list(adict.items()):
        rev_adict[v] = k
    return rev_adict

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def softmax(arr):
    e = np.exp(arr)
    dist = e / np.sum(e)
    return dist

def downsample_image(img):
    img_h, img_w, img_c = img.shape
    img = resize(img, (448 * img_h / img_w, 448))
    return img

def save_attention_visualization(source_img_path, att_map, dest_name):
    """
    Visualize the attention map on the image and save the visualization.
    """
    img = cv2.imread(source_img_path) # cv2.imread does auto-rotate

    # downsample source image
    img = downsample_image(img)
    img_h, img_w, img_c = img.shape

    _, att_h, att_w, _ = att_map.shape
    att_map = att_map.reshape((att_h, att_w))

    # upsample attention map to match original image
    upsample0 = resize(att_map, (img_h, img_w), order=3) # bicubic interpolation
    upsample0 = upsample0 / upsample0.max()

    # create rgb-alpha
    rgba0 = np.zeros((img_h, img_w, img_c + 1))
    rgba0[..., 0:img_c] = img
    rgba0[..., 3] = upsample0

    path0 = os.path.join(VIZ_FOLDER, dest_name + '.png')
    cv2.imwrite(path0, rgba0 * 255.0)

    return path0

# routes
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('demo2.html')

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file was uploaded.'})
    if allowed_file(file.filename):
        start = time()
        file_hash = hashlib.md5(file.read()).hexdigest()
        if file_hash in feature_cache:
            json = {'img_id': file_hash, 'time': time() - start}
            return jsonify(json)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file_hash + '.jpg')
        file.seek(0)
        file.save(save_path)
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({'error': 'Error reading image.'})
        small_img = downsample_image(img)
        cv2.imwrite(save_path, small_img * 255.0)
        preprocessed_img = trim_image(img)
        resnet_net.blobs['data'].data[0,...] = preprocessed_img
        resnet_net.forward()
        feature = resnet_net.blobs[EXTRACT_LAYER].data[0].reshape(EXTRACT_LAYER_SIZE)
        feature = ( feature / np.sqrt((feature**2).sum()) )
        feature_cache[file_hash] = feature
        json = {'img_id': file_hash, 'time': time() - start}
        return jsonify(json)
    else:
        return jsonify({'error': 'Please upload a JPG or PNG.'})

@app.route('/api/upload_question', methods=['POST'])
def upload_question():
    img_hash = request.form['img_id']
    if img_hash not in feature_cache:
        return jsonify({'error': 'Unknown image ID. Try uploading the image again.'})
    start = time()
    img_feature_val = feature_cache[img_hash]
    question = request.form['question']
    img_ques_hash = hashlib.md5(img_hash + question).hexdigest()
    qvec, cvec, avec, glove_matrix = vqa_data_provider.create_batch(question)

    # In TensorFlow, instead of putting the question at the end
    # we put it at the beginning, and perform dynamic computation
    valid_time_idx = qvec.reshape(-1) > 0
    word_indices_val = qvec[:, valid_time_idx].T
    glove_vector_val = glove_matrix[:, valid_time_idx, :].transpose((1,0,2))
    seq_length_val = np.sum(valid_time_idx)
    # Put dummy zeros after the sequence length using dynamic_rnn
    word_indices_val = np.concatenate((word_indices_val,
        np.zeros((max_time-seq_length_val, 1)))).astype(np.int32)
    glove_vector_val = np.concatenate((glove_vector_val,
        np.zeros((max_time-seq_length_val, 1, glove_dim))))
    img_feature_val = img_feature_val[np.newaxis, ...].transpose((0, 2, 3, 1))

    # Forward Pass
    scores, att_map = sess.run([prediction, att_softmax0],
                               feed_dict={word_indices: word_indices_val,
                                          glove_vector: glove_vector_val,
                                          seq_length: [seq_length_val],
                                          img_feature: img_feature_val})
    scores = softmax(np.squeeze(scores))
    top_indices = scores.argsort()[::-1][:5]
    top_answers = [vqa_data_provider.vec_to_answer(i) for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]

    # attention visualization
    source_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_hash + '.jpg')
    path0 = save_attention_visualization(source_img_path, att_map, img_ques_hash)

    json = {'answer': top_answers[0],
        'answers': top_answers,
        'scores': top_scores,
        'viz': [path0],
        'time': time() - start}
    return jsonify(json)

@app.route('/viz/<filename>')
def get_visualization(filename):
    return send_from_directory(VIZ_FOLDER, filename)

if __name__ == '__main__':
    setup()
    app.run(host='0.0.0.0', port=5000, debug=False)
