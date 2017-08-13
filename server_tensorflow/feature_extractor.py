import cv2
import numpy as np

import sys; sys.path.append('/home/mll/Chavosh/caffe/python')
import caffe

RESNET_MEAN_PATH = "../preprocess/ResNet_mean.binaryproto"
RESNET_LARGE_PROTOTXT_PATH = "../preprocess/ResNet-152-448-deploy.prototxt"
RESNET_CAFFEMODEL_PATH = "../data/resnet/ResNet-152-model.caffemodel"
EXTRACT_LAYER = "res5c"
EXTRACT_LAYER_SIZE = (2048, 14, 14)
TARGET_IMG_SIZE = 448


class FeatureExtractor:
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(RESNET_MEAN_PATH, 'rb').read()
	blob.ParseFromString(data)
	resnet_mean = np.array(caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3, 224, 224)
	resnet_mean = np.transpose(cv2.resize(np.transpose(resnet_mean, (1, 2, 0)), (448, 448)), (2, 0, 1))
	caffe.set_mode_gpu()
	resnet_net = caffe.Net(RESNET_LARGE_PROTOTXT_PATH, RESNET_CAFFEMODEL_PATH, caffe.TEST)

	def __init__(self):
		pass

	@staticmethod
	def trim_image(img):
		#Grayscale
		if len(img.shape) == 2:
			img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='float32')
			img_new[:, :, 0] = img
			img_new[:, :, 1] = img
			img_new[:, :, 2] = img
			img = img_new
		resized_img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
		transposed_img = np.transpose(resized_img, (2, 0, 1)).astype(np.float32)
		ivec = transposed_img - FeatureExtractor.resnet_mean
		return ivec

	@staticmethod
	def extract_image_feature(img):
		preprocessed_img = FeatureExtractor.trim_image(img)
		FeatureExtractor.resnet_net.blobs['data'].data[0, ...] = preprocessed_img
		FeatureExtractor.resnet_net.forward()
		feature = FeatureExtractor.resnet_net.blobs[EXTRACT_LAYER].data[0].reshape(EXTRACT_LAYER_SIZE)
		return feature / np.sqrt((feature ** 2).sum())
