from feature_extractor import FeatureExtractor
from vqa_model import VQAModel
from data_loader import get_vqa_data
import numpy as np
import cv2

VALID_ACTIONS = ['End', 'Upper_Up', 'Upper_Down', 'Bottom_Up', 'Bottom_Down', 'Left_Left', 'Left_Right', 'Right_Left', 'Right_Right']
IS_TRAIN = True


class Environment:
	vqa_data = get_vqa_data(IS_TRAIN)
	data_num = 0

	def __init__(self):
		question, answer, img_path = Environment.vqa_data[Environment.data_num]
		Environment.data_num += 1
		if Environment.data_num % 100 == 0:
			print "{} number of {} data passed".format(Environment.data_num, len(Environment.vqa_data))
		if Environment.data_num == len(Environment.vqa_data):
			print "Total dataset passed once"
			Environment.data_num = 0
		self.img_array = cv2.imread(img_path)
		self.question = question
		self.answer = answer
		self.steps = 0
		img_size = self.img_array.shape
		self.crop_coordinates = [0, 0, img_size[0], img_size[1]]
		self.x_alpha = img_size[0] / 10.0
		self.y_alpha = img_size[1] / 10.0
		self.img_features = self.get_resized_region_image_features()
		self.latest_loss, self.latest_accuracy, self.state, _ = VQAModel.get_result(self.img_features, self.question, self.answer)
		self.state = np.concatenate((self.state, np.array([[(self.steps / 80.0)]])), axis=1)

		self.TRIGGER_NEGATIVE_REWARD = -3
		self.TRIGGER_POSITIVE_REWARD = 3
		self.MOVE_NEGATIVE_REWARD = -1
		self.MOVE_POSITIVE_REWARD = 1

	def action(self, action_type):
		self.steps += 1
		if self.steps > 80:
			return self.TRIGGER_NEGATIVE_REWARD * 2, True
		if action_type not in self.valid_actions():
			return self.MOVE_NEGATIVE_REWARD, False
		if action_type == 'End':
			if self.latest_accuracy < 0.1:
				return self.TRIGGER_NEGATIVE_REWARD, True
			if self.latest_accuracy > 0.9:
				return self.TRIGGER_POSITIVE_REWARD, True
		img_size = self.img_array.shape
		if action_type == 'Upper_Up':
			self.crop_coordinates[0] = max(self.crop_coordinates[0] - self.x_alpha, 0)
		if action_type == 'Upper_Down':
			self.crop_coordinates[0] = min(self.crop_coordinates[0] + self.x_alpha, img_size[0])
		if action_type == 'Bottom_Up':
			self.crop_coordinates[2] = max(self.crop_coordinates[2] - self.x_alpha, 0)
		if action_type == 'Bottom_Down':
			self.crop_coordinates[2] = min(self.crop_coordinates[2] + self.x_alpha, img_size[0])
		if action_type == 'Left_Left':
			self.crop_coordinates[1] = max(self.crop_coordinates[1] - self.y_alpha, 0)
		if action_type == 'Left_Right':
			self.crop_coordinates[1] = min(self.crop_coordinates[1] + self.y_alpha, img_size[1])
		if action_type == 'Right_Left':
			self.crop_coordinates[3] = max(self.crop_coordinates[3] - self.y_alpha, 0)
		if action_type == 'Right_Right':
			self.crop_coordinates[3] = min(self.crop_coordinates[3] + self.y_alpha, img_size[1])
		self.img_features = self.get_resized_region_image_features()
		loss, self.latest_accuracy, self.state, _ = VQAModel.get_result(self.img_features, self.question, self.answer)
		self.state = np.concatenate((self.state, np.array([[(self.steps / 80.0)]])), axis=1)
		if self.latest_loss > loss:
			self.latest_loss = loss
			return self.MOVE_POSITIVE_REWARD, False
		else:
			self.latest_loss = loss
			return self.MOVE_NEGATIVE_REWARD, False

	def get_resized_region_image_features(self):
		rounded_coordinates = map(lambda x: int(round(x)), self.crop_coordinates)
		img = self.img_array[rounded_coordinates[0]:rounded_coordinates[2], rounded_coordinates[1]:rounded_coordinates[3], :]
		return FeatureExtractor.extract_image_feature(img)

	def valid_actions(self):
		img_size = self.img_array.shape
		result = ['End']
		if self.crop_coordinates[0] - self.x_alpha >= 0:
			result.append('Upper_Up')
		if (self.crop_coordinates[0] + self.x_alpha + 1 < self.crop_coordinates[2]) and \
				(self.crop_coordinates[0] + self.x_alpha <= img_size[0]):
			result.append('Upper_Down')

		if (self.crop_coordinates[2] - self.x_alpha >= 0) and \
				(self.crop_coordinates[2] - self.x_alpha - 1 > self.crop_coordinates[0]):
			result.append('Bottom_Up')
		if self.crop_coordinates[2] + self.x_alpha <= img_size[0]:
			result.append('Bottom_Down')

		if self.crop_coordinates[1] - self.y_alpha >= 0:
			result.append('Left_Left')
		if (self.crop_coordinates[1] + self.y_alpha + 1 < self.crop_coordinates[3]) and \
				(self.crop_coordinates[1] + self.y_alpha <= img_size[1]):
			result.append('Left_Right')

		if self.crop_coordinates[3] - self.y_alpha >= 0 and \
				(self.crop_coordinates[3] - self.y_alpha - 1 > self.crop_coordinates[1]):
			result.append('Right_Left')
		if self.crop_coordinates[3] + self.y_alpha <= img_size[1]:
			result.append('Right_Right')

		return result

	def reset(self):
		question, answer, img_path = Environment.vqa_data[Environment.data_num]
		Environment.data_num += 1
		if Environment.data_num % 100 == 0:
			print "{} number of {} data passed".format(Environment.data_num, len(Environment.vqa_data))
		if Environment.data_num == len(Environment.vqa_data):
			print "Total dataset passed once"
			Environment.data_num = 0
		self.img_array = cv2.imread(img_path)
		self.question = question
		self.answer = answer
		self.steps = 0
		img_size = self.img_array.shape
		self.crop_coordinates = [0, 0, img_size[0], img_size[1]]
		self.img_features = self.get_resized_region_image_features()
		self.latest_loss, self.latest_accuracy, self.state, _ = VQAModel.get_result(self.img_features, self.question, self.answer)
		self.state = np.concatenate((self.state, np.array([[(self.steps / 80.0)]])), axis=1)
