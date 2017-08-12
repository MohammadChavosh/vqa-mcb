import json
import random
from vqa_model import ADICT_PATH


def get_vqa_data(is_train, sampling_ratio=1):
	with open(ADICT_PATH, 'r') as f:
		answer_vocab = json.load(f)
	if is_train:
		annotations = json.load(open('Data/mscoco_train2014_annotations.json'))['annotations']
		questions = json.load(open('Data/OpenEnded_mscoco_train2014_questions.json'))['questions']
		images_path = 'Data/train2014/COCO_train2014_'
	else:
		annotations = json.load(open('Data/mscoco_val2014_annotations.json'))['annotations']
		questions = json.load(open('Data/OpenEnded_mscoco_val2014_questions.json'))['questions']
		images_path = 'Data/val2014/COCO_val2014_'
	vqa_triplets = list()
	for question, annotation in zip(questions, annotations):
		answer = annotation['multiple_choice_answer']
		if answer not in answer_vocab:
			continue
		if question['question_id'] != annotation['question_id']:
			raise AssertionError("question id's are not equal")
		q = question['question']
		img_num = str(question['image_id'])
		img_path = images_path
		for i in range(12 - len(img_num)):
			img_path += '0'
		img_path += img_num + '.jpg'
		vqa_triplets.append((q, answer, img_path))
	if sampling_ratio < 1:
		vqa_triplets = random.sample(vqa_triplets, int(round(len(vqa_triplets) * sampling_ratio)))
	return vqa_triplets
