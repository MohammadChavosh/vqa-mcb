import sys
import os
import numpy as np
import tensorflow as tf
import time
from inspect import getsourcefile
from env import VALID_ACTIONS, IS_TRAIN, Environment

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
	sys.path.append(import_path)

from estimators import PolicyEstimator
from worker import make_copy_params_op


class PolicyMonitor(object):
	"""
	Helps evaluating a policy by running an episode in an environment,
	saving a video, and plotting summaries to Tensorboard.

	Args:
	  env: environment to run in
	  policy_net: A policy estimator
	  summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
	"""

	def __init__(self, env, policy_net, summary_writer, saver=None):

		# self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
		# self.video_dir = os.path.abspath(self.video_dir)

		# self.env = Monitor(env, directory=self.video_dir, video_callable=lambda x: True, resume=True)
		self.env = env
		self.global_policy_net = policy_net
		self.summary_writer = summary_writer
		self.saver = saver

		self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))

		# os.makedirs(self.video_dir)

		# Local policy net
		with tf.variable_scope("policy_eval"):
			self.policy_net = PolicyEstimator(policy_net.num_outputs)

		# Op to copy params from global policy/value net parameters
		self.copy_params_op = make_copy_params_op(
			tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
			tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

	def _policy_net_predict(self, state, sess):
		feed_dict = {self.policy_net.states: state}
		preds = sess.run(self.policy_net.predictions, feed_dict)
		return preds["probs"][0]

	def eval_once(self, sess):
		with sess.as_default(), sess.graph.as_default():
			# Copy params to local model
			global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

			# Run an episode
			done = False
			self.env.reset()
			accuracy = self.env.latest_accuracy
			state = self.env.state
			total_reward = 0.0
			episode_length = 0
			actions = []
			while not done:
				action_probs = self._policy_net_predict(state, sess)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
				actions.append(VALID_ACTIONS[action])
				reward, done = self.env.action(VALID_ACTIONS[action])
				next_state = self.env.state
				total_reward += reward
				episode_length += 1
				state = next_state

			# Add summaries
			episode_summary = tf.Summary()
			episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
			episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
			self.summary_writer.add_summary(episode_summary, global_step)
			self.summary_writer.flush()

			if self.saver is not None:
				self.saver.save(sess, self.checkpoint_path)

			print "Eval results at step {}: first_accuracy {}, last_reward {}, total_reward {}, episode_length {}".format(global_step, accuracy, reward, total_reward, episode_length)
			print "Actions: {}".format(actions)

			return total_reward, episode_length, accuracy, reward, (self.env.img_path, self.env.question, self.env.answer, episode_length, actions)

	def continuous_eval(self, eval_every, sess, coord):
		"""
		Continuously evaluates the policy every [eval_every] seconds.
		"""
		accuracies = []
		episode_lengths = []
		corrected = 0
		wronged = 0
		try:
			while not coord.should_stop():
				_, episode_length, first_accuracy, reward, data = self.eval_once(sess)
				episode_lengths.append(episode_length)
				if reward == 3:
					accuracies.append(1.0)
				else:
					accuracies.append(0.0)
				if reward == 3 and first_accuracy < 0.1:
					corrected += 1
					with open("corrections.txt", "a") as f:
						f.write("Corrected data: {}\n".format(data))
				elif reward == -3 and first_accuracy > 0.9:
					wronged += 1
					with open("wrongs.txt", "a") as f:
						f.write("Wronged data: {}\n".format(data))
				print "Till now accuracy: {}, corrected: {}, wronged: {}, improved: {}, processed: {}, avg_episode_length: {}".format(sum(accuracies) / len(accuracies), corrected, wronged, float(corrected - wronged) / len(accuracies), len(accuracies), float(sum(episode_lengths))/len(episode_lengths))
				if (not IS_TRAIN) and (len(accuracies) > Environment.data_num):
					break
				# Sleep until next evaluation cycle
				if IS_TRAIN:
					time.sleep(eval_every)
		except tf.errors.CancelledError:
			return
