#! /usr/bin/env python

import sys
import os
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
from inspect import getsourcefile
from env import VALID_ACTIONS, IS_TRAIN, Environment

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
	sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker

tf.flags.DEFINE_string("model_dir", "Data/a3c", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 60, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS


def make_env():
	return Environment()

# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
	NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
	shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
	os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):
	# Keeps track of the number of updates we've performed
	global_step = tf.Variable(0, name="global_step", trainable=False)

	# Global policy and value nets
	with tf.variable_scope("global") as vs:
		policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS))
		value_net = ValueEstimator(reuse=True)

	# Global step iterator
	global_counter = itertools.count()

	# Create worker graphs
	workers = []
	if IS_TRAIN:
		for worker_id in range(NUM_WORKERS):
			# We only write summaries in one of the workers because they're
			# pretty much identical and writing them on all workers
			# would be a waste of space
			worker_summary_writer = None
			if worker_id == 0:
				worker_summary_writer = summary_writer

			worker = Worker(
				name="worker_{}".format(worker_id),
				env=make_env(),
				policy_net=policy_net,
				value_net=value_net,
				global_counter=global_counter,
				discount_factor=0.99,
				summary_writer=worker_summary_writer,
				max_global_steps=FLAGS.max_global_steps)
			workers.append(worker)

	saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

	# Used to occasionally save videos for our policy net
	# and write episode rewards to Tensorboard
	pe = PolicyMonitor(
		env=make_env(),
		policy_net=policy_net,
		summary_writer=summary_writer,
		saver=saver)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()

	# Load a previous checkpoint if it exists
	latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
	print 'checkpoint_dir: ', CHECKPOINT_DIR
	print 'latest_checkpoint: ', latest_checkpoint
	if latest_checkpoint:
		print("Loading model checkpoint: {}".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)

	# Start worker threads
	worker_threads = []
	for worker in workers:
		t = threading.Thread(target=lambda: worker.run(sess, coord, FLAGS.t_max))
		t.start()
		worker_threads.append(t)

	# Start a thread for policy eval task
	monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
	monitor_thread.start()

	# Wait for all workers to finish
	if IS_TRAIN:
		coord.join(worker_threads)
	else:
		coord.join([monitor_thread])
