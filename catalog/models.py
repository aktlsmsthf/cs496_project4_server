from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils import timezone
import uuid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Create your models here.
class Images(models.Model):
	owner = models.ForeignKey('auth.User', on_delete=models.CASCADE)
	id = models.IntegerField(primary_key=True)
	label = models.IntegerField(null=True)
	path = models.IntegerField(null=True)
	img = models.ImageField(upload_to="image/", null=True, blank=True)

	def __str__(self):
		return str(self.id)

class MnistNN(models.Model):
	owner = models.ForeignKey('auth.User', on_delete=models.CASCADE)
	path = models.CharField(max_length=200)

	def __str__(self):
		return self.path

	def work(self, train=False, image=None, username=None):
		im = Image.open('image/3.jpg')
		img = array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
		data = img.reshape([1, 784])
		data = 1-(data/255)

		learning_rate = 0.001
		training_epochs = 15
		batch_size = 100

		keep_prob = tf.placeholder(tf.float32)

		X = tf.placeholder(tf.float32, [None, 784])
		X_img = tf.reshape(X, [-1, 28, 28, 1])
		Y = tf.placeholder(tf.float32, [None, 10])

		W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
		L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1],padding='SAME')
		L1 = tf.nn.relu(L1)
		L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


		W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
		L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1],padding='SAME')
		L2 = tf.nn.relu(L2)
		L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

		W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
		L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1],padding='SAME')
		L3 = tf.nn.relu(L3)
		L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
		L3_flat = tf.reshape(L3, [-1, 128*4*4])

		W4 = tf.get_variable("W4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
		b4 = tf.Variable(tf.random_normal([625]))
		L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
		L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

		W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
		b5 = tf.Variable(tf.random_normal([10]))
		logits = tf.matmul(L4, W5)+b5

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
		optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

		saver = tf.train.Saver()
		init_op = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init_op)
		save_path = "./minist_softmax.ckpt"

		saver.restore(sess, save_path)

		if(not train):
			prediction = sess.run(tf.nn.softmax(logits, 1), feed_dict={X: data, keep_prob:1})
			p = prediction[0]
			print(p)
		else:
			batch_x, batch_y = mnist.train.next_batch(11000)
			image_list = Images.objects.filter(owner=username)
			for image in image_list:
				path = 'image/'+str(image.path)+'.jpg'
				print(path)
				im = Image.open(path)
				img = array(im.resize((28,28), Image.ANTIALIAS).convert("L"))
				data = img.reshape([1, 784])
				data = 1-(data/255)
				print(data)

			total_batch = int(mnist.train.num_examples/batch_size)
			'''
			for epoch in range(30):
				total_cost = 0
				for i in range(total_batch):
					batch_xs, batch_ys = mnist.train.next_batch(batch_size)
					_, cost_val = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:0.8})
				total_cost+=cost_val
				print('Epoch:','%04d' %(epoch+1), 'Avg. cost =', '{:3f}'.format(total_cost/total_batch))
			print("Training Finish")
			is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
			accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
			print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels, keep_prob:1}))
			'''
			



class TestData(models.Model):
	pred = models.IntegerField()

	def __str__(self):
		return self.ped


