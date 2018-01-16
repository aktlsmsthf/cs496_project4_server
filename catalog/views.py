from django.shortcuts import render
from django.views import generic
from django.views.decorators.csrf import csrf_exempt
import base64
from django.core.files.base import ContentFile
from django.shortcuts import render_to_response
from django.template import RequestContext
from .models import  Images, MnistNN
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Max
from PIL import Image
import tensorflow as tf
from time import sleep
import json
import datetime
import random
import time
import scipy.misc
from numpy import array, argmax


# Create your views here.




@login_required
def mypage(request):
	num_image = Images.objects.filter(owner=request.user).count()
	image_list = Images.objects.filter(owner=request.user)
	print(image_list.values('img'))

	return render(
		request,
		'mypage.html',
		context = {'num_images':num_image, 
		'image_list':image_list},
	)

@login_required
def index(request):

	return render(
		request,
		'index.html',
		context={},
	)

@csrf_exempt
def post_list(request):
	return render(request, 'post_list.html', {})

@csrf_exempt
@login_required
def data_return(request):
	temp = 0
	result = list(request.POST.keys())
	if(request.method == 'POST'):
		if(result[0][:5] == "label"):
			print("fail")
			return render(request, 'test.html',{})

		label = result[0][0]
		data = result[0][28:]
	
		index = Images.objects.all().count()+1
		path = Images.objects.all().aggregate(Max('path'))['path__max']
		if path==None:
			path = 1
		else:
			path = path+1
		print(index, path)

		nimage = Images()
		nimage.owner = request.user
		nimage.id = index
		nimage.label = label
		nimage.path = path
		nimage.img = '/image/'+str(index)+'.jpg'
		nimage.save()

		img = base64.b64decode(data)

		fh = open("imageToSave.png", "wb")
		fh.write(img)
		fh.close()

		im = Image.open("imageToSave.png")
		rgb_im = im.convert('RGB')
		rgb_im.save('colors.jpg')
		rgb_im.save('image/'+str(index)+'.jpg')


		im=Image.open('colors.jpg')
		img = array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
		data = img.reshape([1, 784])
		data = 1 - (data/255)
		
		#scipy.misc.imsave('image/test.png', img)

		return render(request, 'test.html',{})

@csrf_exempt
def train(request):
	if(request.method=='POST'):
		mnist = MnistNN.objects.filter(owner=request.user)
		mnist[0].work(True, None, request.user)
		return render(request, 'test.html', {})
		'''
		tf.set_random_seed(777)  # reproducibility

		# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		# Check out https://www.tensorflow.org/get_started/mnist/beginners for
		# more information about the mnist dataset

		# hyper parameters
		learning_rate = 0.001
		training_epochs = 15
		batch_size = 100

		# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
		keep_prob = tf.placeholder(tf.float32)

		 # input place holders
		X = tf.placeholder(tf.float32, [None, 784])
		X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
		Y = tf.placeholder(tf.float32, [None, 10])
		# L1 ImgIn shape=(?, 28, 28, 1)
		W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
		#    Conv     -> (?, 28, 28, 32)
		#    Pool     -> (?, 14, 14, 32)
		L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
		L1 = tf.nn.relu(L1)
		L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1], padding='SAME')
		L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

		# L2 ImgIn shape=(?, 14, 14, 32)
		W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
		#Conv      ->(?, 14, 14, 64)
		#    Pool      ->(?, 7, 7, 64)
		L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
		L2 = tf.nn.relu(L2)
		L2=  tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1, 2, 2, 1], padding='SAME')

		 # L4 FC 4x4x128 inputs -> 625 outputs
		W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
		initializer=tf.contrib.layers.xavier_initializer())
		b4 = tf.Variable(tf.random_normal([625]))
		L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)

		# L5 Final FC 625 inputs -> 10 outputs
		W5 = tf.get_variable("W5", shape=[625, 10],
		initializer=tf.contrib.layers.xavier_initializer())
		b5 = tf.Variable(tf.random_normal([10]))
		logits = tf.matmul(L4, W5) + b5

		saver = tf.train.Saver()
		init_op = tf.global_variables_initializer()
		# with tf.Session() as sess:
		#     sess.run(init_op)
		#     save_path = "./minist_softmax.ckpt"
		#     saver.restore(sess, save_path)
		#     prediction = sess.run(
		#     tf.argmax(logits, 1), feed_dict={X: data, keep_prob: 1})
		#     temp = prediction[0]

		with tf.Session() as sess:
			sess.run(init_op)
			save_path = "./minist_softmax.ckpt"
			saver.restore(sess, save_path)
			prediction = sess.run(
			tf.nn.softmax(logits, 1), feed_dict={X: data, keep_prob: 1})
			temp = prediction[0]

		test = prediction[0]
		max2 = argmax(test)
		max_val = test[max2]*100
		starlist = list()
		max_val = int(max_val)
		print(max2)
		# print(max_val)

		# print("---------------------------------------- 100%")
		for i in range(10):
			to = int(test[i]*40)
			temp = ""
			for j in range(to):
				temp += "*"
			starlist.append(temp)
		# print(starlist[i])
		# print("---------------------------------------- 100%")

		# return render(request, 'templates/post_list.html', {})
		return render(request,'test.html',
			{'max' : max2,
			'max_val' : max_val,
			'tests0' : starlist[0],
			'tests1' : starlist[1],
			'tests2' : starlist[2],
			'tests3' : starlist[3],
			'tests4' : starlist[4],
			'tests5' : starlist[5],
			'tests6' : starlist[6],
			'tests7' : starlist[7],
			'tests8' : starlist[8],
		 	'tests9' : starlist[9],
			}
		)
		'''
