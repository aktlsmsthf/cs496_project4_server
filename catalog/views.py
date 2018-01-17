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
	accuracy = MnistNN.objects.filter(owner=request.user).values('accuracy')[0]['accuracy']

	return render(
		request,
		'mypage.html',
		context = {'num_images':num_image, 
		'image_list':image_list,
		'accuracy': accuracy},
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
def delete(request):
	print("delete")
	result = list(request.POST.keys())
	if(request.method == 'POST'):
		deletes = result[0].split(',')
		for img in deletes:
			Images.objects.filter(img=img[6:]).delete()
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

		nimage = Images()
		nimage.owner = request.user
		nimage.id = path
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
		exist = MnistNN.objects.filter(owner=request.user).count()
		accuracy = 0
		if(exist!=0):
			mnist = MnistNN.objects.filter(owner=request.user)
			accuracy = mnist[0].work(True, None, request.user)
			
		else:
			mnist = MnistNN()
			user = request.user._wrapped if hasattr(request.user, '_wrapped') else request.user
			mnist.owner = user
			mnist.save()
			accuracy = mnist.work(True, None, user)
			mnist.accuracy =accuracy
			mnist.save()
		print(accuracy)
		return render(request, 'test.html', {'accuracy':accuracy})
		
