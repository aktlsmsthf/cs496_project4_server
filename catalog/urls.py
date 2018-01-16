from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
	path('', views.index, name='index'),

	path('mnist/', views.post_list, name="mnist"),
	path('test/',views.data_return, name="test"),
	path('mypage/', views.mypage, name="mypage"),
	path('train/',views.train, name="train"),
]
