from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
	path('', views.index, name='index'),
	path('books/', views.BookListView.as_view(), name='books'),
	path('book/<int:pk>', views.BookDetailView.as_view(), name='book-detail'),
	path('mybooks/', views.LoanedBooksByUserListView.as_view(), name='my-borrowed'),
	path('mnist/', views.post_list, name="mnist"),
	path('test/',views.data_return, name="test"),
]
