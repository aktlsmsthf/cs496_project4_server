from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

# Create your models here.
class Post(models.Model):
	author = models.ForeignKey('auth.User', on_delete=models.CASCADE)
	title = models.CharField(max_length=200)
	text = models.TextField()
	created_date = models.DateTimeField(default=timezone.now)
	published_date = models.DateTimeField(blank=True, null=True)

	def publish(self):
		self.published_data = timezone.now()
		self.save()
	
	def __str__(self):
		return self.title

class TestData(models.Model):
	pred = models.IntegerField()

	def __str__(self):
		return self.ped

class Genre(models.Model):
	name=models.CharField(max_length=200, help_text="Enter a bood genre")

	def __str__(self):
		return self.name

class Book(models.Model):
	title = models.CharField(max_length=200)
	author = models.ForeignKey('Author', on_delete=models.SET_NULL, null=True)

	summary = models.TextField(max_length = 1000, help_text="Enter a brief description of the book")
	isbn = models.CharField('ISBN', max_length=13, help_text='13 character')
	genre = models.ManyToManyField(Genre, help_text="Select a genre for this book")

	def __str__(self):
		return self.title

	def get_absolute_url(self):
		return reverse('book-detail', args=[str(self.id)])

class BookInstance(models.Model):
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, help_text="Unique ID")
	book = models.ForeignKey('Book', on_delete=models.SET_NULL, null=True)
	imprint = models.CharField(max_length=200)
	due_back = models.DateField(null=True, blank=True)
	borrower = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

	LOAN_STATUS=(
		('m', 'Maintenance'),
		('o', 'On loan'),
		('a', 'Available'),
		('r', 'Reserved'),
	)

	status = models.CharField(max_length=1, choices=LOAN_STATUS, blank = True, default = 'm', help_text = 'Book availability')

	class Meta:
		ordering = ["due_back"]

	def __str__(self):
		return '{0} ({1})'.format(self.id, self.book.title)

class Author(models.Model):
	first_name = models.CharField(max_length=100)
	last_name = models.CharField(max_length=100)
	date_of_birth = models.DateField(null=True, blank=True)
	data_of_death = models.DateField(null=True, blank=True)

	class Meta:
		ordering = ["last_name", "first_name"]

	def get_absolute_url(self):
		return reverse('author-detail', args=[str(self.id)])

	def	__str__(self):
		return '{0}, {1}'.format(self.last_name, self.first_name)
