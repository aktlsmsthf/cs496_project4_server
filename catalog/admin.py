from django.contrib import admin
from .models import Images, MnistNN, TestImages

# Register your models here.
admin.site.register(Images)
admin.site.register(MnistNN)
admin.site.register(TestImages)

class BookInstanceAdmin(admin.ModelAdmin):
	list_display = ('book', 'status', 'borrower', 'due_back', 'id')
	list_filter = ('status', 'due_back')

	fieldsets =(
		(None, {
			'fields':('book', 'imprint', 'id')	
		}),
		('Availability',{
			'fields':('status', 'due_back', 'borrower')
		}),
	)
