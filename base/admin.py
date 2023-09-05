from django import forms
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.forms import ModelMultipleChoiceField, ModelForm, SelectMultiple
from django.contrib.admin.widgets import FilteredSelectMultiple
from django.contrib.auth.admin import UserAdmin


# Register your models here.

from .models import *


User = get_user_model() 
class CourseAdmin(admin.ModelAdmin): 
    list_display = ('name', 'created_by', 'created', 'updates',)
    filter_horizontal = ('enrolled',)

    
class UserAdmin(admin.ModelAdmin): 
    list_display = ('username', 'email', 'expenditure', 'questions_asked')

admin.site.register(User, UserAdmin)  
admin.site.register(Course, CourseAdmin)
admin.site.register(Message)
admin.site.register(Lecture)
admin.site.register(LectureChatbot)
admin.site.register(DiscussionMessages)
