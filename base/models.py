from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from django import forms

# Create your models here.

#Any change should result in python manage.py makemigrations

class User(AbstractUser):
    expenditure = models.DecimalField(max_digits=10, decimal_places=10, default=0.0000000000)  # to keep track of how much the user has spent
    email = models.EmailField(unique=True)
    questions_asked = models.IntegerField(default=0)
    question_asked_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    def clean(self):
        super().clean()

        if self.email == "":
            raise ValidationError("Email field cannot be blank")
        
    @property
    def can_send_message(self):
        return self.expenditure < 5.0
    

class Course(models.Model):
    course_image = models.ImageField(upload_to='course_images/', null=True, blank=True)
    professor = models.CharField(max_length=200, blank=True, null=True)
    name = models.CharField(max_length=200) 
    description = models.TextField(null = True, blank = True)
    updates = models.DateField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)#autonowadd is one time
    enrolled = models.ManyToManyField(User, related_name='enrolled', blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    assistants = models.CharField(max_length=200, blank=True, null=True)
    term = models.CharField(max_length=200, null = True, blank = True)


    class Meta: 
        ordering = ['-created', '-updates']

    def __str__(self):
        return self.name



class Lecture (models.Model): 
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    name = models.CharField(max_length=200)
    description = models.CharField(max_length = 600, blank = True, null = True)
    lecture_pdf = models.FileField(upload_to='lectures/', null=True, blank=False)
    embeddings = models.BinaryField(null=True, blank=True)
    lecture_text = models.TextField(null = True, blank=True)
    syllabus = models.BooleanField(default=False)
    lecture_transcript = models.FileField(upload_to='transcripts/', null=True, blank=True)
    transcript_embeddings = models.BinaryField(null=True, blank=True)
    transcript_text = models.TextField(null = True, blank=True)
    studyplan = models.FileField(upload_to='study_plans/', null=True, blank=True)
    practice_quiz = models.FileField(upload_to='practice_quiz/', null=True, blank=True)
    visible = models.BooleanField(default = False)

    def __str__(self): 
        return self.name
    
class Message (models.Model): 
    user = models.ForeignKey('base.User', on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, null = True, blank=True)
    body = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_user = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)

    def __str__(self): 
        return self.body


class LectureChatbot (models.Model): 
    user = models.ForeignKey('base.User', on_delete=models.CASCADE)
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE)
    embeddings = ArrayField(models.FloatField(), blank=True, null=True)
    conversation_history = ArrayField(models.TextField(), blank=True, null=True)


class DiscussionMessages(models.Model): 
    user = models.ForeignKey('base.User', on_delete=models.CASCADE)
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE)
    body = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='replies')

