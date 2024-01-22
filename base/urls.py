from django.urls import path
from django.contrib import admin
from . import views
from django.urls import re_path


urlpatterns = [

    path ('login/', views.loginPage, name = 'login'),
    path ('logout/', views.logoutUser, name = 'logout'),
    path ('register/', views.signUp, name = 'register'),
    path ("", views.home, name = 'home'),
    path ('createCourse/', views.createCourse, name = 'createCourse'),
    path ('addLecture/<str:pk>/', views.addLecture, name = 'addLecture'),
    path ('editLecture/<str:pk>/', views.editLecture, name = 'editLecture'),
    path ('updateCourse/<str:pk>/', views.updateCourse, name = 'updateCourse'),
    path ('deleteCourse/<str:pk>/', views.deleteCourse, name = 'deleteCourse'),
    path ('deleteLecture/<str:pk>/', views.deleteLecture, name = 'deleteLecture'),
    path ('chat/<int:lecture_id>/', views.chat, name = 'chat'),
    path ('chatbot/<int:lecture_id>/', views.chatbot, name = 'chatbot'),
    path ('slides_chatbot/<int:lecture_id>/', views.slides_chatbot, name = 'slides_chatbot'),
    path ('about/', views.about, name = 'about'),  
    path ('enrollStudent/<str:pk>/', views.enroll_student_view, name = 'enrollStudent'),
    path('clear_conversation/<int:lecture_id>/', views.clear_conversation, name='clear_conversation'),
    path ('viewEnrollment/<str:pk>/', views.viewEnrollment, name = 'viewEnrollment'),
    path ('coursepage/', views.coursepage, name = 'coursepage'),
    path ('lecturepage/<str:pk>/', views.lecturepage, name = 'lecturepage'),
    path ('view_study_plan/<int:pk>/', views.view_study_plan, name = 'view_study_plan'),
    path ('view_practice_quiz/<int:pk>/', views.view_practice_quiz, name = 'view_practice_quiz'),
    path ('view_slides/<int:pk>/', views.view_slides, name = 'view_slides'),
    path('update_page_number/', views.update_page_number, name='update_page_number'),
    path('my_account/', views.my_account, name='my_account'),
    path('delete_account/', views.delete_account, name='delete_account'),
    path('lecture/<int:pk>/', views.view_lecture_content, name='view_lecture_content'),
    re_path(r'^get_s3_url/(?P<file_key>.+)/$', views.get_s3_url, name='get_s3_url'),
    re_path(r'^pdf_proxy/(?P<file_name>.+)/$', views.pdf_proxy, name='pdf_proxy')
    ]