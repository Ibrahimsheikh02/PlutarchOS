from django.shortcuts import render, redirect
from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from .models import *
from.forms import CourseForm, AddLecture,  EditLecture
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import Group
from .forms import CustomUserCreationForm
from .forms import EnrollStudentsForm
from openai import ChatCompletion
import openai
from django.shortcuts import get_object_or_404
from PyPDF2 import PdfReader
from asgiref.sync import async_to_sync
from asgiref.sync import sync_to_async
from channels.layers import get_channel_layer
from django.contrib.auth import get_user
import pickle
from django_rq import enqueue
from langchain.embeddings.openai import OpenAIEmbeddings
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from channels import layers
from django.http import JsonResponse
from django.utils import timezone
from decimal import Decimal
from langchain.schema import messages_from_dict, messages_to_dict
from django.urls import reverse
from django.core.files.base import ContentFile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.db.models import F, Case, When
from django.core.files import File
from background_task import background
import PyPDF2
from reportlab.lib.utils import simpleSplit
import django_rq
import tiktoken
from reportlab.lib.pagesizes import A4
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.layout import LTTextBox, LTTextLine
from time import sleep
from reportlab.pdfbase.pdfmetrics import stringWidth
import textwrap
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from django_rq import job
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pdfkit
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rq import Queue
from io import BytesIO
from django.core.files.base import ContentFile
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from .forms import UsernameChangeForm
from django.contrib.auth.forms import PasswordChangeForm
from django.conf import settings
import boto3
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import json
from urllib.parse import quote
import os 
import requests
from django.http import FileResponse
import time 
import io
import base64
from pdf2image import convert_from_path
from django.http import StreamingHttpResponse
import re

import inspect

DJANGO_ENV = os.environ.get('DJANGO_ENV', 'local')
openai.api_key = settings.OPENAI_API_KEY
openai_api_key = settings.OPENAI_API_KEY
openai_chat_model = "gpt-4o"
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
s3 = boto3.client('s3')




# Create your views here.

#Home Page 

def home (request): 
    return render (request, 'homepage.html')

#About Page
def about (request): 

    return render (request, 'about.html')


# Authentication
def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        email = request.POST.get('email').lower()  # Changed from username to email
        password = request.POST.get('password')

        # Authenticate with email
        user = authenticate(request, username=email, password=password)  # username parameter receives the email

        if user is not None:
            login(request, user)
            return redirect('coursepage')  # or any other page you'd like to redirect to
        else:
            messages.error(request, 'Email or Password is incorrect')

    context = {"page": 'login'}
    return render(request, 'login.html', context)

@login_required(login_url='login')
def logoutUser(request):
    logout(request)
    return redirect('home')

def signUp(request): 
    form = CustomUserCreationForm()
    context = {"page": 'register', "form":form}
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid(): 
            user = form.save()
            enroll_in_courses(user)

            # Assign user to a group, if necessary
            group = Group.objects.get(name='Student')
            group.user_set.add(user)

            messages.success(request, 'Account was created for ' + user.email)  # Changed from username to email
            login(request, user)
            return redirect('home')
        
        else: 
            messages.error(request, 'An error has occurred during registration')

    return render(request, 'login.html', context)


def enroll_in_courses(user):
    all_courses = Course.objects.all()
    for course in all_courses:
        if course.roster:
            email_list = [email.strip().lower() for email in course.roster.split(',')]
            if user.email.lower() in email_list:
                course.enrolled.add(user)



#Change password, username, delete account
@login_required(login_url='login')
def my_account(request):
    username_form = None
    password_form = None

    if request.method == 'POST':
        if 'change_username' in request.POST:
            username_form = UsernameChangeForm(request.POST, instance=request.user)
            if username_form.is_valid():
                username_form.save()
                return redirect('home')
            else:
                password_form = PasswordChangeForm(request.user)
        elif 'change_password' in request.POST:
            password_form = PasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                return redirect('home')
            else:
                username_form = UsernameChangeForm(instance=request.user)
    else:
        username_form = UsernameChangeForm(instance=request.user)
        password_form = PasswordChangeForm(request.user)

    return render(request, 'my_account.html', {'username_form': username_form, 'password_form': password_form})

@login_required(login_url='login')
def delete_account (request):
    if request.method == 'POST': 
        user = request.user
        user.delete()
        messages.success(request, 'Account was deleted successfully')
        return redirect ('home')
    return render (request, 'delete_account_confirm.html')


#Courses (Coursepage)

@login_required (login_url='login')
def coursepage (request): 

    user = request.user.email.capitalize() 
    mycourses = Course.objects.all()
    is_professor = request.user.groups.filter(name='Professor').exists()
    context = {'mycourses': mycourses, 'user':user, 'is_professor': is_professor }

    return render (request,  'coursepage.html', context)

@login_required(login_url='login')
def createCourse(request): 
    if request.method == 'POST':
        form = CourseForm(request.POST, request.FILES)
        if form.is_valid(): 
            course = form.save(commit=False)
            course.created_by = request.user
            course.save()
            form.save_m2m()  
            enroll_students(course)
            return redirect('home')
    else:
        form = CourseForm()
    context = {'form': form}
    return render(request, 'create_course.html', context)


def enroll_students(course):
    if course.roster:
        email_list = course.roster.split(',')  # Split the roster into a list of emails
        for email in email_list:
            try:
                user = User.objects.get(email=email.strip())
                course.enrolled.add(user)
            except User.DoesNotExist:
                pass  # Ignore if user does not exist


@login_required(login_url='login')
def updateCourse (request, pk): 
    course = Course.objects.get(id=pk)
    form = CourseForm(instance=course)

    if request.user != course.created_by:
        raise Http404("You are not allowed to edit this course")
    
    if request.method == 'POST':
        form = CourseForm(request.POST, instance=course)
        if form.is_valid(): 
            updated_course = form.save()
            enroll_students(updated_course)
            return redirect ('home')


    context = {'form': form}

    return render (request, 'create_course.html', context)

@login_required(login_url='login')
def deleteCourse (request, pk): 
    course = Course.objects.get(id=pk)
    if request.user != course.created_by: 
        raise Http404("You are not allowed to delete this lecture")
    if request.method == 'POST':
        course.delete()
        return redirect ('home')
    context = {'course': course, 'obj':course}
    return render (request, 'delete.html', context)


#Lectures 
@login_required (login_url='login')
def lecturepage (request, pk):
    courses = Course.objects.get(id=pk)
    lectures = Lecture.objects.filter(course = courses)

    if courses.created_by == request.user or request.user.is_superuser:
        lectures = Lecture.objects.filter(course = courses)
    elif request.user in courses.enrolled.all():
        lectures = Lecture.objects.filter(course = courses)
    else: 
        raise Http404("Course not found")
    
    # Ordering the lectures
    lectures = lectures.annotate(
        custom_order=Case(
            When(syllabus=True, then=0),  # Syllabus comes first
            default=1
        )
    ).order_by('custom_order', '-date')  # Newest lectures on top

    context = {'courses': courses, 'lectures':lectures}
    return render (request, 'lecturepage.html', context)

def create_embeddings (text): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000, 
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key, 
                                  model = "text-embedding-ada-002" )
    VectorStore = FAISS.from_texts(chunks, embeddings)

    return pickle.dumps(VectorStore)

@job
def process_pdf_background(lecture_id, course_id):
    course = Course.objects.get(id=course_id)
    lecture = Lecture.objects.get(id=lecture_id)
    
    # Handle transcript if exists
    if lecture.lecture_transcript:
        transcript_url = generate_presigned_url('lectureme', lecture.lecture_transcript.name)
        transcript_tmp_path = get_temp_file_from_s3(transcript_url)
        t_text = extract_text(transcript_tmp_path)
        lecture.transcript_text = t_text
        os.remove(transcript_tmp_path)
        
    # Process PDF
    pdf_url = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
    pdf_tmp_path = get_temp_file_from_s3(pdf_url)
    
    # Get PDF text and create embeddings
    text_as_list = process_full_pdf(pdf_tmp_path, lecture.name, course)
    text = "\n".join(text_as_list)
    if lecture.transcript_text:
        text = text + lecture.transcript_text
    lecture.lecture_text = text
    lecture.embeddings = create_embeddings(text)
    
    # Save slides text
    json_dict = {str(index + 1): value for index, value in enumerate(text_as_list)}
    lecture.slides_text = json_dict
    
    # Generate and save quiz
    lecture_quiz = get_quiz_from_lecture_slides(text, lecture.name)
    lecture_quiz = str(lecture_quiz)
    pdf_quiz = save_text_to_pdf(lecture_quiz, lecture.name + "_quiz")
    lecture.practice_quiz.save(pdf_quiz.name, pdf_quiz, save=True)
    
    os.remove(pdf_tmp_path)
    lecture.save()
  


@login_required(login_url='login')       
def addLecture(request, pk): 
    course = get_object_or_404(Course, id=pk)
    if request.user != course.created_by:
        raise Http404("You are not allowed to add this lecture")
    form = AddLecture()
    if request.method == 'POST':
        form = AddLecture(request.POST, request.FILES)
        if form.is_valid():
            lecture = form.save(commit=False)
            lecture.course = course
            lecture.save()

            # Instead of processing immediately, queue the job
            if not lecture.syllabus:
                lecture_name = lecture.name
                pdf_url = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
                
                # Queue the processing job
                django_rq.enqueue(process_pdf_background, 
                                lecture.id,
                                course.id,
                                job_timeout='1h')  # Set appropriate timeout

                return redirect('lecturepage', pk=pk)

            # If there's a syllabus, just process the PDF for text extraction
            else:
                pdf_url = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
                pdf_tmp_path = get_temp_file_from_s3(pdf_url)
                text = extract_text(pdf_tmp_path)
                lecture.lecture_text = text
                lecture.embeddings = create_embeddings(text)
                os.remove(pdf_tmp_path)
                lecture.save()

            return redirect('lecturepage', pk=pk)
    context = {'form': form}
    return render(request, 'add_lecture.html', context)

@login_required(login_url='login')
def deleteLecture (request, pk): 
    lecture = Lecture.objects.get(id=pk)
    course = lecture.course 
    if request.user != course.created_by:
        raise Http404("You are not allowed to delete this lecture")
    if request.method == 'POST':
        lecture.delete()
        return redirect ('lecturepage', pk = lecture.course.id)
    context = {'lecture': lecture, 'obj':lecture}
    return render (request, 'delete.html', context)

def cleanTranscript (transcript_chunks, lecture_name): 
    last_response = ""
    all_responses = []

    for chunk in transcript_chunks:
        conversation = [
            {"role": "user", "content": f'You are given a lecture transcript for {lecture_name}. Your task is to clean it, remove noise, check for spelling errors, and ensure you do not remove anything important. "{chunk}"'},
        ]

        if last_response:
            conversation.append({"role": "user", "content": f'Your last cleaned chunk for the same transcript was: {last_response}'})
            sleep (70)

        response = openai.ChatCompletion.create(model='gpt-4o', messages=conversation)
        last_response = response['choices'][0]['message']['content']
        all_responses.append(last_response)

    return ''.join(all_responses)

def editLecture(request, pk):
    lecture = Lecture.objects.get(id=pk)
    course = lecture.course

    if request.user != course.created_by:
        raise Http404("You are not allowed to edit this lecture")

    form = EditLecture(instance=lecture)
    
    if request.method == 'POST':
        form = EditLecture(request.POST, request.FILES, instance=lecture)
        if form.is_valid():
            # Check if lecture text is changed
            if 'lecture_text' in form.changed_data:
                # Save the form first to update the lecture object
                updated_lecture = form.save()
                # Generate new embeddings
                updated_lecture.embeddings = create_embeddings(updated_lecture.lecture_text)
                updated_lecture.save()

            else:
                form.save()

            return redirect('lecturepage', pk=course.id)
        
    context = {'form': form}
    return render(request, 'add_lecture.html', context)  # Assuming you have a separate template for editing


#GPT 4 VISION READING TEXT
def encode_image(image, max_size_mb=20, quality=85):
    # Convert to JPEG for consistency
    image_buffer = io.BytesIO()
    image.save(image_buffer, format='JPEG', quality=quality)
    image_buffer.seek(0)
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    # Check size
    if len(encoded_image) / 1e6 > max_size_mb:
        if quality > 10:  # Prevent quality from going too low
            return encode_image(image, max_size_mb, quality - 10)
        else:
            raise ValueError("Unable to compress image below 20 MB")

    return encoded_image

# def process_pdf(pdf_path, start_page, end_page, lecture_name, course):
#     images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page)
#     messages = [{
#         "role": "user",
#         "content": [{
#             "type": "text",
#             "text": 
# f"""
# You are a visual PDF reader. You are given parts of a \
# lecture titled {lecture_name} in a course titled {course.name}. \
# Your job is to read the slide by returning everything RELEVANT written on the slide and \
# explaining the illustration if there is one on the side. 
# Try to be very detailed as students will use this to understand the lecture. \
# Separate each description by using the format (THIS IS VERY IMPORTANT):
# Slide #{start_page}: your response 
# \n\n
# Slide #{start_page+1}: your response
# \n\n
# etc.
# """
#         }]
#     }]

#     for page_num, image in enumerate(images, start=start_page):
#         base64_image = encode_image(image)
#         image_message = {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{base64_image}",
#                 "detail": "high"
#             }
#         }
#         messages[0]["content"].append(image_message)

#     response = openai.ChatCompletion.create(
#         model="gpt-4-vision-preview",
#         messages=messages,
#         max_tokens=4000
#     )
#     return response.choices[0].message.content

# def get_total_pages(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         return len (reader.pages)
    
# def process_full_pdf(pdf_path, lecture_name, course):
#     total_pages = get_total_pages (pdf_path)
#     responses = ""
#     for start_page in range(1, total_pages + 1, 5):
#         end_page = min(start_page + 4, total_pages)
#         response = process_pdf(pdf_path, start_page, end_page, lecture_name, course)
#         responses = responses +'\n\n' + response
#     return responses


def process_pdf(pdf_path, page_number, lecture_name, course):
    # Note: Now processing a single page at a time
    image = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)[0]
    base64_image = encode_image(image)
    
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": 
f"""
You are a visual PDF reader. You are given part of a \
lecture titled {lecture_name} in a course titled {course.name}. \
Your job is to read the slide by returning everything RELEVANT written on the slide and \
explaining the illustration if there is one on the side. 
Try to be very detailed as students will use this to understand the lecture. \
"""
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "auto"
            }
        }]
    }]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4000
    )
    return response.choices[0].message.content

def get_total_pages(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return len(reader.pages)
    

@job
def process_full_pdf(pdf_path, lecture_name, course):
    total_pages = get_total_pages(pdf_path)
    responses = []
    for page_number in range(1, total_pages + 1):
        print (f"PAGE NUMBER: {page_number}")
        response = process_pdf(pdf_path, page_number, lecture_name, course)
        responses.append (response)
    return responses


#AWS
def get_temp_file_from_s3(url):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name='us-east-2')
    bucket_name = 'lectureme'
    
    # Update this split to handle the URL format you provided
    key_parts = url.split('lectureme.s3.amazonaws.com/')
    if len(key_parts) != 2:
        raise ValueError(f"Unexpected URL format: {url}")
    
    # We need to split off the query parameters from the key
    key = key_parts[1].split('?')[0]
    
    tmp_file_name = os.path.join('/tmp', key.split('/')[-1])
    s3.download_file(bucket_name, key, tmp_file_name)
    
    return tmp_file_name


#lecture page (with tabs)
@login_required (login_url = 'login')
def view_lecture_content (request, pk):
    lecture = Lecture.objects.get (id = pk)
    messages = Message.objects.filter(lecture=lecture, user=request.user, is_deleted = False).order_by('-timestamp')[:50][::-1]
    slide_messages = Slide_Messages.objects.filter(lecture=lecture, user=request.user, is_deleted = False).order_by('-timestamp')[:50][::-1]
    study_plan_path = generate_presigned_url('lectureme', lecture.studyplan.name)
    study_plan_path_encoded = quote(study_plan_path, safe='')  # URL encode the entire S3 pre-signed URL
    practice_quiz_path = generate_presigned_url('lectureme', lecture.practice_quiz.name)
    practice_quiz_path_encoded = quote(practice_quiz_path, safe='')  # URL encode the entire S3 pre-signed URL
    slides_path = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
    slides_path_encoded = quote(slides_path, safe='')  # URL encode the entire S3 pre-signed URL



    context = {
        'lecture': lecture, 
        'messages': messages, 
        'study_plan': study_plan_path_encoded, 
        'practice_quiz': practice_quiz_path_encoded, 
        'slides': slides_path_encoded, 
        'slide_messages': slide_messages
    } 
    return render (request, 'lecture.html', context)


#Chat with LLM


@login_required(login_url='login')
def chat(request, lecture_id):
    lecture = get_object_or_404(Lecture, pk=lecture_id)
    messages = Message.objects.filter(lecture=lecture, user=request.user, is_deleted = False).order_by('-timestamp')[:50][::-1]
    context = {
        'lecture': lecture,
        'messages': messages,
    }
    return render(request, 'chat.html', context)


def create_and_return_message(user, course, lecture, question_body, response_body, is_user_response=True):
    Message.objects.create(user=user, course=course, lecture=lecture, body=question_body, is_user=is_user_response, reply = response_body)


@login_required(login_url='login')
def slides_chatbot (request, lecture_id): 
    model = "gpt-4o"
    if request.method == 'POST':
        lecture = Lecture.objects.get (id = lecture_id)
        slides_messages = Slide_Messages.objects.filter (lecture = lecture, user = request.user, is_deleted = False).order_by('-timestamp')[:5][::-1]
        course = Course.objects.get (lecture = lecture)
        message_context = [{"role" : "system", "content" : "You are a course tutor"}]
        question = request.POST.get('question')
        page_num = request.POST.get('page_number')
        page_num = int (page_num)
        for message in slides_messages:
            message_context.append([{'role':'user', 'content': message.body}])
            message_context.append([{'role':'system', 'content': message.reply}])
        message_context.append ([{'role': 'user', 'content': question}])
        slides_desciptions =lecture.slides_text
        slide_in_question = slides_desciptions.get(str(page_num), None)
        slide_before_description = slides_desciptions.get(str(page_num - 1), None)
        slide_after_description = slides_desciptions.get(str(page_num + 1), None)

        message_to_send = f"""
You are a course tutor for {course.name}. Students are chatting with you.
Your job is to clarify lecture slides for a lecture titled {lecture.name}. 
For context, you are provided with the user's previous interaction. 
The question that follows can be a follow up or a new question. 
Your job is to use the description of the slides provided to answer the question.
The question relates to slide number {page_num}.
For more context and if possible, you will be provided with details of the slide before 
and after the slide in question. 
If the user's question 
cannot be answered through the slide descriptions provided, please say 
"This question cannot be answered." Do not make stuff up and do not go beyond the content provided. 
Your previous interaction with the user (you are the system in this conversation) {str(message_context)}.
The description of the slide before: {slide_before_description}.
The description of the slide in question: {slide_in_question}.
The description of the slide after: {slide_after_description}.
The user's new question: {question}. 
Try to give short responses unless otherwise specified by the user.
"""
        conversation = [{"role": "user", "content": message_to_send}]
        response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=conversation,
        max_tokens=300
    )
        response = response['choices'][0]['message']['content']


        Slide_Messages.objects.create(user=request.user, course=course, lecture=lecture, body=question, reply = response)
        request.user.questions_asked += 1
        request.user.save() 
        return JsonResponse ({'messages': response})



@login_required(login_url='login')
def chatbot_second(request, lecture_id):
    model_name = 'gpt-4-1106-preview'
    input_token_cost = 0.000001
    output_token_cost= 0.000002

    if request.user.can_send_message == False:
        response = 'You have exceeded the rate limit. Please contact the administration'
        create_and_return_message(request.user, lecture.course, lecture, question, response, True)
        messages = Message.objects.filter(lecture=lecture, user=request.user).order_by('-timestamp')[:50]
        context = {'lecture': lecture, 'messages': messages}
        return render(request, 'chat.html', context)




    if request.method == 'POST':
        question = request.POST.get('question')
        lecture = Lecture.objects.get(id=lecture_id)
        course = Course.objects.get (lecture = lecture)

        #Building Context
        previous_messages = Message.objects.filter(lecture=lecture, user=request.user, is_deleted = False).order_by('-timestamp')[:5][::-1]
        conversation = [{"role" : "system", "content" : "You are a helpful assistant"}]
        for message in previous_messages:
            role = "user" if message.is_user else "Assistant"
            conversation.append({"role": role, "content": message.body})

        previous_messages_content = "You are the 'Assistant' in this conversation.\n"

        for messages in previous_messages:
            previous_messages_content = previous_messages_content + "User: " + messages.body + "\n"
            previous_messages_content = previous_messages_content + "Assistant: " + messages.reply + "\n"

        if lecture.lecture_pdf is not None:
            embeddings = pickle.loads(lecture.embeddings)
            docs = embeddings.similarity_search(question, k=6)

        question_context_slides = f"""
You are a course tutor for {course.name} and you are interacting with a student. \
This question is about a lecture titled {lecture.name}. \
Your job is to answer the user's question by ONLY using the lecture material provided here as documents. \
IT IS VERY IMPORTANT to ensure that you use the documents as much as possible.
The question may be a new question or a follow up. \
For reference, you are provided with the user's previous interaction with you. \
Previous interaction: {previous_messages_content}. \
The user's new question is: {question}\
Again, please answer this question using the lecture material provided. IT IS VERY IMPORTANT. 
"""     
        # llm = ChatOpenAI(temperature = 0, max_tokens = 500, openai_api_key = openai_api_key, 
        #                  model_name = model_name, streaming= True)
        # chain = load_qa_chain(llm = llm, chain_type = "stuff")
        # response = chain.run (input_documents = docs, question = question_context_slides)

        def openai_response_generator():
            llm = ChatOpenAI(temperature=0, max_tokens=500, openai_api_key=openai_api_key, 
                            model_name=model_name, streaming=True)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=question_context_slides)
            Message.objects.create(user = request.user, course = course, lecture=lecture, body=question, reply = response)
            messages = JsonResponse( { 'messages': response} )
            yield f"data: {response}\n\n"

        
        with get_openai_callback() as cb:
            cost = round ( Decimal (cb.total_cost), 10 ) 
            completion_tokens = cb.total_tokens - cb.prompt_tokens
            request.user.question_asked_tokens += cb.prompt_tokens
            request.user.completion_tokens += completion_tokens
            request.user.expenditure += cost
            request.user.total_tokens += cb.total_tokens
            request.user.save()
        request.user.questions_asked += 1
        request.user.save() 
        # Message.objects.create(user = request.user, course = course, lecture=lecture, body=question, reply = response)
        # messages = JsonResponse( { 'messages': response} )
        
        #return messages

        return StreamingHttpResponse(openai_response_generator(), content_type="text/event-stream")


def get_previous_messages(lecture, user):
    return list(Message.objects.filter(lecture=lecture, user=user, is_deleted=False).order_by('-timestamp')[:5][::-1])

async def chatbot (request, lecture_id):

    
    channel_layer = get_channel_layer()
    group_name = f'chat_{lecture_id}'
    model_name = 'gpt-4o'
    input_token_cost = 0.000001
    output_token_cost= 0.000002
    user = await sync_to_async(get_user)(request)
    

    if user.can_send_message == False:
        response = 'You have exceeded the rate limit. Please contact the administration'
        create_and_return_message(request.user, lecture.course, lecture, question, response, True)
        messages = Message.objects.filter(lecture=lecture, user=request.user).order_by('-timestamp')[:50]
        context = {'lecture': lecture, 'messages': messages}
        return render(request, 'chat.html', context)




    if request.method == 'POST':
        question = request.POST.get('question')
        lecture = await sync_to_async(Lecture.objects.get)(id=lecture_id)
        course = await sync_to_async(Course.objects.get) (lecture = lecture)

        

        #Building Context
        previous_messages = await sync_to_async(get_previous_messages)(lecture, user)
        conversation = [{"role" : "system", "content" : "You are a helpful assistant"}]
        for message in previous_messages:
            role = "user" if message.is_user else "Assistant"
            conversation.append({"role": role, "content": message.body})

        previous_messages_content = "You are the 'Assistant' in this conversation.\n"

        for messages in previous_messages:
            previous_messages_content = previous_messages_content + "User: " + messages.body + "\n"
            previous_messages_content = previous_messages_content + "Assistant: " + messages.reply + "\n"

        if lecture.lecture_pdf is not None:
            embeddings = pickle.loads(lecture.embeddings)
            docs = embeddings.similarity_search(question, k=4) #Updated
            docs_text = ""
            for doc in docs:
                docs_text += doc.page_content



        question_context_slides = f"""
You are a course tutor for {course.name} and you are interacting with a student. 
This question is about a lecture titled {lecture.name}. 
Your job is to answer the user's question by ONLY using the lecture material provided here as documents. 
IT IS VERY IMPORTANT to ensure that you use the documents as much as possible.
The question may be a new question or a follow up. 
For reference, you are provided with the user's previous interaction with you. 
Previous interaction: {previous_messages_content}. 
The user's new question is: {question}
The relevant lecture material is: {docs_text}
Again, please answer this question using the lecture material provided. IT IS VERY IMPORTANT. 
"""     
        
        conversation = [{"role": "user", "content": question_context_slides}]
        response = openai.ChatCompletion.create (
        model=model_name,
        messages=conversation,
        max_tokens=800, 
        stream = True
        )

        full_response = ""
        
        for chunk in response:
            choices = chunk.get('choices', [])
            if choices:
                first_choice = choices[0]
                stop_reason = first_choice['finish_reason']
                if isinstance(stop_reason, str) and stop_reason == 'stop':
                    break
                delta = first_choice.get('delta', {})
                content = delta.get('content')
                if content:
                    # Send message to group
                    await channel_layer.group_send(
                        group_name,
                        {
                            'type': 'chat_message',
                            'message': content
                        }
                    )

                    full_response += content

        await sync_to_async(Message.objects.create)(
        user=request.user, 
        course=course, 
        lecture=lecture, 
        body=question, 
        reply=full_response
    )
        
        return JsonResponse({'status': 'success', 'message': 'Message Complete'})





@login_required(login_url='login')
def clear_conversation(request, lecture_id):
    user_messages = Message.objects.filter(user=request.user, lecture = lecture_id)

    for message in user_messages:
        message.is_deleted = True
        message.save()

    return redirect('chat', lecture_id=lecture_id)


def get_pdf_from_s3(bucket_name, file_key):
    file_key = 'media/' + file_key
    s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)
    s3_file_content = s3_object['Body'].read()
    return BytesIO(s3_file_content)
#Finding relevant pages

def find_document_pages(pdf_stream, documents):
    # Step 1: Extract pages from the PDF
    pdf_pages = []
    for page_layout in extract_pages(pdf_stream):
        page_text = ''
        for element in page_layout:
            if isinstance(element, (LTTextBox, LTTextLine)):
                page_text += element.get_text()
        pdf_pages.append(page_text)

    vectorizer = TfidfVectorizer().fit([doc.page_content for doc in documents] + pdf_pages)
    
    page_results = []

    # Step 2: Compare each document to each page
    for doc in documents:
        doc_vector = vectorizer.transform([doc.page_content])
        pages_found = []

        for i, page_text in enumerate(pdf_pages):
            page_vector = vectorizer.transform([page_text])
            similarity = cosine_similarity(doc_vector, page_vector)

            # Step 3: Check if similarity is above the threshold
            if similarity[0][0] >= 0.5:
                pages_found.append(str(i + 1))  # +1 because pages typically start from 1, not 0.

        page_results.append(','.join(pages_found))

    page_results = get_first_two_numbers(page_results)

    return page_results

def get_first_two_numbers(data_list):
    result = []

    for sublist in data_list:
        nums = sorted([int(num) for num in sublist.split(',') if num.strip()])
        result.extend(nums[:2])

    # Remove duplicates and sort
    unique_sorted_result = sorted(set(result))

    # Convert to string
    return ', '.join(map(str, unique_sorted_result))


#Create a study plan (background task)


model = 'gpt-4o'

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text



def get_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))
    return tokens

def chunk_text(text, max_tokens=15500):
    if get_tokens(text) <= max_tokens:
        return [text]
    words = text.split()
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for word in words:
        word_tokens = get_tokens(word)
        if current_tokens + word_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = ""
            current_tokens = 0
        current_chunk += word + " "
        current_tokens += word_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def iterative_api_calls(chunks, lecture_name, model=model, ):
    last_response = ""
    all_responses = []

    for chunk in chunks:
        conversation = [
            {"role": "user", "content": f'You are given lecture content for {lecture_name} (transcript or slides). Your task is to provide a study plan that covers the important topics in this chunk. Be thorough, precise, and accurate. Do not repeat yourself. "{chunk}"'},
        ]

        if last_response:
            conversation.append({"role": "user", "content": f'Your last study plan was: {last_response}'})
            sleep (70)

        response = openai.ChatCompletion.create(model=model, messages=conversation)
        last_response = response['choices'][0]['message']['content']
        all_responses.append(last_response)



        

    return " ".join(all_responses)

def get_final_plan (studyplan, lecture_name): 
    conversation = [ 
        { 
            "role": "user", "content": f'These study plans were generated by you.\
                Review and concatenate it by dividing it in sub-divisions and make sure there are no errors and format\
                it properly with bullet points. Give it the title "Study Plan for {lecture_name}" and place it in the center \
                end it with "This study plan was created using the lecture slides, transcript, and artificial intelligence. \
                There is a possibility of errors. Please review it and make sure it is accurate."\
                Make sure there is NO OTHER TEXT other than the study \
                plan and avoid repititions in the bullet points but make sure everything is covered. \
                Provide html formatted text.\
                  "{studyplan}"'
        }
    ]
    response = openai.ChatCompletion.create(model='gpt-4', messages=conversation)
    final_plan = response['choices'][0]['message']['content']
    return final_plan

def save_text_to_pdf(text, filename):

    if DJANGO_ENV == 'production':
    # Configuration for Heroku deployment
        config = pdfkit.configuration(wkhtmltopdf='/app/bin/wkhtmltopdf')
    if DJANGO_ENV == 'local':
        config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')

    # Convert string to PDF and save to in-memory file
    pdf_bytes = pdfkit.from_string(text, False, configuration=config)  # False means don't save to a file
    pdf_content = ContentFile(pdf_bytes)
    pdf_content.name = filename  # Assign a filename

    return pdf_content


#Quiz
#Create a practice quiz 

def quiz_api_call (chunks, lecture_name): 
    last_response = ''
    all_responses = [] 
    for chunk in chunks:
        conversation = [{'role': 'user', 'content': f'You are given lecture material for {lecture_name} which is either lecture slides or lecture transcript. \
                    Your task is to create a practice six questions multiple choice quiz with 4 options of which 1 is correct for each question for students\
                          based on the material provided. If you are unable to create at least 3 questions, please respond with "I cannot complete the task". The lecture material is:\n {chunk}'}]
        if last_response:
            conversation.append({"role": "user", "content": f'Your last practice quiz was: {last_response}'})
            sleep(70)
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k', messages=conversation)
        last_response = response['choices'][0]['message']['content']
        all_responses.append(last_response)
    
    final_response = " ".join(all_responses)
    return  final_response

def create_quiz_gpt4 (chunks, lecture_name):
    last_response = ''
    all_responses = [] 
    for chunk in chunks:
        conversation = [{'role': 'user', 'content': f'You are given part of the lecture transcript for {lecture_name}. \
                    Your task is to create a practice six questions multiple choice quiz with 4 options of which 1 is correct for each question for students\
                    based on the material provided. If this is a successive call, then your previous quiz will be provided after the transcript. The lecture material is:\n {chunk}'}]
        if last_response:
            conversation.append({"role": "user", "content": f'Your last practice quiz was: {last_response}'})
            sleep (70)
        response = openai.ChatCompletion.create(model='gpt-4', messages=conversation)
        last_response = response['choices'][0]['message']['content']
        all_responses.append(last_response)
        
    final_response = " ".join(all_responses)
    return final_response


def get_final_quiz (combined_quiz, lecture_name):
    conversation = [
    {'role': 'user', 'content': f'You were given the task to create a quiz using lecture slides for {lecture_name} and \
                    a quiz using lecture transcript for {lecture_name}. Your job now is to concatenate the quizzes and return A QUIZ with as many \
                    questions as you can and to ensure \
                    there is no repetition. Also try to ensure the quiz covers a range of topics. Format the quiz so that there is a question,\
                    then its four options (each its own line) then the next question and so on. The questions and answers should be left-formatted.\
                    The title should be Practice Quiz for {lecture_name} in the center and on a new page below it, list all the answers in order.\
                    Give the quiz in html formatted text. Be careful and do not make mistakes. At the end, write: "This quiz was generated using \
                    artificial intelligence, lecture transcript, and slides. It is prone to errors so please be careful and review it carefully.  \
                    Do not write anything else\
                    Quiz:\n {combined_quiz}'}
    ]
    response = openai.ChatCompletion.create(model='gpt-4', messages=conversation)
    quiz = response['choices'][0]['message']['content']
    return quiz


@job
def get_quiz_from_lecture_slides (slides_text, lecture_name):
    conversation = [{'role': 'user', 'content': f"""
                     
                     
You are given the description of the lecture slides for {lecture_name}. \
Your task is to create a 10 questions multiple choice quiz \
with 4 options of where 1 is correct for each question \
based on the material provided. The difficulty should be high as these are college students. \
Return an html format quiz which can be automatically converted to a PDF. 
The title should be Quiz for {lecture_name}
Please format the questions like this where the answers are at the end of the document.
1) What's 2 + 2? 
a) 3
b) 4
c) 5
d) 6
2) What's 3 + 3 
a) 1
b) 3
c) 6
d) 10

Correct Responses
1) b
2) c

etc. 
 The lecture material is:\n {slides_text}
"""}]
    response = openai.ChatCompletion.create(model='gpt-4-1106-preview', messages=conversation)
    response = response['choices'][0]['message']['content']

    return response






#This function creates both: Practice Quiz and Study Plan
@job
def generate_study_plan_and_quiz (pk):
    lecture = Lecture.objects.get(id=pk)

    slides_url = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
    slides_tmp_path = get_temp_file_from_s3(slides_url)
    slides = extract_text_from_pdf(slides_tmp_path)
    os.remove(slides_tmp_path)

    transcript_url = generate_presigned_url('lectureme', lecture.lecture_transcript.name)
    transcript_tmp_path = get_temp_file_from_s3(transcript_url)
    transcript = extract_text_from_pdf(transcript_tmp_path)
    os.remove(transcript_tmp_path)

    slides_chunks = chunk_text(slides)
    transcript_chunks_for_quiz = chunk_text(transcript, max_tokens=7500)
    transcript_chunks = chunk_text(transcript)

    #Used GPT 4 for transcript because it was more accurate
    slides_quiz = quiz_api_call(slides_chunks, lecture.name)
    transcript_quiz = create_quiz_gpt4(transcript_chunks_for_quiz, lecture.name)

    slides_study_plan = iterative_api_calls(slides_chunks, lecture.name)
    transcript_study_plan = iterative_api_calls(transcript_chunks, lecture.name)

    combined_quiz = "Quiz from slides: \n" + slides_quiz + "\n Quiz from transcript: \n" + transcript_quiz
    combined_study_plan = "Study plan from slides \n" + slides_study_plan + "\n Study plan from transcript" + transcript_study_plan
    sleep(70)
    final_study_plan = get_final_plan(combined_study_plan, lecture.name)
    sleep (70)
    the_quiz = get_final_quiz (combined_quiz, lecture.name)


    study_plan = save_text_to_pdf(final_study_plan, f"study_plan_{pk}.pdf")
    quiz = save_text_to_pdf(the_quiz, f"practice_quiz_{pk}.pdf")

    lecture.studyplan.save(f"study_plan_{pk}.pdf", ContentFile(study_plan.getvalue()))
    lecture.practice_quiz.save(f"practice_quiz_{pk}.pdf", ContentFile(quiz.getvalue()))

    lecture.save()


#AWS url security 


def generate_presigned_url(bucket_name, object_name, expiration=3600):
    object_name = 'media/' + object_name
    s3_client = boto3.client('s3', region_name = 'us-east-2')
    response = s3_client.generate_presigned_url('get_object',
                                                Params={'Bucket': bucket_name,
                                                        'Key': object_name},
                                                ExpiresIn=expiration)
    return response


@login_required
def get_s3_url(request, file_key):
    if request.method == "GET":
        url = generate_presigned_url('lectureme', file_key)
        return JsonResponse({'url': url})
    return JsonResponse({'error': 'Invalid Method'}, status=400)

#View PDF in pdf.js

@login_required(login_url='login')
def view_study_plan(request, pk):
    lecture = Lecture.objects.get(id=pk)
    study_plan_path = generate_presigned_url('lectureme', lecture.studyplan.name)
    study_plan_path_encoded = quote(study_plan_path, safe='')  # URL encode the entire S3 pre-signed URL
    context = {'study_plan_path': study_plan_path_encoded, 'lecture': lecture}
    return render(request, 'studyplan_pdf.html', context)

@login_required(login_url='login')
def view_practice_quiz(request, pk):
    lecture = Lecture.objects.get(id=pk)
    practice_quiz_path = generate_presigned_url('lectureme', lecture.practice_quiz.name)
    practice_quiz_path_encoded = quote(practice_quiz_path, safe='')  # URL encode the entire S3 pre-signed URL
    context = {'practice_quiz_path': practice_quiz_path_encoded, 'lecture': lecture}
    return render(request, 'practicequiz_pdf.html', context)

@login_required(login_url='login')
def view_slides(request, pk):
    lecture = Lecture.objects.get(id=pk)
    slides_path = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
    slides_path_encoded = quote(slides_path, safe='')  # URL encode the entire S3 pre-signed URL
    context = {'slides_path': slides_path_encoded, 'lecture': lecture}
    return render(request, 'slides_pdf.html', context)


@require_POST
def update_page_number(request):
    data = json.loads(request.body)
    page_number = data.get('page')
    # Process the page number as needed

    return JsonResponse({'status': 'success'})


@login_required(login_url='login')
def pdf_proxy(request, file_name):

    file_name = file_name
    # Generate presigned S3 URL
    pdf_url = generate_presigned_url('lectureme', file_name)


    # Fetch the PDF from S3 using the pre-signed URL
    response = requests.get(pdf_url, stream=True)

    # Stream the PDF file
    return FileResponse(response.iter_content(chunk_size=8192), content_type='application/pdf')


#Enrollment
@login_required (login_url='login')
def enroll_student_view(request, pk):
    course = Course.objects.get(pk=pk)
    if request.user != course.created_by: 
        raise Http404("You are not allowed to enroll students")
    if request.method == 'POST':
        form = EnrollStudentsForm(request.POST, course=course)
        if form.is_valid():
            students = form.cleaned_data['enrolled']
            course.enrolled.add(*students)
            messages.success(request, f'Successfully enrolled {len(students)} student(s).')
            return redirect('lecturepage', pk = pk)
        else: 
            messages.error(request, 'There was an error processing your request.')
    else:
        form = EnrollStudentsForm(course=course)
    return render(request, 'enroll.html', {'form': form})

@login_required (login_url='login')
def viewEnrollment (request, pk): 
    course = Course.objects.get(id=pk)
    if request.user != course.created_by:
        raise Http404("You are not allowed to delete this lecture")
    enrolled = course.enrolled.all()
    context = {'course': course, 'enrolled': enrolled}
    return render (request, 'enrollment.html', context)

