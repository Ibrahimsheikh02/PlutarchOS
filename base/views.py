from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from .models import Course, Message , Lecture, User, LectureChatbot, DiscussionMessages
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
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
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
from urllib.parse import quote
import os 
import time

DJANGO_ENV = os.environ.get('DJANGO_ENV', 'local')
openai.api_key = settings.OPENAI_API_KEY
openai_api_key = settings.OPENAI_API_KEY
openai_chat_model = "gpt-3.5-turbo"
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


#Authentication
def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST': 
        username = request.POST.get('username').lower()
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home') # or any other page you'd like to redirect to
        else:
            messages.error(request, 'Username or Password is incorrect')

    context = {"page": 'login'}
    return render(request, 'login.html', context)

@login_required(login_url='login')
def logoutUser(request):
    logout(request)
    return redirect('home')

def signUp (request): 
    form = CustomUserCreationForm()
    context = {"page": 'register', "form":form}
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid(): 
            form.save(commit=False)
            user = form.save(commit = False)
            user.username = user.username.lower()
            user.save()

            group = Group.objects.get(name='Student')
            group.user_set.add(user)

            messages.success(request, 'Account was created for ' + user.username)
            login(request, user)
            return redirect ('home')
        
        else: 
            messages.error (request, 'An error has occured during registration')

    return render (request, 'login.html', context)

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

    user = request.user.username.capitalize() 
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
            return redirect('home')
    else:
        form = CourseForm()
    context = {'form': form}
    return render(request, 'create_course.html', context)

@login_required(login_url='login')
def updateCourse (request, pk): 
    course = Course.objects.get(id=pk)
    form = CourseForm(instance=course)

    if request.user != course.created_by:
        raise Http404("You are not allowed to edit this course")
    
    if request.method == 'POST':
        form = CourseForm(request.POST, instance=course)
        if form.is_valid(): 
            form.save()
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
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key )
    VectorStore = FAISS.from_texts(chunks, embeddings)

    return pickle.dumps(VectorStore)

@login_required(login_url='login')       
def addLecture(request, pk): 
    course = get_object_or_404(Course, id=pk)
    if request.user != course.created_by:
        raise Http404("You are not allowed to add this lecture")
    form = AddLecture()
    if request.method == 'POST':
        form = AddLecture(request.POST, request.FILES)
        if form.is_valid(): 
             lecture_name = form.instance.name
             lecture = form.save(commit=False)
             lecture.course = course
             lecture.save()
             if lecture.lecture_pdf:  # Check if lecture_pdf is not None

                pdf_url = generate_presigned_url('lectureme', lecture.lecture_pdf.name)
                pdf_tmp_path = get_temp_file_from_s3(pdf_url)
                text = extract_text(pdf_tmp_path)   

                lecture.lecture_text = text
                lecture.embeddings = create_embeddings(text)
                os.remove(pdf_tmp_path)

             if lecture.lecture_transcript:  # Check if lecture_transcript is not None
                transcript_url = generate_presigned_url('lectureme', lecture.lecture_transcript.name)
                transcript_tmp_path = get_temp_file_from_s3(transcript_url)
                t_text = extract_text(transcript_tmp_path)

                lecture.transcript_text = t_text
                lecture.transcript_embeddings = create_embeddings(t_text)
                os.remove(transcript_tmp_path)

             lecture.save()
             if lecture.syllabus == False:
                #generate_quiz_debug.delay(pk=lecture.id)
                generate_study_plan_and_quiz.delay(lecture.id)
                
             return redirect ('lecturepage', pk = pk)
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

        response = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k', messages=conversation)
        last_response = response['choices'][0]['message']['content']
        all_responses.append(last_response)

    return ''.join(all_responses)

def editLecture(request, pk):
    lecture = Lecture.objects.get(id=pk)
    course = lecture.course
    form = EditLecture(instance=lecture)
    
    if request.user != course.created_by:
        raise Http404("You are not allowed to edit this lecture")
    
    if request.method == 'POST':
        form = EditLecture(request.POST, request.FILES, instance=lecture)  # Notice the request.FILES here
        if form.is_valid():
            form.save()
            return redirect('lecturepage', pk=course.id)
        
    context = {'form': form}
    return render(request, 'add_lecture.html', context)

    


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
    message_user = Message.objects.create(user=user, course=course, lecture=lecture, body=question_body, is_user=is_user_response)
    message_chatbot = Message.objects.create(user=user, course=course, lecture=lecture, body=response_body, is_user=not is_user_response)
    return JsonResponse({
        "message": response_body,
        "timestamp": timezone.localtime(message_chatbot.timestamp).strftime('%Y-%m-%d %H:%M:%S')
    })


@login_required(login_url='login')
def chatbot(request, lecture_id):
    model_name = 'gpt-3.5-turbo'

    #Toggle switch: Lecture Only
    lecture_only = request.POST.get('lectureOnly')


    if request.user.can_send_message == False:
        return HttpResponse("You have exceeded your message limit. Please reach out to the administration.")
    
    if request.method == 'POST':
        question = request.POST.get('question')
        lecture = Lecture.objects.get(id=lecture_id)

        #Building Context
        previous_messages = Message.objects.filter(lecture=lecture, user=request.user, is_deleted = False).order_by('-timestamp')[:5][::-1]
        conversation = [{"role" : "system", "content" : "You are a helpful assistant"}]
        for message in previous_messages:
            role = "user" if message.is_user else "assistant"
            conversation.append({"role": role, "content": message.body})
        conversation.append({"role": "user", "content": question})
        



        if lecture.lecture_pdf is None and lecture.lecture_transcript is None: 
            return JsonResponse({"message": "Sorry, there is no information on this topic"}, status=200)
        
        if lecture.lecture_pdf is None:
            response_pdf = "" 


        if lecture.lecture_transcript is None:
            response_transcript = "" 



    #Building context here

        previous_messages_content = "You are the 'Assistant' in this conversation.\n"

        for messages in previous_messages:
            if messages.is_user: 
                previous_messages_content = previous_messages_content + "User: " + messages.body + "\n"
            else:
                previous_messages_content = previous_messages_content + "Assistant: " + messages.body + "\n"


        question_context_slides = "You are given the previous interaction. Read it carefully as the next question may be a follow up.\
                Your previous interaction with the user: " + previous_messages_content + \
                "\nIf the question is about the lecture, your role is to answer the user's question using ONLY the lecture slides provided.\
                If an answer does not exist in the document (Lecture's Slides) provided, you MUST include 'sorry' in your response.\
                Do not make stuff up and you must ensure what you say is accurate. This lecture is about {lecture.name}. \
                Try to elaborate as much as you can. " + "\n" + "User's new question: " + question  


        question_context_transcript = "You are given the previous interaction. Read it carefully as the next question may be a follow up. \
                Your previous interaction with the user: " + previous_messages_content +\
                "\nYour role is to answer the user's question using ONLY the lecture transcript provided. \
                If an answer does not exist in the document (Lecture's Transcript) provided, you MUST include 'sorry' in your response. \
                Do not make stuff up.This lecture is about {lecture.name}. Try to elaborate as much as you can" + "\n" + \
                "User's new question: " + question     

#"gpt-3.5-turbo-16k"
        #PDF is provided
        if lecture.lecture_pdf is not None:
            if lecture.syllabus == True:
                question_context_slides = "Your previous interaction with the user: " + previous_messages_content + "\nYour role is to answer the user's question using ONLY the input provided. If an answer does not exist in the document (Class Syllabus) provided, you MUST include 'sorry' in your response. Do not make stuff up. Try to elaborate as much as you can. " + "\n" + "User's new question: " + question
            embeddings = pickle.loads(lecture.embeddings)
            docs = embeddings.similarity_search(question, k=3)
            llm = ChatOpenAI(temperature = 0, max_tokens = 300, openai_api_key = openai_api_key, model_name = model_name)
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            pdf_stream = get_pdf_from_s3('lectureme', lecture.lecture_pdf.name)
            relevant_pages = find_document_pages(pdf_stream, docs)
            with get_openai_callback() as cb:
                if lecture.syllabus == True:
                    response_pdf = "\nFrom Syllabus: \n" + chain.run (input_documents = docs, question = question_context_slides) + "\n\nRelevant pages: " + relevant_pages
                else:
                    response_pdf = "\nFrom Slides: \n" + chain.run (input_documents = docs, question = question_context_slides)+ "\n\nRelevant slides: " + relevant_pages 
                request.user.questions_asked += 1
                cost = round ( Decimal (cb.total_cost), 10 ) 
                completion_tokens = cb.total_tokens - cb.prompt_tokens
                request.user.question_asked_tokens += cb.prompt_tokens
                request.user.completion_tokens += completion_tokens
                request.user.expenditure += cost
                request.user.total_tokens += cb.total_tokens
                request.user.save()

            


        #Transcript is provided
        if lecture.lecture_transcript is not None: 
            if lecture.transcript_embeddings is not None:
                transcript_embeddings = pickle.loads(lecture.transcript_embeddings)
                transcript_docs = transcript_embeddings.similarity_search(question, k=3)
                llm = ChatOpenAI(temperature = 0, max_tokens= 300, openai_api_key = openai_api_key, model_name = model_name)
                chain = load_qa_chain(llm = llm, chain_type = "stuff")
                transcript_stream = get_pdf_from_s3('lectureme', lecture.lecture_transcript.name)
                transcript_relevant_pages = find_document_pages(transcript_stream, transcript_docs)
                with get_openai_callback() as cb:
                    response_transcript = "\n \n From Transcript: \n" + chain.run (input_documents = transcript_docs, question = question_context_transcript) + "\n\nRelevant pages: " +  transcript_relevant_pages 
                    request.user.questions_asked += 1
                    cost = round ( Decimal (cb.total_cost), 10 ) 
                    completion_tokens = cb.total_tokens - cb.prompt_tokens
                    request.user.question_asked_tokens += cb.prompt_tokens
                    request.user.completion_tokens += completion_tokens
                    request.user.expenditure += cost
                    request.user.total_tokens += cb.total_tokens
                    request.user.save()

            else: 
                response_transcript = ""
        
    





        response = response_pdf + " " + response_transcript

        #Returning if user has said they only want lecture
        if lecture_only == 'on':
            return create_and_return_message(request.user, lecture.course, lecture, question, response, True)


        
        phrases_to_check = [
    "there is no information", 
    "there is no specific information", 
    "sorry", 
    "insufficient data", 
    "no clear question", 
    "i don't know", 
    "there is not enough", 
    "there is no answer",
    "i apologize for the inconvenience", 
    "i apologize"
                            ]
        response_pdf_has_phrases = any(phrase in response_pdf.lower() for phrase in phrases_to_check)
        response_transcript_has_phrases = any(phrase in response_transcript.lower() for phrase in phrases_to_check)
        
        
        if response_pdf_has_phrases and response_transcript_has_phrases:
            chatgpt = openai.ChatCompletion.create(
                model = model_name,
                messages = conversation, 
                max_tokens = 300,
            )
            chatresponse = "This information is not in your lectures. But here goes: " +  chatgpt['choices'][0]['message']['content']
            request.user.questions_asked += 1

    

            # Fetch and save the number of tokens used:
            prompt_tokens = chatgpt['usage']['prompt_tokens']
            tokens_used = chatgpt['usage']['total_tokens']
            completion_tokens = tokens_used - prompt_tokens
            request.user.total_tokens += tokens_used
            request.user.completion_tokens += completion_tokens
            request.user.question_asked_tokens += prompt_tokens

            # Fetch and save the cost:
            costpertoken_prompt = 0.0000015
            costpertoken_completion = 0.000002
            prompt_cost = round ( Decimal (prompt_tokens * costpertoken_prompt), 10 )
            completion_cost = round ( Decimal (completion_tokens * costpertoken_completion), 10 )
            cost = prompt_cost + completion_cost
            request.user.expenditure += cost
            request.user.save()

            #Returning if answer was not in lecture so chat-gpt answered
            return create_and_return_message(request.user, lecture.course, lecture, question, chatresponse, True)
        


        #Returning if response found in lecture: 
        return create_and_return_message(request.user, lecture.course, lecture, question, response, True)

    messages = Message.objects.filter(lecture=lecture, user=request.user).order_by('-timestamp')[:50]
    context = {'lecture': lecture, 'messages': messages}
    return render(request, 'chat.html', context)


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


model = 'gpt-3.5-turbo-16k'

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

def save_text_to_pdf(text, file_name):

    if DJANGO_ENV == 'production':
    # Configuration for Heroku deployment
        config = pdfkit.configuration(wkhtmltopdf='/app/bin/wkhtmltopdf')
    if DJANGO_ENV == 'local':
        config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')

    # Convert string to PDF and save to in-memory file
    pdf_bytes = pdfkit.from_string(text, False, configuration=config)  # False means don't save to a file
    pdf_buffer = BytesIO(pdf_bytes)

    return pdf_buffer


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
    context = {'study_plan_path': study_plan_path_encoded}
    return render(request, 'studyplan_pdf.html', context)

@login_required(login_url='login')
def view_practice_quiz(request, pk):
    lecture = Lecture.objects.get(id=pk)
    practice_quiz_path = generate_presigned_url('lectureme', lecture.practice_quiz.name)
    practice_quiz_path_encoded = quote(practice_quiz_path, safe='')  # URL encode the entire S3 pre-signed URL
    context = {'practice_quiz_path': practice_quiz_path_encoded}
    return render(request, 'practicequiz_pdf.html', context)




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





