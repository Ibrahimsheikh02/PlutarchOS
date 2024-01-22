from django.forms import ModelForm
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User
from .models import Course, Lecture
from django.contrib.auth.models import Group
from django.contrib.auth.forms import PasswordChangeForm

class CourseForm(ModelForm): 

    class Meta: 
        model = Course
        fields = '__all__'
        exclude = ['created_by', 'enrolled']

    def save(self, commit=True):
        return super().save(commit)

class AddLecture (ModelForm): 
    class Meta: 
        model = Lecture
        fields = '__all__'
        exclude =  ['lecture_text', 'transcript_text', 'course', 'studyplan', 'practice_quiz', 'slides_text']
        

class EditLecture (ModelForm): 
    class Meta: 
        model = Lecture
        fields = '__all__'
        exclude =  ['lecture_text', 'transcript_text', 'course']
        
class EnrollStudentsForm(forms.ModelForm):
    class Meta:
        model = Course  
        fields = ['enrolled']  

    def __init__(self, *args, **kwargs):
        self.course = kwargs.pop('course')  # we pass the course instance when initializing the form
        super(EnrollStudentsForm, self).__init__(*args, **kwargs)
        
        # Get all the users who are in the group 'Student'
        student_group = Group.objects.get(name='Student')
        all_students = student_group.user_set.all()

        # Exclude the students who are already enrolled in the course.
        self.fields['enrolled'].queryset = all_students.exclude(id__in=self.course.enrolled.all())

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('first_name','last_name', 'password1', 'password2', 'email')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if email == "":
            raise forms.ValidationError("Email field cannot be blank")
        return email
    


class UsernameChangeForm (forms.ModelForm):
    class Meta: 
        model = User 
        fields = ['first_name', 'last_name', 'relevant_pages']



        