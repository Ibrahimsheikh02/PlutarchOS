import json
import pickle
import openai
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings
from channels.layers import get_channel_layer
import asyncio
from custom_redis import get_redis_connection

openai.api_key = settings.OPENAI_API_KEY

class YourStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.lecture_id = self.scope['url_route']['kwargs']['lecture_id']
        self.group_name = f'chat_{self.lecture_id}'

        # Get Redis connection
        self.redis_conn = await get_redis_connection()

        # Join room group
        await self.redis_conn.sadd(self.group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.redis_conn.srem(self.group_name, self.channel_name)
        self.redis_conn.close()
        await self.redis_conn.wait_closed()

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        question = text_data_json['message']
        user = self.scope['user'] if self.scope['user'].is_authenticated else None
        lecture_id = self.lecture_id
        await self.chatbot(user, question, lecture_id)

    async def chat_message(self, message, is_new_message=True):
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message,
            'new_message': is_new_message
        }))

    @database_sync_to_async
    def get_previous_messages(self, lecture, user):
        from .models import Message, Lecture, Course
        return list(Message.objects.filter(lecture=lecture, user=user, is_deleted=False).order_by('-timestamp')[:5][::-1])

    async def chatbot(self, user, question, lecture_id):
        from .models import Message, Lecture, Course
        lecture = await database_sync_to_async(Lecture.objects.get)(id=lecture_id)
        course = await database_sync_to_async(Course.objects.get)(lecture=lecture)
        model_name = 'gpt-4-1106-preview'
        input_token_cost = 0.000001
        output_token_cost = 0.000002

        if not user.can_send_message:
            response = 'You have exceeded the rate limit. Please contact the administration'
            await self.chat_message(response)
            return

        # Building Context
        previous_messages = await self.get_previous_messages(lecture, user)
        conversation = [{"role": "system", "content": "You are a helpful assistant"}]
        for message in previous_messages:
            role = "user" if message.is_user else "Assistant"
            conversation.append({"role": role, "content": message.body})

        previous_messages_content = "You are the 'Assistant' in this conversation.\n"
        for message in previous_messages:
            previous_messages_content += f"User: {message.body}\n"
            previous_messages_content += f"Assistant: {message.reply}\n"

        if lecture.lecture_pdf is not None:
            embeddings = pickle.loads(lecture.embeddings)
            docs = embeddings.similarity_search(question, k=4)
            docs_text = "".join(doc.page_content for doc in docs)

        question_context_slides = f"""
    You are a course tutor for {course.name} and you are interacting with a student. Be professional and only discuss the lecture. 
    This question is about a lecture titled {lecture.name}. 
    Your job is to answer the user's question by ONLY using the lecture material provided here as documents. 
    IT IS VERY IMPORTANT to ensure that you use the documents as much as possible.
    The question may be a new question or a follow up. 
    For reference, you are provided with the user's previous interaction with you. 
    Previous interaction: {previous_messages_content}. 
    The user's new question is: {question}
    You have access to the lecture material. Here it is: 
    {docs_text}
    Again, please answer this question using the lecture material provided. IT IS VERY IMPORTANT. 
    """     
        
        conversation = [{"role": "user", "content": question_context_slides}]
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=conversation,
            max_tokens=800, 
            stream=True
        )

        full_response = ""
        is_new_message = True
        for chunk in response:
            choices = chunk.get('choices', [])
            if choices:
                first_choice = choices[0]
                stop_reason = first_choice.get('finish_reason')
                if stop_reason == 'stop':
                    break
                delta = first_choice.get('delta', {})
                content = delta.get('content')
                if content:
                    await self.chat_message(content, is_new_message)
                    full_response += content
                    is_new_message = False
                    await asyncio.sleep(0.000001)

        await database_sync_to_async(Message.objects.create)(
            user=user, 
            course=course, 
            lecture=lecture, 
            body=question, 
            reply=full_response
        )
        
        return 0
