from django.urls import path
from . import consumers
from django.urls import re_path

websocket_urlpatterns = [
    re_path(r'^ws/chat/(?P<lecture_id>\d+)/$', consumers.YourStreamConsumer.as_asgi()),
]
