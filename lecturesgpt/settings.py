"""
Django settings for lecturesgpt project.

Generated by 'django-admin startproject' using Django 4.2.1.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import os
from decouple import Csv, config, Csv
import dj_database_url
import dotenv
from urllib.parse import urlparse
import redis
import ssl

DJANGO_ENV = os.environ.get('DJANGO_ENV', 'local')  # 'local' will be default if not specified
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
# Load .env file for local development
env_file = os.path.join(BASE_DIR, '.env')
if os.path.exists(env_file):
    dotenv.load_dotenv(env_file)



AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = 'lectureme'
AWS_S3_CUSTOM_DOMAIN = '%s.s3.amazonaws.com' % AWS_STORAGE_BUCKET_NAME
AWS_S3_OBJECT_PARAMETERS = {
    'CacheControl': 'max-age=86400',
}
AWS_LOCATION = 'media'
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
MEDIA_URL = 'https://%s/%s/' % (AWS_S3_CUSTOM_DOMAIN, AWS_LOCATION)
AWS_DEFAULT_ACL = None
MEDIA_ROOT = os.path.join(BASE_DIR, 'media') 
SECRET_KEY = config('DJANGO_SECRET_KEY')
OPENAI_API_KEY = config('OPENAI_API_KEY')

AUTH_USER_MODEL = 'base.User'

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

STATIC_URL = '/static/'
STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'),]
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

ROOT_URLCONF = "lecturesgpt.urls"

# SECURITY WARNING: don't run with debug turned on in production!

if DJANGO_ENV == 'local':
    DEBUG = True
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql_psycopg2',
            'NAME': 'mydatabase',
            'USER': 'Ibrahim_Work',
            'PASSWORD': 'bismillah786',
            'HOST': 'localhost',
            'PORT': '5432',
        }
    }
    MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    ]
    INSTALLED_APPS = [
    "channels",
    "base.apps.BaseConfig",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_rq",
    ]
    ALLOWED_HOSTS = ['127.0.0.1','127.0.0.1:8000', 'localhost']
    RQ_QUEUES = {
        'default': {
            'HOST': 'localhost',
            'PORT': 6379,
            'DB': 0,
            'DEFAULT_TIMEOUT': 1800,
        },
    }
    CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
            "capacity": 1500,
        },
    },
}


# Application definition

if DJANGO_ENV == 'production':
    INSTALLED_APPS = [
        "base.apps.BaseConfig",
        "django.contrib.admin",
        "django.contrib.auth",
        "django.contrib.contenttypes",
        "django.contrib.sessions",
        "django.contrib.messages",
        "django.contrib.staticfiles",
        "django_rq",
        "storages",
        "channels"
        
    ]
    CORS_ALLOWED_ORIGINS = [
    "https://plutarch.us",
    "https://lectureme-df4d6f65ea89.herokuapp.com",
    # Add more allowed domains as needed
]
    


    redis_url = os.environ.get('REDIS_URL')

    # Use 'rediss://' for a secure connection
    if redis_url.startswith("redis://"):
        redis_url = redis_url.replace("redis://", "rediss://")

    redis_url = urlparse(redis_url)



    CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [{
                'address': f"rediss://{redis_url.hostname}:{redis_url.port}",
                'password': redis_url.password,
                'ssl_cert_reqs': None,
            }],
        },
    },
}
    
    print (CHANNEL_LAYERS['default']['CONFIG']['hosts'])

# RQ Queues for background tasks
    RQ_QUEUES = {
    'default': {
        'HOST': redis_url.hostname,
        'PORT': redis_url.port,
        'DB': 0,
        'PASSWORD': redis_url.password,
        'SSL': True,  # Enable SSL
        'SSL_CERT_REQS': None,  # Disable SSL verification
        'DEFAULT_TIMEOUT': 1200,
    },
}


    MIDDLEWARE = [
        "django.middleware.security.SecurityMiddleware",
        "django.contrib.sessions.middleware.SessionMiddleware",
        "django.middleware.common.CommonMiddleware",
        "django.middleware.csrf.CsrfViewMiddleware",
        "django.contrib.auth.middleware.AuthenticationMiddleware",
        "django.contrib.messages.middleware.MessageMiddleware",
        "django.middleware.clickjacking.XFrameOptionsMiddleware",
        "whitenoise.middleware.WhiteNoiseMiddleware",
    ]
    DATABASES = {
    'default': 
    dj_database_url.config(default=os.environ.get('DATABASE_URL'))
    
    }
    ALLOWED_HOSTS = ['.herokuapp.com', 'www.lectureme.ai', 'lectureme.ai', 'www.plutarch.us', 'plutarch.us']
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True





TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [

            BASE_DIR / "templates"
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "base.context_processors.google_analytics"
            ],
        },
    },
]

WSGI_APPLICATION = "lecturesgpt.wsgi.application"
ASGI_APPLICATION = "lecturesgpt.asgi.application"

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases




# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",},
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "America/New_York"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = '/static/'

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# settings.py

GA_MEASUREMENT_ID = 'G-2ZGPZC9TEC'


#python manage.py runserver 10.17.243.47:8000
X_FRAME_OPTIONS = 'SAMEORIGIN'


LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'ERROR',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'ERROR',
            'propagate': True,
        },
    },
}
