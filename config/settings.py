"""
Django settings for HumayDrone Cloud Backend.
"""

from pathlib import Path
import importlib.util
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'django-insecure-humaydrone-dev-secret-key-change-in-prod',
)
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get('ALLOWED_HOSTS', '*').split(',')
    if h.strip()
] or ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party apps
    'rest_framework',
    'corsheaders',

    # Project apps
    'core',
    'inference',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    *(
        ['whitenoise.middleware.WhiteNoiseMiddleware']
        if importlib.util.find_spec('whitenoise') is not None
        else []
    ),
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database
# Default to SQLite for initial development; easily switchable to PostgreSQL via environment variables or DATABASE_URL
database_url = os.environ.get('DATABASE_URL')
if database_url:
    try:
        import dj_database_url
        DATABASES = {
            'default': dj_database_url.parse(database_url, conn_max_age=600)
        }
    except ImportError:
        DATABASES = {
            'default': {
                'ENGINE': os.environ.get('DB_ENGINE', 'django.db.backends.sqlite3'),
                'NAME': os.environ.get('DB_NAME', str(BASE_DIR / 'db.sqlite3')),
            }
        }
else:
    DATABASES = {
        'default': {
            'ENGINE': os.environ.get('DB_ENGINE', 'django.db.backends.sqlite3'),
            'NAME': os.environ.get('DB_NAME', str(BASE_DIR / 'db.sqlite3')),
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Manila'
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Supabase Integration (Dedicated public 'inference-media' bucket & Bearer JWT auth)
SUPABASE_URL = os.environ.get('SUPABASE_URL', '').strip().rstrip('/')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '').strip()
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '').strip()
SUPABASE_STORAGE_BUCKET = os.environ.get('SUPABASE_STORAGE_BUCKET', 'inference-media').strip() or 'inference-media'
SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET', '').strip()

_USE_SUPABASE_STORAGE = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
_HAS_WHITENOISE = importlib.util.find_spec('whitenoise') is not None

STORAGES = {
    'default': {
        'BACKEND': (
            'core.storage.SupabaseMediaStorage'
            if _USE_SUPABASE_STORAGE
            else 'django.core.files.storage.FileSystemStorage'
        ),
    },
    'staticfiles': {
        'BACKEND': (
            'whitenoise.storage.CompressedManifestStaticFilesStorage'
            if _HAS_WHITENOISE
            else 'django.contrib.staticfiles.storage.StaticFilesStorage'
        ),
    },
}

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',
    ],
}

# CORS settings for Mobile App & Web Dashboard
CORS_ALLOW_ALL_ORIGINS = True

# Support HTTPS reverse proxy headers (Render, Railway, Fly.io)
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
CSRF_TRUSTED_ORIGINS = [
    'https://*.onrender.com',
    'https://*.railway.app',
    'https://*.vercel.app',
    'http://localhost:5173',
    'http://localhost:8443',
    'http://localhost:3000',
]

# Celery Configuration
# Defaults to Redis, or falls back to eager execution in test/local modes if Redis is absent
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
# If CELERY_ALWAYS_EAGER is True, tasks run in-memory without needing a separate Redis instance
CELERY_TASK_ALWAYS_EAGER = os.environ.get('CELERY_ALWAYS_EAGER', 'False').lower() in ('true', '1')

