from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.http import JsonResponse
from django.views.static import serve

from core.storage import is_supabase_storage_configured


def root_health_check(request):
    return JsonResponse({
        "status": "online",
        "service": "HumayDrone Cloud Backend & Neural Inference API",
        "version": "1.1.0",
        "media_storage": "supabase" if is_supabase_storage_configured() else "local",
        "endpoints": {
            "survey_missions": "/api/survey-missions/",
            "flights": "/api/flights/",
            "images_upload": "/api/images/upload/",
            "admin": "/admin/",
        }
    })


urlpatterns = [
    path('', root_health_check, name='root-health-check'),
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),
]

if settings.DEBUG or not is_supabase_storage_configured():
    urlpatterns += [
        re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    ]
