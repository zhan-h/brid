from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('request_verification_code/', views.request_verification_code, name='request_verification_code'),
    path('deepseek_api/', views.deepseek_api, name='deepseek_api'),
    path('audio_recognition/', views.audio_recognition, name='audio_recognition'),
    path('clean_media_folder/', views.clean_media_folder, name='clean_media_folder'),
    path('logout/', views.logout, name='logout'),
    path('image_recognition/', views.image_recognition, name='image_recognition')
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
