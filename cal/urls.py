from django.urls import path

from . import views

urlpatterns = [
    path('result',views.home, name='result'),
]
