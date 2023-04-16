from django.urls import path
from .views import *


urlpatterns = [
    path('api/v1/weather', AiWeatherView.as_view()),

]
