
from django.urls import path
from . import views

urlpatterns = [
    path('', views.getRoutes , name = "routes"),
    path('getSimilar/', views.getSimilar , name = "getSimilar"),
    path('getRecommendations/', views.getRecommendations , name = "getRecommendations"),


]