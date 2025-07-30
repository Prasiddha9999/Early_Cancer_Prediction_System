from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('search/', views.search_report, name='search_report'),
    path('results/<int:prediction_id>/', views.results, name='results'),
]
