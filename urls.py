"""high_res URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from high_app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index_page, name='front_page'),  
    path('index_page/', index_page, name='index_page'), 
    path('about_page/', about_page, name='about_page'), 
    path("register/", register, name="register"),
    path("login/", login, name="login"),
    path('service_page/', service_page, name='service_page'), 
    path('why_page/', why_page, name='why_page'), 
    path('login_page/', login_page, name='login_page'), 
    path('home_page/', home_page, name='home_page'), 
    path('logout/', logout, name='logout'),
    path('high_page/', high_page, name='high_page'),
    path('convert_image/', convert_image, name='convert_image'),
    path('image_page/', image_page, name='image_page'), 
    path('style_transfer/', style_transfer, name='style_transfer'), 
    path('convert_images/', convert_images, name='convert_images'),
    path("style_transfer/", style_transfer, name="style_transfer"),
]


