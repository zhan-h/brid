from django.contrib import admin
from .models import *
admin.site.site_header = '鸟类识别后台管理'

# Register your models here.

admin.site.register(User)
admin.site.register(DeepseekUser)
