from django.db import models


# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

    class Meta:
        db_table = 'users'


class DeepseekUser(models.Model):
    username = models.CharField(max_length=50)
    text = models.TextField()
    deepseek_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

    class Meta:
        db_table = 'deepseek_users'

