import os
import sys
import django

# If this file is not the same level as manage.py, then use this code below
# Set the root path of Django ORM database, make sure finding settings.py
# sys.path.append('D:/NUS/CETA/database')  # modify into the Django ORM database path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'database.settings')  # setting env, make sure pointing at settings.py

# Initialize Django environment
django.setup()

from db.models import Cloth, Mask, UserUpload, Model

# Warning: Delete all images!
def delete_Cloth():
    Cloth.objects.all().delete()

def delete_Mask():
    Mask.objects.all().delete()

def delete_UserUpload():
    UserUpload.objects.all().delete()

def delete_model():
    Model.objects.all().delete()

# main
if __name__ == '__main__':
    # delete_Cloth()
    # delete_Mask()
    # delete_UserUpload()
    delete_model()
