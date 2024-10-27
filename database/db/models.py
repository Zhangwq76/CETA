from django.db import models

# Cloth table (without description engineered)
# class Cloth(models.Model):
#     cloth_id = models.CharField(max_length=20, primary_key=True)  # primary key
#     description = models.TextField()
#     cloth_image = models.CharField(max_length=255)
#     # We use CharField to save the path of image. We have to secure the path is correct!
#
#     def __str__(self):
#         return self.cloth_id


# Cloth table (description engineered)
class Cloth(models.Model):
    cloth_id = models.CharField(max_length=20, primary_key=True)
    cloth_image = models.CharField(max_length=255, null=False)

    # cloth features
    body_part = models.CharField(max_length=50, default='none')
    visible_part = models.CharField(max_length=50, default='none')
    species = models.CharField(max_length=50, default='none')
    fabric = models.CharField(max_length=50, default='none')
    placket = models.CharField(max_length=50, default='none')
    season = models.CharField(max_length=50, default='none')
    pocket = models.CharField(max_length=50, default='none')
    type_version = models.CharField(max_length=50, default='none')
    texture = models.CharField(max_length=50, default='none')
    skirt_length = models.CharField(max_length=50, default='none')
    cloth_orientation = models.CharField(max_length=50, default='none')
    waist = models.CharField(max_length=50, default='none')
    suitable_age = models.CharField(max_length=50, default='none')
    category = models.CharField(max_length=50, default='none')
    hem = models.CharField(max_length=50, default='none')
    upper_dress_contour = models.CharField(max_length=50, default='none')
    collar = models.CharField(max_length=50, default='none')
    sleeve_type = models.CharField(max_length=50, default='none')
    coat_length = models.CharField(max_length=50, default='none')
    sleeve_length = models.CharField(max_length=50, default='none')
    elbow = models.CharField(max_length=50, default='none')
    suit = models.CharField(max_length=50, default='none')
    pant_length = models.CharField(max_length=50, default='none')
    pants_contour = models.CharField(max_length=50, default='none')
    craft = models.CharField(max_length=50, default='none')
    style = models.CharField(max_length=50, default='none')
    accessories = models.CharField(max_length=50, default='none')
    pattern = models.CharField(max_length=50, default='none')

    # txt descriptions
    color = models.CharField(max_length=50, default='none')
    description = models.TextField()

    def __str__(self):
        return self.cloth_id


# Mask table
class Mask(models.Model):
    cloth = models.ForeignKey(Cloth, on_delete=models.CASCADE)  # foreign key, using cloth_id
    mask_id = models.IntegerField()
    mask_type = models.CharField(max_length=100)
    mask_image = models.CharField(max_length=255)
    # We use CharField to save the path of image. We have to secure the path is correct!

    class Meta:
        unique_together = ('cloth', 'mask_id')

    def __str__(self):
        return f'{self.cloth.cloth_id} - {self.mask_type}'


# We should also save some model images
class Model(models.Model):
    model_sex = models.CharField(max_length=10)   # male or female
    position_type = models.CharField(max_length=50)  # top, bottom, whole (上半身、下半身、全身) Choose between these three and save
    model_image = models.CharField(max_length=255)
    # We use CharField to save the path of image. We have to secure the path is correct!


# 临时存储地点，自写记忆功能. We try to deploy a memory function
class Memory(models.Model):
    text = models.CharField(max_length=255)  # save the text infomation
    image = models.CharField(max_length=255, default="no image")  # save the image information
    transaction = models.CharField(max_length=255, default="user upload")  # save the type of this data
    timestamp = models.DateTimeField(auto_now_add=True)  # save the timestamp when the data written into this table



"""REMEMBER TO MIGRATE!!!"""

"""
Copy and paste the followings to Terminal:

python manage.py makemigrations
python manage.py migrate
"""