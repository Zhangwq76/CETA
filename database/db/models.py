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


# User upload pictures
class UserUpload(models.Model):
    image_type = models.CharField(max_length=100)  # model? cloth? segment? Choose between these three and save
    user_description = models.TextField()  # save what user describe on this
    user_image = models.CharField(max_length=255)  # assign a place to save users' upload
    upload_time = models.DateTimeField(auto_now_add=True)  # save when the user upload this image.


"""REMEMBER TO MIGRATE!!!"""

"""
Copy and paste the followings to Terminal:

python manage.py makemigrations
python manage.py migrate
"""