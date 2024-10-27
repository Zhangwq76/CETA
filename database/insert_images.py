import os
import sys
import django
import json

# If this file is not the same level as manage.py, then use this code
# Set the root path of Django ORM database, make sure finding settings.py
sys.path.append('E:/CodeAllNight/CETA/CETA/database')  # modify into the Django ORM database path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'database.settings')  # setting env, make sure pointing at settings.py

# Initialize Django environment
django.setup()

# 导入模型
from db.models import Cloth, Mask, Model, Memory


# Iterate the images folder
def insert_in_cloth_table():
    """
    @return: none
    """
    root_dir = './cloth_data_'  # modify to the images folder path
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # according to the id_XXXXXXXXX naming structure...
        if os.path.isdir(subdir_path) and subdir.startswith('id_'):
            # extract cloth_id
            cloth_id = subdir.split('_')[1]
            print(f'Inserting %s into the database' % {cloth_id})

            cloth_image = None
            description = None

            # Iterate 3 files/folders
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)

                # jpg file
                if file.endswith('.jpg') and file.startswith(f'id_{cloth_id}'):
                    cloth_image = file_path

                # txt file
                elif file.endswith('.txt') and file.startswith(f'id_{cloth_id}'):
                    with open(file_path, 'r', encoding='utf-8') as txt_file:
                        description = txt_file.read().strip()

                # json file
                elif file.endswith('.json') and file.startswith(f'id_{cloth_id}'):
                    with open(file_path, 'r') as json_file:
                        features = json.load(json_file)

            color = ' '.join(description.split(' ')[0:2]).strip()

            # save into Cloth table
            if cloth_image and description and features:
                cloth_obj = Cloth(
                    cloth_id=cloth_id,
                    cloth_image=cloth_image,
                    body_part=features['body_part'],
                    visible_part=features['visible_part'],
                    species=features['species'],
                    fabric=features['fabric'],
                    placket=features['placket'],
                    season=features['season'],
                    pocket=features['pocket'],
                    type_version=features['type_version'],
                    texture=features['texture'],
                    skirt_length=features['skirt_length'],
                    cloth_orientation=features['cloth_orientation'],
                    waist=features['waist'],
                    suitable_age=features['suitable_age'],
                    category=features['category'],
                    hem=features['hem'],
                    upper_dress_contour=features['upper_dress_contour'],
                    collar=features['collar'],
                    sleeve_type=features['sleeve_type'],
                    coat_length=features['coat_length'],
                    sleeve_length=features['sleeve_length'],
                    elbow=features['elbow'],
                    suit=features['suit'],
                    pant_length=features['pant_length'],
                    pants_contour=features['pants_contour'],
                    craft=features['craft'],
                    style=features['style'],
                    accessories=features['accessories'],
                    pattern=features['pattern'],
                    color=color,
                    description=description,
                )
                cloth_obj.save()

                # the seg folder...
                seg_dir = os.path.join(subdir_path, f'id_{cloth_id}_seg')
                if os.path.exists(seg_dir) and os.path.isdir(seg_dir):
                    insert_in_mask_table(seg_dir, cloth_obj)


# process the seg folder, incorporated into the insert_in_cloth_table()
def insert_in_mask_table(seg_dir, cloth_obj):
    """
    @param seg_dir: mask picture folder's path
    @param cloth_obj: saving the cloth_id information, in order to set the foreign key
    @return: none
    """
    for bmp_file in os.listdir(seg_dir):
        if bmp_file.endswith('.bmp'):
            bmp_path = os.path.join(seg_dir, bmp_file)

            parts = bmp_file.split('_')
            if len(parts) >= 4:
                cloth_id = parts[1]
                mask_id = int(parts[2])
                mask_type = '_'.join(parts[3:]).replace('.bmp', '')

                # save into Mask table
                mask_obj = Mask(
                    cloth=cloth_obj,
                    mask_id=mask_id,
                    mask_type=mask_type,
                    mask_image=bmp_path
                )
                mask_obj.save()


# inserting test models images. Manual labeling required!!
def insert_in_model_table():
    """
    @return: none
    """
    root_dir = './model_image_'  # modify to the images folder path
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)

        # save into Model table
        model_obj = Model(
            model_sex="TBD",
            position_type="TBD",
            model_image=file_path,
        )
        model_obj.save()


# 将数据存入记忆表
def insert_in_memory_table(text, transaction, image_path=''):
    """
    @param text: text info.
    @type text: str

    @param image_path: img info.
    @type image_path: str
    
    @return: no return.
    
    @raise Exception: Exception description.
    """
    memory_obj = Memory(
        text=text,
        image=image_path,
        transaction=transaction
    )

    memory_obj.save()



# main
if __name__ == '__main__':
    # initialize our database
    # insert_in_cloth_table()
    # insert_in_model_table()

    # Test the insert_in_userUpload_table function
    # filepath = "D:/Poison_intern/出价8-3/8394807_2.jpg"
    # image_type = "model"
    # description = "This is my own photo, and I want to save it for further try-on."
    #
    # flag = insert_in_userUpload_table(filepath, image_type, description)
    pass