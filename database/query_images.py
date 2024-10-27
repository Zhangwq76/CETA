import os
import sys
import django

# If this file is not the same level as manage.py, then use this code
# Set the root path of Django ORM database, make sure finding settings.py
# sys.path.append('D:/NUS/CETA/database')  # modify into the Django ORM database path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'database.settings')  # setting env, make sure pointing at settings.py

# Initialize Django environment
django.setup()

from db.models import Cloth, Mask, Model, Memory
from django.db.models import Q


# Query clothes by list. Use less.
def query_cloth_by_list(query_list):
    """
    @param query_list: The user's query must be engineered into a list.
    @return: result -> queryset: The query queryset.
    """
    query = Q()

    for keyword in query_list:
        query &= (
            Q(body_part=keyword) |
            Q(visible_part=keyword) |
            Q(species=keyword) |
            Q(fabric=keyword) |
            Q(placket=keyword) |
            Q(season=keyword) |
            Q(pocket=keyword) |
            Q(type_version=keyword) |
            Q(texture=keyword) |
            Q(skirt_length=keyword) |
            Q(cloth_orientation=keyword) |
            Q(waist=keyword) |
            Q(suitable_age__icontains=keyword) |
            Q(category=keyword) |
            Q(hem=keyword) |
            Q(upper_dress_contour=keyword) |
            Q(collar=keyword) |
            Q(sleeve_type=keyword) |
            Q(coat_length=keyword) |
            Q(sleeve_length=keyword) |
            Q(elbow=keyword) |
            Q(suit=keyword) |
            Q(pant_length=keyword) |
            Q(pants_contour=keyword) |
            Q(craft=keyword) |
            Q(style__icontains=keyword) |
            Q(accessories=keyword) |
            Q(pattern=keyword) |
            Q(color__icontains=keyword)
            # Q(description__icontains=keyword)  # maybe only keep this is also enough
        )

    results = Cloth.objects.filter(query)

    # show the querying result, can be omitted
    if results.exists():
        for cloth in results:
            print(
                f"Cloth ID: {cloth.cloth_id}, Image Path: {cloth.cloth_image}")
    else:
        print("No matching queryset found.")

    return results


# Query clothes by dict. Widely used
def query_cloth_by_dict(query_dict):
    """
    @param query_dict: The user's query being engineered into a dictionary by GPT.
    @type query_dict: dict
    @return: A queryset of all the matched clothes. Have to check whether there is no matched result.
    @rtype: queryset
    """
    query = Q()

    for field, value in query_dict.items():
        if field in ("color", "style"):
            query |= Q(description__icontains=value) | Q(**{f"{field}": value})
        else:
            query &= (Q(**{f"{field}": value}) | Q(description__icontains=value))
        # query &= Q(description__icontains=value)

    results = Cloth.objects.filter(query)

    # show the querying result, can be omitted
    # if results.exists():
    #     for cloth in results:
    #         print(
    #             f"Cloth ID: {cloth.cloth_id}, Image Path: {cloth.cloth_image}")
    # else:
    #     print("No matching queryset found.")

    return results


# Query clothes by dict
def query_mask(cloth_id, mask_type):
    """
    @param cloth_id: string of cloth_id
    @type cloth_id: str
    @param mask_type: collar, sleeves, buttons, ...
    @return: A queryset of all the matched masks. Have to check whether there is no matched result.
    @rtype: str
    """
    query = Q()
    query &= Q(cloth_id=cloth_id)
    query &= Q(mask_type=mask_type)

    results = Mask.objects.filter(query)

    # # show the querying result, can be omitted
    # if results.exists():
    #     for mask in results:
    #         print(
    #             f"Image Path: {mask.mask_image}")
    # else:
    #     print("No matching queryset found.")

    return results


# If the user requires to give him a model picture
def query_model(gender, pos_type):
    """
    @param gender: The gender of the model the user requires.
    @param pos_type: The position type (top, bottom, whole) the user requires.
    @return: result -> queryset
    """
    query = Q()
    query &= Q(model_sex=gender)
    query &= Q(position_type=pos_type)

    results = Model.objects.filter(query)

    # show the querying result, can be omitted
    if results.exists():
        for model in results:
            print(
                f"Image Path: {model.model_image}")
    else:
        print("No matching queryset found.")

    return results


# query data from memory (temporary)
def query_memory():
    """
    @param param_name: Description of param.
    @type param_name: param_type
    
    @return: the latest data in Memory table. (temporary)
    @rtype: queryset
    
    @raise Exception: none.
    """
    results = Memory.objects.order_by('-timestamp').first()
    return results




if __name__ == '__main__':
    # print("Test the query_cloth_by_list function")
    # keywords = ["khaki", "men"]
    # queryset1 = query_cloth_by_list(keywords)
    #
    # print('---------------------')

    # print("Test the query_cloth_by_dict function")
    # pairs = {
    #     'color': 'white',
    #     'season': 'summer',
    #     'fabric': 'cotton'
    # }
    # queryset2 = query_cloth_by_dict(pairs)
    #
    # print('----------------------')
    #
    # print("Test the query_mask function")
    # cloth_id = queryset2[0].cloth_id
    # mask_type = "collar"
    # queryset3 = query_mask(cloth_id, mask_type)
    #
    # print('----------------------')
    #
    # print("Test the query_model function")
    # gender = "female"
    # pos_type = "top"
    # queryset4 = query_model(gender, pos_type)
    #
    # print('----------------------')

    # print("Test the query_userUpload function")
    # image_type = "model"
    # queryset5 = query_userUpload(image_type)

    queryset6 = query_memory()
    print(queryset6.image)

