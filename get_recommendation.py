# For database
import os
import sys
import django
import random
import json

# If this file is not the same level as manage.py, then use this code
# Set the root path of Django ORM database, make sure finding settings.py
sys.path.append('D:/NUS/CETA_project/CETA/database')  # modify into the Django ORM database path
sys.path.append('D:/NUS/CETA_project/CETA/prompt_for_chatgptAPI')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'database.settings')  # setting env, make sure pointing at settings.py

# Initialize Django environment
django.setup()

from db.models import Cloth, Mask, Model, UserUpload
from django.db.models import Q
from query_images import query_cloth_by_dict

# For ChatGPT
from original_get_recommendation import parse_user_input


def get_recommendation(user_input):
    """
    @param user_input: Natural language string representing the user's clothing preferences. Example: "Show me a blue jacket."
    @type user_input: str
    
    @return: The file path of a randomly selected image from the query results, representing the requested clothing item. **OPEN ISSUE**
        If no matching query result, the function will return -1
    @rtype: str
    
    @raise Exception: None
    """
    parsed_result = parse_user_input(user_input)
    print(parsed_result) # for debugging

    queryset = query_cloth_by_dict(parsed_result)

    if not queryset:
        return -1

    image_path = queryset[random.choice(range(0,len(queryset)))].cloth_image
    return image_path


if __name__ == '__main__':
    # Example: User Input
    # No result example
    # user_input = "I want a blue long-sleeve shirt without pockets for summer."
    # Having result example
    user_input = "I want a Prussian-blue, mint-green, and purple Arjuna color-block sports bra from ERES, designed with a color-block pattern. It features a square neckline, crossover shoulder straps, and a straight hem. Made from cotton fabric, this sports bra is tailored for children, ideal for summer wear. It has a tight fit with a solid color texture, a deep V-neck collar, and no sleeves. The top is high-waisted with a closing hem and an H-type upper dress contour. The style reflects a sporty, Korean-inspired, fresh look, and it comes without pockets or additional accessories."
    
    img_path = get_recommendation(user_input)
    

    if img_path == -1:
        print("No matching result")
    else:
        print("Recommendation" + img_path)


