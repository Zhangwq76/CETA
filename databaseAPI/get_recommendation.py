# For database
import os
import sys
import django
import random
import json
from PIL import Image

# If this file is not the same level as manage.py, then use this code
# Set the root path of Django ORM database, make sure finding settings.py
sys.path.append('E:/CodeAllNight/CETA/CETA/database')  # modify into the Django ORM database path
sys.path.append('E:/CodeAllNight/CETA/CETA/')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'database.settings')  # setting env, make sure pointing at settings.py

# Initialize Django environment
django.setup()

from django.db.models import Q
from query_images import query_cloth_by_dict
from insert_images import insert_in_memory_table
from databaseAPI.adjust_path import adjust_path

# For ChatGPT
from promptforchatgptAPI.get_recommendation_add_sector import parse_user_input


def get_recommendation(user_input):
    """
    @param user_input: Natural language string representing the user's clothing preferences. Example: "Show me a blue jacket."
    @type user_input: str
    
    @return: A dictionary consists the following things
        flag: the result status of running this function
        parsed_dict: The search dict that need to be saved in the memory table
        img_path: The file path of a randomly selected image from the query results, representing the requested clothing item. **OPEN ISSUE**
    @rtype: dict
    
    @raise Exception: 
        1. If failed to connect to chatgpt, the function will return -1  # 在前端可以回应Network error等字样
        2. If no matching query result, the function will return -2   # 在前端对应回复用户数据库中没有合适推荐。。。数据数量多的情况下可以避免这种情况发生
        3. If the data path has some problem (e.g. the saved path is not correct), the function will return -3  # 可回应 Database error 等字样
    """
    try:
        parsed_result = parse_user_input(user_input)
        # print(parsed_result) # for debugging
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return {'flag': -1, 'parsed_dict': {}, 'img_path':''}

    queryset = query_cloth_by_dict(parsed_result)

    if not queryset:
        return {'flag': -2, 'parsed_dict': {}, 'img_path':''}

    # randomly select one image
    selected_item = queryset[random.choice(range(0, len(queryset)))]
    image_path = selected_item.cloth_image
    
    image_path = adjust_path(image_path)
    if not os.path.exists(image_path):
        # raise FileNotFoundError(0f"No such file or directory: {image_path}")
        return {'flag': -3, 'parsed_dict': {}, 'img_path':image_path}

    # 记忆储存
    insert_in_memory_table(parsed_result, "get recommendation", image_path)
    return {'flag': 0, 'parsed_dict': parsed_result, 'img_path': image_path}


# Use case example
if __name__ == '__main__':
    # Example: User Input - no result
    user_input = "I want a Prussian-blue, mint-green, and purple Arjuna color-block sports bra from ERES, designed with a color-block pattern. It features a square neckline, crossover shoulder straps, and a straight hem. Made from cotton fabric, this sports bra is tailored for children, ideal for summer wear. It has a tight fit with a solid color texture, a deep V-neck collar, and no sleeves. The top is high-waisted with a closing hem and an H-type upper dress contour. The style reflects a sporty, Korean-inspired, fresh look, and it comes without pockets or additional accessories."
    # user_input = "I want a cotton white t-shirt."

    result_dict = get_recommendation(user_input)
    
    if result_dict['flag'] == -1:
        print("get_recommendation.py: Fail to connect to ChatGPT API, please check your network connection.")
    elif result_dict['flag'] == -2:
        print("get_recommendation.py: Currently no qualified querying result due to the limited cloth dataset based on your request.")
    elif result_dict['flag'] == -3:
        print(f"get_recommendation.py: File not found error: no such file or directory: {result_dict['img_path']}")
    else:
        print("Recommendation" + result_dict['img_path'])

        image = Image.open(result_dict['img_path'])
        image.show()


