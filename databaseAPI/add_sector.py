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
sys.path.append('E:/CodeAllNight/CETA/CETA/databaseAPI')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'database.settings')  # setting env, make sure pointing at settings.py

# Initialize Django environment
django.setup()

from django.db.models import Q
from query_images import query_cloth_by_dict, query_memory, query_mask
from insert_images import insert_in_memory_table
from adjust_path import adjust_path

# For ChatGPT
from promptforchatgptAPI.get_recommendation_add_sector import parse_user_input, parse_accessory_demand, update_feature_with_accessory


def add_sector_by_memory(user_input):
    """
    @param user_input: Natural language string representing the user's clothing preferences. Example: "I want to add a belt on this t-shirt."
    @type user_input: str
    
    @return: 
        1. The search dict that need to be saved in the memory table
        2. The file path of a randomly selected image from the query results, representing the requested clothing item. **OPEN ISSUE**
    @rtype: str
    
    @raise Exception: 
        1. If nothing in the system's memory but the user still try to call the memory, return -1  # 提示用户还没有发过消息，不存在历史记录
        2. If no matching query result, the function will return -2   # 在前端对应回复用户数据库中没有合适推荐。。。数据数量多的情况下可以避免这种情况发生
        3. If the data path has some problem (e.g. the saved path is not correct), the function will return -3  # 可回应 Database error 等字样
    """
    accessory = parse_accessory_demand(user_input)
    latest_memory = query_memory()
    if not latest_memory:
        return {'flag': -1, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path':'', 'original_image_path': ''}

    original_dict = latest_memory.text
    # original_img_path = query_memory().image
    new_dict = update_feature_with_accessory(original_dict, accessory)
    queryset = query_cloth_by_dict(new_dict)

    if not queryset:
        return {'flag': -2, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path':'', 'original_image_path': ''}
    
    # --获取目标组件所在的衣服--
    selected_item = queryset[random.choice(range(0, len(queryset)))]
    selected_image_path = selected_item.cloth_image
    selected_image_path = adjust_path(selected_image_path)
    if not os.path.exists(selected_image_path):
        # raise FileNotFoundError(f"No such file or directory: {selected_image_path}")
        return {'flag': -31, 'new_dict':{}, 'selected_image_path': selected_image_path, 'selected_mask_path':'', 'original_image_path': ''}
    
    # --获取目标组件的掩码--
    selected_cloth_id = selected_item.cloth_id
    mask_queryset = query_mask(selected_cloth_id, accessory)
    selected_mask = mask_queryset[random.choice(range(0, len(mask_queryset)))]
    selected_mask_path = selected_mask.mask_image
    selected_mask_path = adjust_path(selected_mask_path)
    if not os.path.exists(selected_mask_path):
        # raise FileNotFoundError(f"No such file or directory: {selected_mask_path}")
        return {'flag': -32, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path': selected_mask_path, 'original_image_path': ''}
    
    # -- 获取原衣服路径--
    original_image_path = latest_memory.image
    original_image_path = adjust_path(original_image_path)
    if not os.path.exists(original_image_path):
        # raise FileNotFoundError(f"No such file or directory: {selected_image_path}")
        return {'flag': -33, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path':'', 'original_image_path': original_image_path}

    return {
        'flag': 0, 
        'new_dict': new_dict, 
        'selected_image_path': selected_image_path, 
        'selected_mask_path': selected_mask_path, 
        'original_image_path': original_image_path
    }


def add_sector_by_uploading(user_input):
    try:
        parsed_result = parse_user_input(user_input)
        print(parsed_result) # for debugging
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return {'flag': -1, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path':''}
    
    # querydict = update_feature_with_accessory(parsed_result, accessory)
    queryset = query_cloth_by_dict(parsed_result)

    if not queryset:
        return {'flag': -2, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path':'', 'original_image_path': ''}
    
    # --获取目标组件所在的衣服--
    selected_item = queryset[random.choice(range(0, len(queryset)))]
    selected_image_path = selected_item.cloth_image
    selected_image_path = adjust_path(selected_image_path)
    if not os.path.exists(selected_image_path):
        # raise FileNotFoundError(f"No such file or directory: {selected_image_path}")
        return {'flag': -31, 'new_dict':{}, 'selected_image_path': selected_image_path, 'selected_mask_path':'', 'original_image_path': ''}
    
    # --获取目标组件的掩码--
    selected_cloth_id = selected_item.cloth_id
    print(selected_cloth_id) # for debugging
    mask_queryset = query_mask(selected_cloth_id, parsed_result['accessories'])
    print(mask_queryset) # for debugging 
    selected_mask = mask_queryset[random.choice(range(0, len(mask_queryset)))]
    selected_mask_path = selected_mask.mask_image
    selected_mask_path = adjust_path(selected_mask_path)
    if not os.path.exists(selected_mask_path):
        # raise FileNotFoundError(f"No such file or directory: {selected_mask_path}")
        return {'flag': -32, 'new_dict':{}, 'selected_image_path': '', 'selected_mask_path': selected_mask_path, 'original_image_path': ''}

    # return {
    #     'flag': 0, 
    #     'new_dict': querydict, 
    #     'selected_image_path': selected_image_path, 
    #     'selected_mask_path': selected_mask_path
    # }
    result_mask = {'flag': 0, 'new_dict':{}, 'selected_image_path': selected_image_path, 'selected_mask_path': selected_mask_path, 'original_image_path': '', 'accessory': parsed_result['accessories']}    
    # print(result_mask) # for debugging 
    return result_mask
    # , 'accessory': parsed_result['accessories']

# Use case example
if __name__ == '__main__':
    # # Example: User Input
    # # No result example
    # user_input = "I want a blue long-sleeve shirt without pockets for summer."
    # # Having result example
    # # user_input = "I want a simple green long dress for the night party."
    
    # result_dict = add_sector_by_memory(user_input)
    
    # if result_dict['flag'] == -1:
    #     print("add_sector.py: Fail to connect to ChatGPT API, please check your network connection.")
    # elif result_dict['flag'] == -2:
    #     print("add_sector.py: Currently no qualified querying result due to the limited cloth dataset based on your request.")
    # elif result_dict['flag'] == -31:
    #     print(f"add_sector.py: File not found error: no such file or directory: {result_dict['selected_image_path']}")
    # elif result_dict['flag'] == -32:
    #     print(f"add_sector.py: File not found error: no such file or directory: {result_dict['selected_mask_path']}")
    # elif result_dict['flag'] == -33:
    #     print(f"add_sector.py: File not found error: no such file or directory: {result_dict['original_image_path']}")
    # else:
    #     print("Selected cloth path" + result_dict['selected_image_path'])
    #     print("Selected mask path" + result_dict['selected_mask_path'])
    #     print("Original cloth path" + result_dict['original_image_path'])

    #     # 记忆储存
    #     insert_in_memory_table(result_dict['new_dict'], "add sector", result_dict['img_path'])
    #     image = Image.open(result_dict['img_path'])
    #     image.show()

    # =================================================================================

    # Example: User Input
    # No result example
    user_input = "I want a blue long-sleeve shirt without pockets for summer."
    # Having result example
    # user_input = "I want a simple green long dress for the night party."
    
    result_dict = add_sector_by_uploading(user_input, "belt")
    
    if result_dict['flag'] == -1:
        print("add_sector.py: Fail to connect to ChatGPT API, please check your network connection.")
    elif result_dict['flag'] == -2:
        print("add_sector.py: Currently no qualified querying result due to the limited cloth dataset based on your request.")
    elif result_dict['flag'] == -31:
        print(f"add_sector.py: File not found error: no such file or directory: {result_dict['selected_image_path']}")
    elif result_dict['flag'] == -32:
        print(f"add_sector.py: File not found error: no such file or directory: {result_dict['selected_mask_path']}")
    else:
        print("Selected cloth path: " + result_dict['selected_image_path'])
        print("Selected mask path: " + result_dict['selected_mask_path'])

        # # 记忆储存
        # insert_in_memory_table(result_dict['new_dict'], "add sector", result_dict['img_path'])
        # image = Image.open(result_dict['img_path'])
        # image.show()



