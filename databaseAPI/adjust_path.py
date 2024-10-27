import os

# 将数据库中保留的图像的路径从绝对路径改成相对路径
def adjust_path(image_path):
    # image_path = os.path.join('./cloth_data_', queryset[random.choice(range(0, len(queryset)))].cloth_image)
    # image_path = queryset[random.choice(range(0,len(queryset)))].cloth_image
    # 处理路径：去掉前面的 "./" 并加上 "database" 文件夹
    # if image_path.startswith('./'):
    #     image_path = image_path[2:]  # 去掉 "./"

    # 在前面加上 "database" 文件夹
    image_path = os.path.join('../database', image_path) 
        # 获取 cloth_image 值并处理路径
    # selected_item = queryset[random.choice(range(0, len(queryset)))]
    # image_path = selected_item.cloth_image
    # if image_path.startswith('./'):
    #     image_path = image_path[2:]
    # # 如果 cloth_image 已经包含完整路径，就直接使用，否则拼接路径
    # if not os.path.isabs(image_path):
    #     image_path = os.path.join('.\database', image_path)

    # print(f"Generated image path: {image_path}")

    # 使用 os.path.normpath 来规范路径
    image_path = os.path.normpath(image_path)

    return image_path