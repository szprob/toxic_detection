from typing import Union

from PIL import Image


def read_im(input: Union[str, Image.Image]) -> Image.Image:
    """read im

    Args:
        input (Union[str,Image.Image]):
            img contains faces

    Returns:
        Image.Image
    """
    if isinstance(input, str):
        im = Image.open(input)
    else:
        im = input
    if not isinstance(im, Image.Image):
        raise ValueError("""`input` should be a str or bytes or Image.Image!""")

    im = im.convert("RGB")

    return im


def get_pieces_from_img(img):
    width, height = img.size
    
    # 计算左上角的坐标
    left = 0
    top = 0
    right = int(width * 0.65)
    bottom = int(height * 0.65)
    # 获取左上角的部分
    left_top = img.crop((left, top, right, bottom))
    
    # 计算右下角的坐标
    left = int(width * 0.35)
    top = int(height * 0.35)
    right = width
    bottom = height
    # 获取右下角的部分
    right_bottom = img.crop((left, top, right, bottom))
    
    # 计算左下角的坐标
    left = 0
    top = int(height * 0.35)
    right = int(width * 0.65)
    bottom = height
    # 获取左下角的部分
    left_bottom = img.crop((left, top, right, bottom))
    
    # 计算右上角的坐标
    left = int(width * 0.35)
    top = 0
    right = width
    bottom = int(height * 0.65)
    # 获取右上角的部分
    right_top = img.crop((left, top, right, bottom))
    
    # 计算中间部分的坐标
    middle_left = int(width * 0.2)
    middle_top = int(height * 0.2)
    middle_right = int(width * 0.8)
    middle_bottom = int(height * 0.8)
    # 获取中间部分
    middle = img.crop((middle_left, middle_top, middle_right, middle_bottom))
    
    
    # 左边
    left_width = width // 2
    right_width = width - left_width
    top_height = height // 2
    bottom_height = height - top_height
    
    left_half = img.crop((0, 0, left_width, height))
    right_half = img.crop((left_width, 0, width, height))
    top_half = img.crop((0, 0, width, top_height))
    bottom_half = img.crop((0, top_height, width, height))
    
    return left_top,left_bottom,right_top,right_bottom,middle,left_half,right_half,top_half,bottom_half


def get_max_dict(dict1, dict2):
    max_dict = {}
    for key in dict1.keys():
        if key in dict2:
            value1 = dict1[key]
            value2 = dict2[key]
            max_value = max(value1, value2)
            max_dict[key] = max_value
    return max_dict