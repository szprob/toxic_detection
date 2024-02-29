from typing import Union
import base64
import os
from PIL import Image
from urllib.parse import urlparse
import requests
import io


def _is_url(str):
    result = urlparse(str)
    return all([result.scheme, result.netloc])


def read_im(input: Union[str, bytes, Image.Image]) -> Image.Image:
    """read im

    Args:
        input (Union[str,Image.Image]):
            img path/url/base64/Image.Image

    Returns:
        Image.Image
    """
    if isinstance(input, str):
        # path
        if os.path.isfile(input):
            im = Image.open(input)
        # url
        elif _is_url(input):
            response = requests.get(input)
            if response.status_code != 200:
                raise ValueError("image_get_error")
            im = Image.open(io.BytesIO(response.content))
        # base64
        else:
            image_data = base64.b64decode(input)
            im = Image.open(io.BytesIO(image_data))
    elif isinstance(input, bytes):
        im = Image.open(io.BytesIO(input))
    else:
        im = input
    if not isinstance(im, Image.Image):
        raise ValueError("""`input` should be a str or bytes or Image.Image!""")

    im = im.convert("RGB")

    return im


def img2base64(img: Image.Image):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte)
    img_base64_str = img_base64.decode()
    return img_base64_str


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

    return (
        left_top,
        left_bottom,
        right_top,
        right_bottom,
        middle,
        left_half,
        right_half,
        top_half,
        bottom_half,
    )


def get_max_dict(dict1, dict2):
    max_dict = {}
    for key in dict1.keys():
        if key in dict2:
            value1 = dict1[key]
            value2 = dict2[key]
            max_value = max(value1, value2)
            max_dict[key] = max_value
    return max_dict