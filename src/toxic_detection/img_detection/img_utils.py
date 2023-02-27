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
