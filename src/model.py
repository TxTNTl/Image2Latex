from PIL import Image
from pix2tex.cli import LatexOCR


def output(str):
    img = Image.open(str)
    model = LatexOCR()
    return model(img)
