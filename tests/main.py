from PIL import Image
from pix2tex import cli

image_path = '../image_set/image1.png'
image = Image.open(image_path)

model = cli.LatexOCR()

print_code = model(image)

print(print_code)


