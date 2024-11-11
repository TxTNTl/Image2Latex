# Image2Latex

## Introduction
This project aims to research and implement a printed mathematical formula recognition algorithm and system based on deep learning. 
Through the analysis and design of deep learning algorithms, 
it enables the recognition of mathematical formulas from images and converts the recognition results into LaTeX-formatted mathematical expressions. 
The project includes experimental analysis of the algorithm and the development of a software prototype system, 
dedicated to providing an efficient solution for recognizing printed mathematical formulas.

## Features
- **Mathematical Formula Image Recognition**: Uses deep learning models to recognize images of printed mathematical formulas and extract the formula structure.
- **LaTeX Format Output**: Converts recognized mathematical formulas into LaTeX format, making it convenient for document usage.
- **Design Based on the Latex_OCR Model**: Employs a deep learning model based on Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and attention mechanisms to enhance the accuracy and robustness of mathematical formula recognition.

## Installation
The system used by the author is Windows 11 (It does not run on macOS by testing, Linux compatibility unknown), 
with Python version 3.8 (tested, does not work with 3.11, other versions untested) and PyTorch version 11.8.

For PyTorch installation, I used the following command. You may copy it or customize your installation from the official site [here](https://pytorch.org/get-started/locally/).

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Also for PaddlePaddle, I used:

`python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/`

## Run
When running the app.py file, an url will be displayed in the output window. 
You can copy this and paste it into your browser, or simply click on it to access the application.

## Possible Problem
If a “None” appears during runtime and the backend displays an encoding issue, 
modify line 17 in run.py to use "gb18030" or another encoding as needed.

## Author
ltx lxt lzw ltc cyt

## 中文版本
### 介绍
本项目旨在研究并实现基于深度学习的打印体数学公式图像识别算法和系统。
通过分析与设计深度学习算法，
实现对数学公式的图像识别，
并将识别结果转换为LaTeX格式的数学公式表达。
项目包含算法实验分析及软件原型系统开发，致力于为打印体数学公式识别提供高效的解决方案。

### 功能
- **数学公式图像识别**：利用深度学习模型识别打印体数学公式的图像，并提取公式结构。
- **LaTeX格式输出**：将识别出的数学公式转换为LaTeX格式，便于在文档中使用。
- **基于Latex_OCR模型设计**：采用基于卷积神经网络（CNN）、循环神经网络（RNN）以及注意力机制的深度学习模型，提高数学公式识别的准确性和鲁棒性。

### 安装
写者使用的系统为win11（经测试mac无法运行，Linux未知），
python版本为3.8（经测试3.11不行，别的版本未知），
pytorch版本为11.8

关于pytorch我使用的安装代码为，请酌情考虑复制或从官方[这里](https://pytorch.org/get-started/locally/)自定义安装

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

另外关于Palled我使用的为

`python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/`

### 运行
运行app.py文件，在输出窗口会显示一个地址，复制到浏览器或直接点击即可使用。

### 可能问题
运行过程中如果出现none，后台显示是encoding的问题时，更换run.py中的17行位置，换为"gb18030"或其他

### 作者
ltx lxt lzw ltc cyt


