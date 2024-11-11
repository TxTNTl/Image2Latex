import subprocess
import sys

def run_text_recognition(image_path, rec_algorithm="LaTeXOCR", rec_batch_num=1, rec_model_dir=r"inference\rec_latex_ocr_infer", rec_char_dict_path=r"ppocr\utils\dict\latex_ocr_tokenizer.json"):
    command = [
        sys.executable,
        r"tools\infer\predict_rec.py", # 预测脚本路径
        f"--image_dir={image_path}",
        f"--rec_algorithm={rec_algorithm}",
        f"--rec_batch_num={rec_batch_num}",
        f"--rec_model_dir={rec_model_dir}",
        f"--rec_char_dict_path={rec_char_dict_path}"
    ]

    try:
 
        result = subprocess.run(command, check=True, text=True, capture_output=True,encoding='utf-8')
        print(type(result))
        stdout = str(result.stdout).strip()
        print( stdout)
        return stdout
    except subprocess.CalledProcessError as e:
        print("命令执行失败:", e)
        print("错误输出:", e.stderr)
        return e.stderr

if __name__ == "__main__":
    image_dir = r'dataset\28.png'  # 图片路径
    rec_algorithm = "LaTeXOCR"
    rec_batch_num = 1
    rec_model_dir = r"inference\rec_latex_ocr_infer"  # 预测模型路径
    rec_char_dict_path = r"ppocr\utils\dict\latex_ocr_tokenizer.json"  # 字典路径

    # 调用函数
    run_text_recognition(image_dir, rec_algorithm, rec_batch_num, rec_model_dir, rec_char_dict_path)
