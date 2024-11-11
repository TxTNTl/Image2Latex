from flask import Flask, render_template, request, jsonify
import os
import uuid
import re
from run import run_text_recognition

app = Flask(__name__)

UPLOAD_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 调用模型
        result = run_text_recognition(app.config['UPLOAD_FOLDER'] + filename)

        return jsonify({'message': re.sub(r'\x1B\[[0-?9;]*[mK]', '', result)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
