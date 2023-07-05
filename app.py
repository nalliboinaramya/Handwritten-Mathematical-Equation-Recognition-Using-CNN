#new
from flask import Flask, render_template, request, jsonify, flash,redirect, url_for
import base64
import io
from PIL import Image
from equation_calculator import equation_solver_function
import os
import cv2
from solve_equation_file import solve_equation


app = Flask(__name__)
app.config['SECRET_KEY']='abcdef'
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('homepage.html')

@app.route('/upload_image', methods=['POST','GET'])
def upload_image():
    if 'equation.png' in os.listdir('static'):
        os.remove(os.path.join('static', 'equation.png'))
    return render_template('uploadimage.html')

@app.route('/canvas_image', methods=['POST','GET'])
def canvas_image():
    return render_template('canvasimage.html')


@app.route('/predict_upload_image', methods=['POST','GET'])
def predict_upload_image():
    if 'equation.png'  in os.listdir('static'):
        equation = equation_solver_function('static\\equation.png')
        if(len(equation) == 0):
            equation = "Empty image"
        else:
            equation = equation.replace('**','^')
        return jsonify(equation)
    else:
        return jsonify("Please write or upload some image first")

@app.route('/solve_equation_func', methods=['POST','GET'])
def solve_equation_func():
    input_text = request.form['inputequation']
    input_text = input_text.replace('^','**')
    result= solve_equation(input_text)
    return jsonify(str(result))


@app.route('/save', methods=['POST'])
def save():

    img_data = request.values['canvas_data']

    img_data = img_data.replace("data:image/png;base64,", "")

    img = Image.open(io.BytesIO(base64.b64decode(img_data)))

    new_img = Image.new('RGB', img.size, (255, 255, 255))
    new_img.paste(img, mask=img.split()[3])

    new_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'equation.png'))
    return 'Image saved!'


@app.route('/square', methods=['GET'])
def square():
    if 'equation.png' not in os.listdir('static'):
        return jsonify('Please upload or write something')
    equation = equation_solver_function('static\\equation.png')
    equation = equation[:-1] + "**"+equation[-1:]
    print(equation)
    print("Here")
    #return jsonify(equation)
    try:
        result = eval(equation)
    except Exception as e:
        return jsonify("Please write or upload properly equation.")
    if 'equation.png' in os.listdir('static'):
        os.remove(os.path.join('static','equation.png'))
    return jsonify(result)


@app.route('/upload', methods=['POST'])
def upload_file():
    image_file = request.files['image']
    if image_file:
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'equation.png'))
        input_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'equation.png'))
        ret,bw_img = cv2.threshold(input_image, 127,255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'equation.png'), bw_img)

        return render_template('uploadimage.html', filename='static\equation.png')
    else:
        flash('No image was selected.', 'warning')
        return render_template('uploadimage.html')


if __name__ == '__main__':
    app.run(debug=True)