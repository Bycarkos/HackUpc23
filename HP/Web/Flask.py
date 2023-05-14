from flask import Flask, render_template, request
from load_model import predict_individual, predict_csv
from werkzeug.utils import secure_filename
import os
import pandas as pd


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/submit_id', methods=['POST'])
def submit_id():
    id = request.form['id']
    # Aquí puedes hacer lo que quieras con el ID, como guardarlo en una variable
    try:
        prediction = predict_individual(id)
        return 'Predicción: {} unitats'.format(prediction)

    except:
        return "Error: Formato Incorrecto"

@app.route('/submit_csv', methods=['GET','POST'])
def submit_csv():
    # f = request.files.get('file')
    # data_filename = secure_filename(f.filename)
    # f.save(os.path.join(app.config['UPLOAD_FOLDER'],
    #                         data_filename))
    # session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    csv_file = request.files['csv']
    csv_file.save(secure_filename(f'{csv_file.filename}'))
    # with open(csv_file,'rw') as file:
    # csv_file.write(f'static/{csv_file.filename}')
    # Aquí puedes hacer lo que quieras con el archivo CSV, como guardarlo en una variable
    dataset = pd.read_csv(f'{csv_file.filename}')
    prediction = predict_csv(dataset)
    return 'Prediccions model: {}'.format(list(prediction))

if __name__ == '__main__':
    app.run(debug=True)