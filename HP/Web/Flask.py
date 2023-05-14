
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/submit_id', methods=['POST'])
def submit_id():
    id = request.form['id']
    # Aquí puedes hacer lo que quieras con el ID, como guardarlo en una variable
    return 'ID enviado: {}'.format(id)

@app.route('/submit_csv', methods=['POST'])
def submit_csv():
    
    csv_file = request.files['csv']
    # Aquí puedes hacer lo que quieras con el archivo CSV, como guardarlo en una variable
    return 'Archivo CSV enviado: {}'.format(csv_file.filename)

if __name__ == '__main__':
    app.run(debug=True)