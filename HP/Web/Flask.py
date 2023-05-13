
# importing Flask and other modules
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


 
# Flask constructor
app = Flask(__name__)  

   
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'File uploaded Successfully!'
   return render_template('upload.html')
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       first_name = request.form.get("name")
       return "Aproximación inventario: "+ first_name
    return render_template("form.html")
 
if __name__=='__main__':
   app.run()
