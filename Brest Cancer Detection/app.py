from flask import Flask,render_template,request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'brest-cancer-prediction-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        texture_mean = float(request.form['texture_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        texture_se = float(request.form['texture_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        concavity_se = float(request.form['concavity_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        smoothness_worst = float(request.form['smoothness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
        data = np.array([[texture_mean,area_mean,smoothness_mean,concavity_mean,symmetry_mean,fractal_dimension_mean,texture_se,area_se,smoothness_se,concavity_se,symmetry_se,fractal_dimension_se,smoothness_worst,concavity_worst,symmetry_worst,fractal_dimension_worst]])
        my_prediction = classifier.predict(data)
        return render_template('result.html', prediction=my_prediction)
if __name__ == "__main__":
    app.run(debug=True)        