from flask import Flask, render_template, request, jsonify
from test import initialize, extract_symptoms, predict_disease
import test  # ✅ Import the module to access global variables

app = Flask(__name__)

# Initialize model once at startup
print("🚀 Initializing model...")
initialize()
test.getSeverityDict()
test.getDescription()
test.getprecautionDict()
print("✅ Model ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms_input = data.get('symptoms', '')

        # Extract symptoms using the initialized model columns
        symptoms_list = extract_symptoms(symptoms_input, test.cols)

        if not symptoms_list:
            return jsonify({'error': 'No valid symptoms detected. Please describe your symptoms in more detail.'})

        disease, confidence, _ = predict_disease(symptoms_list)

        description = test.description_list.get(disease, 'No description available.')
        precautions = test.precautionDictionary.get(disease, [])

        return jsonify({
            'success': True,
            'disease': disease,
            'confidence': float(confidence),
            'description': description,
            'precautions': precautions,
            'detected_symptoms': symptoms_list
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
