
import csv
import pickle
import random
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

# Load the model
with open("/home/sunbeam/Documents/Sunbeam/Project/House Rent Prediction/HouseRentPrediction/xgboost.pkl", "rb") as file:
    model = pickle.load(file)


def read_csv(filename):
    values = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            values.append(row[0])
    return values


@app.route("/", methods=["GET"])
def root():
    # Read values from CSV file
    dropdown_values = read_csv(
        '/home/sunbeam/Documents/Sunbeam/Project/House Rent Prediction/HouseRentPrediction/frontend/cities.csv')
    return render_template('index.html', dropdown_values=dropdown_values)


@app.route("/classify", methods=["POST"])
def classify():
    try:
        # Validate and get form values
        seller_type = float(request.form.get("seller_type"))
        bedroom = float(request.form.get("Bedroom"))
        layout_type = float(request.form.get("layout_type"))
        property_type = float(request.form.get("property_type"))
        area = float(request.form.get("area"))
        furnish_type = float(request.form.get("furnish_type"))
        bathroom = float(request.form.get("bathroom"))
        city = float(request.form.get("city"))

        # Generate a random locality value (for demonstration purposes)
        locality_value = random.randint(1, 4000)

        # Create an array for prediction
        output_list = [seller_type, bedroom, layout_type, property_type, locality_value, area, furnish_type, bathroom, city]
        arr = np.array(output_list).reshape(1, -1)

        # Make prediction using the model
        answers = model.predict(arr)

        # Assuming 'answers' is an array or a single value from the prediction
        result = answers[0]

        return render_template('result.html', result=result)

    except ValueError as ve:
        # Handle validation errors
        return render_template('error.html', error_message=f"Invalid input: {str(ve)}")

    except Exception as e:
        # Handle other errors gracefully
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
