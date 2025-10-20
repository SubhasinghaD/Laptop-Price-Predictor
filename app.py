from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model once at startup (faster and avoids repeated file I/O)
MODEL_PATH = os.path.join("model", "predictor.pickle")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def prediction(features):
    pred_value = model.predict([features])
    return pred_value


@app.route("/", methods=["GET", "POST"])
def index():
    pred = None

    if request.method == "POST":
        try:
            ram = int(request.form.get("ram"))
            weight = float(request.form.get("weight"))
            touchscreen = 1 if request.form.get("touchscreen") == "on" else 0
            ips = 1 if request.form.get("ips") == "on" else 0
            cpu = request.form.get("cpuname").lower().replace(" ", "")
            gpu = request.form.get("gpuname").lower()
            company = request.form.get("company").lower()
            os_name = request.form.get("opsys").lower()
            laptop_type = request.form.get("typename").lower().replace(" ", "")

            # Build feature list
            feature_list = [ram, weight, touchscreen, ips]

            # Define categories
            company_list = ['acer', 'apple', 'asus', 'dell', 'hp', 'lenovo', 'msi', 'other', 'toshiba']
            typename_list = ['2in1convertible', 'gaming', 'netbook', 'notebook', 'ultrabook', 'workstation']
            opsys_list = ['linux', 'mac', 'other', 'windows', 'chrome']
            cpu_list = ['amd', 'intelcorei3', 'intelcorei5', 'intelcorei7', 'other']
            gpu_list = ['amd', 'intel', 'nvidia']

            # One-hot encoding
            for item in company_list:
                feature_list.append(1 if item == company else 0)

            for item in typename_list:
                feature_list.append(1 if item == laptop_type else 0)

            for item in opsys_list:
                feature_list.append(1 if item == os_name else 0)

            for item in cpu_list:
                feature_list.append(1 if item == cpu else 0)

            for item in gpu_list:
                feature_list.append(1 if item == gpu else 0)

            # Make prediction
            pred = prediction(feature_list) * 302.77
            pred = float(np.round(pred[0], 2))

        except Exception as e:
            print("Error during prediction:", e)
            pred = "Error"

    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True)
