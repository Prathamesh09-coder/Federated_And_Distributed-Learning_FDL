from flask import Flask, request, jsonify
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.model import SimpleNN

app = Flask(__name__)

model = SimpleNN()

MODEL_PATH = "global_model.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))


@app.route("/get_model", methods=["GET"])
def get_model():

    weights = {}

    for k, v in model.state_dict().items():
        weights[k] = v.tolist()

    return jsonify(weights)


@app.route("/update_model", methods=["POST"])
def update_model():

    data = request.json

    weights = {}

    for k, v in data.items():
        weights[k] = torch.tensor(v)

    model.load_state_dict(weights)

    torch.save(model.state_dict(), MODEL_PATH)

    return jsonify({"message": "Model Updated Successfully"})


if __name__ == "__main__":
    app.run(port=5000)