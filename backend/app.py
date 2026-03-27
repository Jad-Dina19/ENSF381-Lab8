from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

SEEDED_USERS = {
    "1": {"id": "1", "first_name": "Ava", "user_group": 11},
    "2": {"id": "2", "first_name": "Ben", "user_group": 22},
    "3": {"id": "3", "first_name": "Chloe", "user_group": 33},
    "4": {"id": "4", "first_name": "Diego", "user_group": 44},
    "5": {"id": "5", "first_name": "Ella", "user_group": 55},
}

MODEL_PATH = Path(__file__).resolve().parent / "src" / "random_forest_model.pkl"
PREDICTION_COLUMNS = [
    "city",
    "province",
    "latitude",
    "longitude",
    "lease_term",
    "type",
    "beds",
    "baths",
    "sq_feet",
    "furnishing",
    "smoking",
    "cats",
    "dogs",
]

app = Flask(__name__)
# For this lab, allow cross-origin requests from the React dev server.
# This broad setup keeps local development simple and is not standard
# production practice.
CORS(app)
users = deepcopy(SEEDED_USERS)


# TODO: Define these Flask routes with @app.route():
# - GET /users
#   Return 200 on success. The frontend still expects a JSON array,
#   so return list(users.values()) instead of the dict directly.
# - POST /users
#   Return 201 for a successful create, 400 for invalid input,
#   and 409 if the id already exists. Since users is a dict keyed by
#   id, use the id as the lookup key when checking for duplicates.
# - PUT /users/<user_id>
#   Return 200 for a successful update, 400 for invalid input,
#   and 404 if the user does not exist. Update the matching record
#   with users[user_id] = {...} after confirming the key exists.
# - DELETE /users/<user_id>
#   Return 200 for a successful delete and 404 if the user does not
#   exist. Delete with del users[user_id] after confirming the key
#   exists.
#   Exercise2
# - POST /predict_house_price

# get_user_ex1
@app.route("/users", methods=["GET"])
def get_user():
    return jsonify(list(users.values())), 200

# post_user_ex1
@app.route("/users", methods=["POST"])
def post_user():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"message": "Invalid input"}), 400

    user_id = json_data.get("id")
    if not user_id:
        return jsonify({"message": "User ID is required"}), 400

    if user_id in users:
        return jsonify({"message": f"User {user_id} already exists"}), 409

    users[user_id] = json_data
    return jsonify(json_data), 201

# put_user_ex1
@app.route("/users/<user_id>", methods=["PUT"])
def put_user(user_id):
    user = request.get_json()
    user_name = user.get("first_name")
    user_group = user.get("user_group")

    if user_id not in users:
        return jsonify({"message": f"User {user_id} not found"}), 404

    if not user_name or not user_group:
        return jsonify({"message": "Invalid input"}), 400

    users[user_id] = {
        "id": user_id,
        "first_name": user_name,
        "user_group": user_group,
    }
    return jsonify(json_data), 200

# delete_user_ex1
@app.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    if user_id not in users:
        return jsonify({"message": f"User {user_id } not found"}), 404

    del users[user_id]
    return jsonify({"message": f"Deleted user {user_id}"}), 200

##########################################################################

# post_predict_ex2
@app.route("/predict_house_price", methods=["POST"])
def predict_house_price():
    model = joblib.load(MODEL_PATH)
    data = request.get_json()
    pets_allowed = bool(data.get('pets_allowed', False))

    if not data:
        return jsonify({"message": "Invalid input"}), 400

    cats = pets_allowed
    dogs = pets_allowed

    sample_data = [
    data['city'],
    data['province'],
    float(data['latitude']),
    float(data['longitude']),
    data['lease_term'],
    data['type'],
    float(data['beds']),
    float(data['baths']),
    float(data['sq_feet']),
    data['furnishing'],
    data['smoking'],
    cats,
    dogs,
    ]
    sample_df = pd.DataFrame([sample_data], columns=[
        'city', 'province', 'latitude', 'longitude', 'lease_term',
        'type', 'beds', 'baths', 'sq_feet', 'furnishing',
        'smoking', 'cats', 'dogs'
    ])
    predicted_price = model.predict(sample_df)[0]

    return jsonify({"predicted_price": float(predicted_price)}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5050)