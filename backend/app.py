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
@ app.route("/users", methods=["GET"])
def get_users():
    return jsonify(list(users.values())), 200

# - POST /users
#   Return 201 for a successful create, 400 for invalid input,
#   and 409 if the id already exists. Since users is a dict keyed by
#   id, use the id as the lookup key when checking for duplicates.
@ app.route("/users", methods=["POST"])
def create_user():
    new_user = request.get_json(silent=True)

    user_id = new_user.get("id")
    first_name = new_user.get("first_name")
    user_group = new_user.get("user_group")

    if user_id is None or user_id == "" or first_name == "" or user_group == 0:
        return jsonify({
            "message": "Request body must include id, first_name, and user_group."
        }), 400

    user_id = str(user_id)

    if user_id in users:
        return jsonify({"message": f"User {user_id} already exists."}), 409

    created_user = {
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
    }
    users[user_id] = created_user
    return jsonify({
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
        "message": f"Created user {user_id}.",
    }), 201
# - PUT /users/<user_id>
#   Return 200 for a successful update, 400 for invalid input,
#   and 404 if the user does not exist. Update the matching record
#   with users[user_id] = {...} after confirming the key exists.
@ app.route("/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.get_json()
    first_name = data.get("first_name")
    user_group = data.get("user_group")
    if first_name == "" or user_group == 0:
        return jsonify({
            "message": "Request body must include id, first_name, and user_group."
        }), 400
    
    if user_id not in users:
        return jsonify({"message": f"User {user_id} not found."}), 404
    
    users[user_id] = {
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
    }
    return jsonify({
        "id": user_id,
        "first_name": first_name,
        "user_group": user_group,
        "message": f"Updated user {user_id}.",
    }), 200
    
# - DELETE /users/<user_id>
#   Return 200 for a successful delete and 404 if the user does not
#   exist. Delete with del users[user_id] after confirming the key
#   exists.

@app.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    if(user_id is None or user_id == ""):
        return jsonify({
            "message": "Request body must include id."
        }), 404
    user_id = str(user_id)
    if(user_id not in users):
                return jsonify({
            "message": "Request body must include id."
        }), 404
    
    users.pop(user_id)
    return jsonify({
        "message" : f"Deleted user {user_id} "
    })

#   Exercise2
# - POST /predict_house_price

@app.route("/predict_house_price", methods = ["POST"])
def predict_house_price():
    data = request.get_json(silent=True) or {}

    try:
        sample_df = pd.DataFrame([{
            "city": data["city"],
            "province": data["province"],
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "lease_term": data["lease_term"],
            "type": data["type"],
            "beds": float(data["beds"]),
            "baths": float(data["baths"]),
            "sq_feet": float(data["sq_feet"]),
            "furnishing": data["furnishing"],
            "smoking": data["smoking"],
            "cats": bool(data.get("pets", False)),
            "dogs": bool(data.get("pets", False)),
        }], columns=PREDICTION_COLUMNS)
    except KeyError as error:
        return jsonify({"message": f"Missing required field: {error.args[0]}."}), 400
    except (TypeError, ValueError):
        return jsonify({
            "message": "latitude, longitude, beds, baths, and sq_feet must be numbers."
        }), 400

    try:
        model = joblib.load(MODEL_PATH)
        predicted_price = float(model.predict(sample_df)[0])
    except Exception as error:
        return jsonify({"message": str(error)}), 400

    return jsonify({"predicted_price": predicted_price}), 200



if __name__ == "__main__":
    app.run(debug=True, port=5050)
    
