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


if __name__ == "__main__":
    app.run(debug=True, port=5050)
