import json
import os

FEEDBACK_FILE = "app/data/feedback_logs.json"


def save_feedback(query, response, rating):

    feedback = {
        "query": query,
        "response": response,
        "rating": rating
    }

    data = []

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)

    data.append(feedback)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return True