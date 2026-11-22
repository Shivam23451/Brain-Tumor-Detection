import os
import json
import io
from datetime import datetime

import numpy as np
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file
)

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId
from werkzeug.utils import secure_filename

# FLASK APP

app = Flask(__name__)


# MONGODB CONNECTION

MONGO_URI = "mongodb://localhost:27017/"

client = MongoClient(MONGO_URI)

db = client["brain_scan"]

# GridFS for MRI images
fs = GridFS(db)

# Prediction collection
predictions_collection = db["predictions"]

print("MongoDB connected successfully.")


# CONFIGURATION

UPLOAD_FOLDER = os.path.join(
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# LOAD BRAIN TUMOR MODEL

MODEL_PATH = "brain_tumor_model.keras"

CLASS_NAMES_PATH = "class_names.json"

print("Loading brain tumor model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input,
        "function": preprocess_input
    }
)

print("Brain tumor model loaded successfully.")



# LOAD MRI VALIDATOR

VALIDATOR_PATH = "mri_validator.keras"

print("Loading MRI validator...")

mri_validator = keras.models.load_model(
    VALIDATOR_PATH
)

print("MRI validator loaded successfully.")


# LOAD CLASS NAMES

with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)

print("Classes:", class_names)


# HOME PAGE

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# PREDICTION PAGE

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    # GET REQUEST
    

    if request.method == "GET":

        return render_template(
            "predict.html"
        )

    
    # CHECK FILE
    

    if "mri_image" not in request.files:

        return render_template(
            "predict.html",
            error="Please select an MRI image."
        )

    file = request.files["mri_image"]

    if file.filename == "":

        return render_template(
            "predict.html",
            error="Please select an MRI image."
        )

    
    # CHECK EXTENSION
    

    filename = secure_filename(
        file.filename
    )

    extension = os.path.splitext(
        filename
    )[1].lower()

    print(
        "Uploaded filename:",
        filename
    )

    print(
        "Detected extension:",
        extension
    )

    if extension not in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:

        return render_template(
            "predict.html",
            error="Only JPG, JPEG and PNG images are supported."
        )

    if filename == "":

        return render_template(
            "predict.html",
            error="Invalid filename."
        )

    
    # SAVE IMAGE TEMPORARILY
    

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(
            filepath
        )

        print(
            "Image saved temporarily:",
            filename
        )

    except Exception as e:

        print(
            "File save error:",
            e
        )

        return render_template(
            "predict.html",
            error="Unable to save uploaded image."
        )

    # OPEN IMAGE
    

    try:

        image = Image.open(
            filepath
        ).convert("RGB")

        image = image.resize(
            (224, 224)
        )

        image_array = np.array(
            image
        ).astype(
            np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

    except Exception as e:

        print(
            "Image processing error:",
            e
        )

        try:
            os.remove(filepath)
        except Exception:
            pass

        return render_template(
            "predict.html",
            error="Invalid image file."
        )

    # MRI / NON-MRI VALIDATION
    

    print(
        "Checking whether image is a brain MRI..."
    )

    try:

        validator_prediction = float(
            mri_validator.predict(
                image_array,
                verbose=0
            )[0][0]
        )

    except Exception as e:

        print(
            "MRI validator error:",
            e
        )

        try:
            os.remove(filepath)
        except Exception:
            pass

        return render_template(
            "predict.html",
            error="MRI validation failed."
        )

    print(
        "MRI validator output:",
        validator_prediction
    )

    # Current validator is trained as:
    # 0 = MRI
    # 1 = non-MRI
    #
    # Therefore >= 0.50 means non-MRI.

    if validator_prediction >= 0.50:

        print(
            "INVALID IMAGE - NOT A BRAIN MRI"
        )

        try:
            os.remove(filepath)
        except Exception:
            pass

        return render_template(
            "predict.html",
            error="Please upload a valid brain MRI image."
        )

    print(
        "VALID BRAIN MRI DETECTED."
    )

    # SAVE VALID MRI TO MONGODB GRIDFS
    

    try:

        with open(
            filepath,
            "rb"
        ) as image_file:

            image_id = fs.put(
                image_file,
                filename=filename,
                content_type=(
                    file.content_type
                    or "image/jpeg"
                )
            )

        print(
            "MRI image saved to MongoDB."
        )

        print(
            "MongoDB Image ID:",
            image_id
        )

    except Exception as e:

        print(
            "MongoDB image save error:",
            e
        )

        try:
            os.remove(filepath)
        except Exception:
            pass

        return render_template(
            "predict.html",
            error="Unable to save MRI image to MongoDB."
        )

    
    # RESNET50 PREPROCESSING
    

    image_array = preprocess_input(
        image_array
    )

    
    # BRAIN TUMOR PREDICTION

    try:

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(
                predictions
            )
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = round(
            float(
                predictions[
                    predicted_index
                ] * 100
            ),
            2
        )

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            confidence
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        try:
            fs.delete(
                image_id
            )
        except Exception:
            pass

        try:
            os.remove(filepath)
        except Exception:
            pass

        return render_template(
            "predict.html",
            error="Prediction failed. Please try again."
        )

    # SAVE PREDICTION TO MONGODB
    

    try:

        prediction_document = {

            "image_id": image_id,

            "image_name": filename,

            "prediction": predicted_class,

            "confidence": confidence,

            "created_at": datetime.now()

        }

        predictions_collection.insert_one(
            prediction_document
        )

        print(
            "Prediction saved to MongoDB."
        )

    except Exception as e:

        print(
            "Prediction database error:",
            e
        )

        # Prediction can still be shown even if
        # the history insert fails.

    # RESULT PAGE

    return render_template(

        "result.html",

        image_url=url_for(
            "serve_mri",
            image_id=str(
                image_id
            )
        ),

        prediction=predicted_class,

        confidence=confidence

    )

# SERVE MRI IMAGE FROM MONGODB GRIDFS

@app.route(
    "/mri/<image_id>"
)
def serve_mri(image_id):

    try:

        print(
            "Requested MongoDB image:",
            image_id
        )

        object_id = ObjectId(
            image_id
        )

        image_file = fs.get(
            object_id
        )

        print(
            "MRI loaded from MongoDB:",
            image_file.filename
        )

        image_data = image_file.read()

        return send_file(

            io.BytesIO(
                image_data
            ),

            mimetype=(
                image_file.content_type
                or "image/jpeg"
            )
        )

    except Exception as e:

        print(
            "Error loading MRI from MongoDB:",
            e
        )

        return (
            "MRI image not found",
            404
        )

# PREDICTION HISTORY

@app.route(
    "/history"
)
def history():

    try:

        records = list(

            predictions_collection
            .find()
            .sort(
                "created_at",
                -1
            )

        )

        return render_template(

            "history.html",

            records=records

        )

    except Exception as e:

        print(
            "History error:",
            e
        )

        return (
            "Unable to load prediction history.",
            500
        )

# RUN APPLICATION

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False,

        use_reloader=False

    )