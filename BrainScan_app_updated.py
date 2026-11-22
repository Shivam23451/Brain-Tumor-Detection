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

# REPORTLAB - PDF GENERATION
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from xml.sax.saxutils import escape

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

# Medical report and doctor-entered prescription collections
medical_reports_collection = db["medical_reports"]
prescriptions_collection = db["prescriptions"]

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

    prediction_id = None
    

    try:

        prediction_document = {

            "image_id": image_id,

            "image_name": filename,

            "prediction": predicted_class,

            "confidence": confidence,

            "created_at": datetime.now()

        }

        prediction_result = predictions_collection.insert_one(
            prediction_document
        )
        prediction_id = str(prediction_result.inserted_id)

        print(
            "Prediction saved to MongoDB. ID:",
            prediction_id
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

        confidence=confidence,

        prediction_id=prediction_id

    )



# PDF GENERATION HELPERS

def create_pdf_response(story, filename):
    """Build a PDF in memory and return it as a downloadable response."""
    pdf_buffer = io.BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )
    document.build(story)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )


def get_prediction_for_form():
    """Load prediction details from MongoDB; never trust hidden prediction fields."""
    prediction_id = (request.form.get("prediction_id") or "").strip()

    if not ObjectId.is_valid(prediction_id):
        return None

    return predictions_collection.find_one(
        {"_id": ObjectId(prediction_id)}
    )


def pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="SmallNoteCustom",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    ))
    return styles


# AI MEDICAL REPORT PDF

@app.route("/generate-medical-report", methods=["POST"])
def generate_medical_report():

    patient_name = (request.form.get("patient_name") or "").strip()
    patient_age = (request.form.get("patient_age") or "").strip()
    patient_gender = (request.form.get("patient_gender") or "").strip()

    if not patient_name or len(patient_name) > 100:
        return "Please enter a valid patient name.", 400

    if not patient_age.isdigit() or not 1 <= int(patient_age) <= 120:
        return "Please enter a valid patient age.", 400

    allowed_genders = {"Male", "Female", "Other", "Prefer not to say"}
    if patient_gender not in allowed_genders:
        return "Please select a valid gender option.", 400

    try:
        prediction_record = get_prediction_for_form()
    except Exception as error:
        app.logger.exception("Could not retrieve prediction for medical report: %s", error)
        return "Unable to retrieve the MRI prediction record.", 500

    if not prediction_record:
        return "Prediction record not found. Please upload the MRI again.", 404

    prediction = str(prediction_record.get("prediction", "Not available"))
    confidence = prediction_record.get("confidence", "Not available")
    created_at = prediction_record.get("created_at", datetime.now())

    # Escape all user/model text before passing it to ReportLab Paragraph.
    safe_patient_name = escape(patient_name)
    safe_prediction = escape(prediction)
    safe_gender = escape(patient_gender)

    styles = pdf_styles()
    story = [
        Paragraph("BrainScan – AI MRI Analysis Report", styles["ReportTitleCustom"]),
        Paragraph(
            "ACADEMIC / RESEARCH INFORMATION — NOT A MEDICAL DIAGNOSIS",
            styles["Heading2"]
        ),
        Spacer(1, 8),
        Paragraph("Patient Information", styles["Heading2"]),
        Paragraph(f"<b>Name:</b> {safe_patient_name}", styles["BodyText"]),
        Paragraph(f"<b>Age:</b> {int(patient_age)}", styles["BodyText"]),
        Paragraph(f"<b>Gender:</b> {safe_gender}", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("AI Model Output", styles["Heading2"]),
        Paragraph(f"<b>Predicted category:</b> {safe_prediction}", styles["BodyText"]),
        Paragraph(
            f"<b>Model confidence:</b> {escape(str(confidence))}%",
            styles["BodyText"]
        ),
        Paragraph(
            f"<b>Analysis record date:</b> {escape(str(created_at))}",
            styles["BodyText"]
        ),
        Spacer(1, 12),
        Paragraph("Important Interpretation", styles["Heading2"]),
        Paragraph(
            "This document reports the output of an academic AI classification "
            "prototype. The predicted category and confidence score are not a "
            "confirmed diagnosis, do not establish tumor type or grade, and must "
            "not be used to select treatment or medication.",
            styles["BodyText"]
        ),
        Spacer(1, 8),
        Paragraph("Suggested next step", styles["Heading2"]),
        Paragraph(
            "Have the original MRI images reviewed by a qualified radiologist "
            "and discuss the findings with an appropriately qualified medical "
            "specialist.",
            styles["BodyText"]
        ),
        Spacer(1, 18),
        Paragraph(
            "Disclaimer: BrainScan is an academic and research prototype. "
            "This report is not a substitute for professional medical "
            "interpretation, diagnosis, or treatment advice.",
            styles["SmallNoteCustom"]
        )
    ]

    report_document = {
        "prediction_id": prediction_record["_id"],
        "patient_name": patient_name,
        "patient_age": int(patient_age),
        "patient_gender": patient_gender,
        "prediction": prediction,
        "confidence": confidence,
        "created_at": datetime.now()
    }

    try:
        medical_reports_collection.insert_one(report_document)
    except Exception as error:
        app.logger.exception("Could not save medical report metadata: %s", error)
        return "Unable to save the medical report record.", 500

    return create_pdf_response(story, "BrainScan_AI_Medical_Report.pdf")


# DOCTOR-ENTERED PRESCRIPTION PDF

@app.route("/generate-prescription", methods=["POST"])
def generate_prescription():

    patient_name = (request.form.get("patient_name") or "").strip()
    patient_age = (request.form.get("patient_age") or "").strip()
    doctor_name = (request.form.get("doctor_name") or "").strip()
    registration_number = (request.form.get("registration_number") or "").strip()
    clinical_diagnosis = (request.form.get("clinical_diagnosis") or "").strip()
    medicine = (request.form.get("medicine") or "").strip()
    dosage = (request.form.get("dosage") or "").strip()
    duration = (request.form.get("duration") or "").strip()
    doctor_advice = (request.form.get("doctor_advice") or "").strip()

    required_values = [
        patient_name, patient_age, doctor_name, registration_number,
        clinical_diagnosis, medicine, dosage, duration
    ]
    if any(not value for value in required_values):
        return "Please complete all required prescription fields.", 400

    if len(patient_name) > 100 or len(doctor_name) > 100:
        return "A name field is too long.", 400

    if not patient_age.isdigit() or not 1 <= int(patient_age) <= 120:
        return "Please enter a valid patient age.", 400

    if len(registration_number) > 60:
        return "Please enter a valid registration number.", 400

    if request.form.get("doctor_confirmation") != "on":
        return "Doctor confirmation is required to generate this draft.", 400

    try:
        prediction_record = get_prediction_for_form()
    except Exception as error:
        app.logger.exception("Could not retrieve prediction for prescription: %s", error)
        return "Unable to retrieve the MRI prediction record.", 500

    if not prediction_record:
        return "Prediction record not found. Please upload the MRI again.", 404

    # Escape every free-text field before adding it to the PDF.
    safe = lambda value: escape(value).replace("\n", "<br/>")
    styles = pdf_styles()

    story = [
        Paragraph("BrainScan – Doctor-Entered Prescription Draft", styles["ReportTitleCustom"]),
        Paragraph(
            "UNSIGNED DRAFT — NOT A VALID PRESCRIPTION",
            styles["Heading2"]
        ),
        Spacer(1, 8),
        Paragraph("Patient Details", styles["Heading2"]),
        Paragraph(f"<b>Patient name:</b> {safe(patient_name)}", styles["BodyText"]),
        Paragraph(f"<b>Age:</b> {int(patient_age)}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Doctor Details (entered by user)", styles["Heading2"]),
        Paragraph(f"<b>Doctor name:</b> {safe(doctor_name)}", styles["BodyText"]),
        Paragraph(
            f"<b>Registration number entered:</b> {safe(registration_number)}",
            styles["BodyText"]
        ),
        Spacer(1, 8),
        Paragraph("Clinical Information", styles["Heading2"]),
        Paragraph(f"<b>Clinical assessment:</b> {safe(clinical_diagnosis)}", styles["BodyText"]),
        Paragraph(f"<b>Medicine:</b> {safe(medicine)}", styles["BodyText"]),
        Paragraph(f"<b>Dosage and frequency:</b> {safe(dosage)}", styles["BodyText"]),
        Paragraph(f"<b>Duration:</b> {safe(duration)}", styles["BodyText"]),
        Paragraph(f"<b>Doctor's advice:</b> {safe(doctor_advice) if doctor_advice else 'Not provided'}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(
            "This prototype does not verify the identity, qualifications, or "
            "registration of the person entering these details. The form entry "
            "and checkbox are not an electronic signature or proof of doctor "
            "authorization. A qualified practitioner must independently review, "
            "correct, sign, and issue any actual prescription through an "
            "appropriate authorized process.",
            styles["SmallNoteCustom"]
        )
    ]

    prescription_document = {
        "prediction_id": prediction_record["_id"],
        "patient_name": patient_name,
        "patient_age": int(patient_age),
        "doctor_name": doctor_name,
        "registration_number": registration_number,
        "clinical_diagnosis": clinical_diagnosis,
        "medicine": medicine,
        "dosage": dosage,
        "duration": duration,
        "doctor_advice": doctor_advice,
        "status": "UNSIGNED_DRAFT_NOT_VALID_PRESCRIPTION",
        "created_at": datetime.now()
    }

    try:
        prescriptions_collection.insert_one(prescription_document)
    except Exception as error:
        app.logger.exception("Could not save prescription draft: %s", error)
        return "Unable to save the prescription draft.", 500

    return create_pdf_response(story, "BrainScan_Prescription_DRAFT.pdf")


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