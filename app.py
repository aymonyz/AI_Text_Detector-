from flask import Flask, render_template, request
import os
import uuid

from analyzer.engine import analyze_text
from analyzer.file_reader import read_docx, read_pdf

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    text = ""

    if request.method == "POST":
        # 1) نص مباشر
        text = request.form.get("text", "").strip()

        # 2) ملف
        file = request.files.get("file")
        if file and file.filename:
            ext = file.filename.lower().split(".")[-1]
            fname = f"{uuid.uuid4()}.{ext}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(path)

            if ext == "pdf":
                text = read_pdf(path)
            elif ext == "docx":
                text = read_docx(path)

        if text:
            result = analyze_text(text)

    return render_template("index.html", result=result, text=text)


if __name__ == "__main__":
    app.run(debug=True)
