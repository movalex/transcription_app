import os
from flask import Flask, request, send_file, render_template_string
import openai
import whisper
from werkzeug.utils import secure_filename

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8" /><title>Audio to Text Transcription</title></head>
<body>
<h1>Upload an Audio File for Transcription</h1>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="audio_file" accept="audio/*" required>
  <button type="submit">Transcribe</button>
</form>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

model = whisper.load_model("base")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return "No file part", 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Pass the file path to Whisper
        result = model.transcribe(filepath)
        
        # If you want just the text:
        transcription_text = result["text"]

        # Create a .txt version of the file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        
        return send_file(txt_filepath, as_attachment=True)

    except Exception as e:
        print("Error during transcription:", e)
        return "Error occurred while transcribing the audio.", 500

    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    # run the dev server
    app.run(debug=True, host='0.0.0.0', port=5000)