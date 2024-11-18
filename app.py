import os
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from flask import Flask, request, render_template, jsonify
import re
import time

app = Flask(__name__)

# Initialize Vertex AI
vertexai.init(project="project1-436215", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-002")

def multiturn_generate_content(audio_data, mime_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not audio_data or len(audio_data) == 0:
                return {"error": "Empty audio data received"}

            chat = model.start_chat()
            
            audio_part = Part.from_data(audio_data, mime_type=mime_type)
            
            response = chat.send_message(
                [
                    audio_part, 
                    "Please provide an accurate, word-for-word transcription of the audio content. "
                    "Then, on a new line, provide a brief sentiment analysis. "
                    "Format your response as follows:\n"
                    "Transcription: [exact transcription here]\n"
                    "Sentiment Analysis: [brief analysis here]"
                ],
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0.1,
                    "top_p": 0.95,
                },
                safety_settings=safety_settings
            )
            
            if not response.candidates:
                raise Exception("No response from model")
            
            content = response.candidates[0].content.parts[0].text

            # Parsing for transcription
            transcription_match = re.search(r'Transcription:\s*(.*?)(?=\nSentiment Analysis:|$)', content, re.DOTALL | re.IGNORECASE)
            
            # Fallback parsing if the expected format is not found
            if not transcription_match:
                transcription_match = re.search(r'(?:The audio says:|The transcript is:)\s*"?(.*?)"?(?=\n|$)', content, re.DOTALL)
            
            transcription = transcription_match.group(1).strip() if transcription_match else "Transcription not available."

            # Parsing for sentiment analysis
            sentiment_match = re.search(r'Sentiment Analysis:(.*)', content, re.DOTALL | re.IGNORECASE)
            sentiment = sentiment_match.group(1).strip() if sentiment_match else "Sentiment analysis not available."

            return {
                "transcription": transcription,
                "sentiment": sentiment,
                "raw_response": content
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait for 1 second before retrying
                continue
            return {"error": str(e)}

safety_settings = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            audio_data = None
            mime_type = None
            
            if 'audio' in request.files:
                audio_file = request.files['audio']
                audio_data = audio_file.read()
                mime_type = audio_file.content_type
            elif 'audio' in request.json:
                audio_data = base64.b64decode(request.json['audio'].split(',')[1])
                mime_type = 'audio/webm'
            
            if audio_data and mime_type:
                result = multiturn_generate_content(audio_data, mime_type)
                return jsonify(result)
            else:
                return jsonify({"error": "No audio data received"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
