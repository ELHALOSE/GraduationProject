from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import requests
import re
from gtts import gTTS
import os
import uuid

# إعداد Flask App
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JSON_AS_ASCII'] = False  # دعم الترميز العربي

# تحميل نموذج YOLO
model = YOLO("E:\\EELU\\GP\\project\\Data\\datasets\\best2.pt")

# إعداد OpenRouter API
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# عرض الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# معالجة اكتشاف الإشارات
@app.route('/detect', methods=['POST'])
def detect_sign():
    data = request.get_json()
    image_data = data.get('image', '')

    if not image_data:
        return jsonify({"error": "لم يتم إرسال صورة"}), 400

    try:
        img_data = base64.b64decode(image_data)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (640, 480))

        results = model(frame)
        detected_sign = "لم يتم اكتشاف إشارة"
        if len(results[0].boxes) > 0:
            detected_sign = results[0].names[int(results[0].boxes[0].cls)]

        return jsonify({"detectedSign": detected_sign})
    except Exception as e:
        return jsonify({"detectedSign": f"خطأ: {str(e)}"}), 500

# معالجة الرد باستخدام OpenRouter API
@app.route('/chat', methods=['POST'])
def chat_with_model():
    data = request.get_json()
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({"chat_response": "لم يتم تقديم نص"}), 400

    try:
        # إعداد الـ Prompt
        prompt = f"المستخدم قال بلغة الإشارة: '{input_text}'. أجب بجملة عربية قصيرة ومنطقية تتعلق بالسؤال."

        # إرسال طلب إلى OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Deafbot"
        }
        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.9,
            "top_p": 0.9
        }

        response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            print("Error Response:", response.text)
            raise requests.exceptions.RequestException(
                f"HTTP Error {response.status_code}: {response.text}"
            )

        response_data = response.json()
        response_text = response_data['choices'][0]['message']['content'].strip()

        # تنظيف الاستجابة
        if prompt in response_text:
            response_text = response_text.replace(prompt, "").strip()
        response_text = re.sub(r'[.!?:]{2,}|\s{2,}', ' ', response_text).strip()
        response_text = " ".join(response_text.split())

        if not response_text or len(response_text) < 2:
            response_text = "مش فاهم، ممكن توضيح؟"

        return jsonify({"chat_response": response_text})

    except requests.exceptions.RequestException as e:
        print(f"Error in chat_with_model: {str(e)}")
        return jsonify({"chat_response": f"خطأ: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"chat_response": f"خطأ: {str(e)}"}), 500

# معالجة طلب النطق
@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "لم يتم تقديم نص للنطق"}), 400

    try:
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(app.static_folder, audio_filename)
        
        tts = gTTS(text=text, lang='ar', slow=False)
        tts.save(audio_path)

        return jsonify({"audio_url": f"/static/{audio_filename}"})
    except Exception as e:
        print(f"Error in speak: {str(e)}")
        return jsonify({"error": f"خطأ: {str(e)}"}), 500

# تنظيف الملفات القديمة
@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        for filename in os.listdir(app.static_folder):
            if filename.startswith('response_') and filename.endswith('.mp3'):
                file_path = os.path.join(app.static_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# إعداد مسار ثابت للملف الصوتي
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000)