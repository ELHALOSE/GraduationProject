from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import requests
import re
from gtts import gTTS
import os
import time
import uuid

# إعداد Flask App
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JSON_AS_ASCII'] = False  # دعم الترميز العربي

# تحميل نموذج YOLO
model = YOLO("E:\\EELU\\GP\\project\\Data\\datasets\\best2.pt")

# إعداد OpenRouter API
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# صفحة HTML مع تصميم معدل مع إضافة زر النطق
template = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> Deafbot </title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background-color: #f0f2f5;
            font-family: 'Arial', sans-serif;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 1200px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
        }
        #video {
            width: 100%;
            max-width: 350px;
            height: 250px;
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            background-color: #000;
            flex-shrink: 0;
            object-fit: cover;
        }
        .chat-wrapper {
            width: 100%;
            max-width: 700px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .chat-container {
            height: 80vh;
            background-image: url('/static/Images/deaf.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            overflow-y: auto;
            padding: 15px;
            scrollbar-width: thin;
            scrollbar-color: #888 #f1f1f1;
            direction: rtl;
            position: relative;
        }
        .chat-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: inherit;
            filter: blur(5px);
            z-index: -1;
            border-radius: 15px;
        }
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        .chat-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .chat-container::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        #chat-input {
            width: 100%;
            padding: 12px;
            border: 1px solid #dee2e6;
            border-radius: 20px;
            font-size: 16px;
            resize: none;
            direction: rtl;
        }
        #send-button, #speak-button {
            padding: 12px 25px;
            background-color: #3498db;
            color: #fff;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 16px;
        }
        #send-button:hover, #speak-button:hover {
            background-color: #2980b9;
        }
        .message {
            max-width: 70%;
            margin-bottom: 10px;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 16px;
            line-height: 1.5;
            word-wrap: break-word;
            background-color: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(5px);
        }
        .user-message {
            background-color: #3498db;
            color: #fff;
            margin-right: 10px;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: rgba(233, 236, 239, 0.9);
            color: #2c3e50;
            border: 1px solid #dee2e6;
            margin-left: 10px;
            margin-right: auto;
            text-align: right;
        }
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                padding: 10px;
                gap: 10px;
            }
            #video, .chat-wrapper {
                max-width: 100%;
            }
            #video {
                order: 2;
                height: 200px;
            }
            .chat-wrapper {
                order: 1;
            }
            .chat-container {
                height: 50vh;
                font-size: 14px;
            }
            .message {
                font-size: 14px;
                padding: 8px 12px;
            }
            #chat-input {
                font-size: 14px;
            }
            #send-button, #speak-button {
                font-size: 14px;
                padding: 10px 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-wrapper">
            <div id="chat" class="chat-container"></div>
            <div class="input-area">
                <textarea id="chat-input" placeholder="اكتب رسالتك هنا..."></textarea>
                <button id="send-button">إرسال</button>
                <button id="speak-button" style="display: none;">النطق</button>
            </div>
        </div>
        <video id="video" autoplay playsinline></video>
    </div>

    <script>
    const video = document.getElementById('video');
    const chatContainer = document.getElementById('chat');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const speakButton = document.getElementById('speak-button');
    let lastDetectedSign = '';
    let accumulatedText = '';
    let currentUserMessage = null;
    let lastBotMessage = null;
    let isProcessingSign = false;
    let signBuffer = '';
    let lastSignTime = 0;
    const minSignInterval = 700;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Error accessing camera: ", err);
            addMessage("خطأ: لا يمكن الوصول إلى الكاميرا", "bot-message");
        });

    function addMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        messageDiv.textContent = text;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        if (className === 'bot-message') {
            lastBotMessage = messageDiv;
            speakButton.style.display = 'inline-block';
        }
        return messageDiv;
    }

    async function sendFrameToServer() {
        if (isProcessingSign) return;

        isProcessingSign = true;

        let canvas = document.createElement("canvas");
        let context = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        let imageData = canvas.toDataURL("image/jpeg").split(",")[1];

        let response = await fetch("/detect", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: imageData })
        });

        let result = await response.json();
        let detectedSign = result.detectedSign;
        let currentTime = Date.now();

        if (detectedSign && detectedSign !== "لم يتم اكتشاف إشارة" && detectedSign !== "undefined") {
            if (detectedSign !== lastDetectedSign || (currentTime - lastSignTime >= minSignInterval)) {
                if (detectedSign !== signBuffer) {
                    accumulatedText += detectedSign;
                    chatInput.value = accumulatedText;
                    lastDetectedSign = detectedSign;
                    signBuffer = detectedSign;
                    lastSignTime = currentTime;

                    if (accumulatedText) {
                        if (!currentUserMessage) {
                            currentUserMessage = addMessage(`الإشارات المكتشفة: ${accumulatedText}`, "user-message");
                        } else {
                            currentUserMessage.textContent = `الإشارات المكتشفة: ${accumulatedText}`;
                        }
                    }
                }
            }
        } else {
            signBuffer = '';
        }

        setTimeout(() => {
            isProcessingSign = false;
        }, 300);
    }

    chatInput.addEventListener('input', () => {
        accumulatedText = chatInput.value;
        if (accumulatedText) {
            if (!currentUserMessage) {
                currentUserMessage = addMessage(`الإشارات المكتشفة: ${accumulatedText}`, "user-message");
            } else {
                currentUserMessage.textContent = `الإشارات المكتشفة: ${accumulatedText}`;
            }
        } else if (currentUserMessage) {
            chatContainer.removeChild(currentUserMessage);
            currentUserMessage = null;
        }
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });

    sendButton.addEventListener('click', async () => {
        if (accumulatedText) {
            if (currentUserMessage) {
                currentUserMessage.textContent = `الإشارات المكتشفة: ${accumulatedText}`;
            } else {
                currentUserMessage = addMessage(`الإشارات المكتشفة: ${accumulatedText}`, "user-message");
            }

            let chatResponse = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: accumulatedText })
            });

            if (!chatResponse.ok) {
                const errorStatus = chatResponse.status;
                const errorText = await chatResponse.text();
                addMessage(`الرد: خطأ - تعذر معالجة الطلب (كود: ${errorStatus}, تفاصيل: ${errorText})`, "bot-message");
            } else {
                let chatResult = await chatResponse.json();
                let responseText = chatResult.chat_response;
                if (/^[\.z\s]+$/.test(responseText) || responseText.length < 2) {
                    responseText = "مش فاهم، ممكن توضيح؟";
                }
                lastBotMessage = addMessage(`الرد: ${responseText}`, "bot-message");
            }
            chatInput.value = '';
            accumulatedText = '';
            currentUserMessage = null;
        }
    });

    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendButton.click();
        }
    });

    speakButton.addEventListener('click', async () => {
        if (lastBotMessage) {
            const botText = lastBotMessage.textContent.replace('الرد: ', '');
            try {
                const response = await fetch('/speak', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: botText })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                const audio = new Audio(`${data.audio_url}?t=${new Date().getTime()}`); // إضافة معلمة زمنية لتجنب الكاش
                audio.oncanplaythrough = () => audio.play();
                audio.onerror = () => console.error('Failed to load audio');
                audio.onended = () => {
                    // حذف الملف بعد التشغيل (اختياري)
                    fetch('/cleanup', { method: 'POST' });
                };
            } catch (error) {
                console.error('Error fetching or playing audio:', error);
                addMessage('خطأ: تعذر تشغيل الصوت', 'bot-message');
            }
        }
    });

    setInterval(sendFrameToServer, 800);
</script>
</body>
</html>
"""

# عرض الصفحة الرئيسية
@app.route('/')
def index():
    return render_template_string(template)

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

        # طباعة الطلب للتصحيح
        print("Sending request to OpenRouter with payload:", payload)
        print("Headers:", headers)

        response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)

        # التحقق من حالة الاستجابة
        if response.status_code != 200:
            print("Error Response:", response.text)
            raise requests.exceptions.RequestException(
                f"HTTP Error {response.status_code}: {response.text}"
            )

        # استخراج النص من الاستجابة
        response_data = response.json()
        print("Response Data:", response_data)
        response_text = response_data['choices'][0]['message']['content'].strip()

        # تنظيف الاستجابة
        if prompt in response_text:
            response_text = response_text.replace(prompt, "").strip()
        response_text = re.sub(r'[.!?:]{2,}|\s{2,}', ' ', response_text).strip()
        response_text = " ".join(response_text.split())

        # إذا كانت الاستجابة فارغة أو غير منطقية
        if not response_text or len(response_text) < 2:
            response_text = "مش فاهم، ممكن توضيح؟"

        print(f"Generated Response: {response_text}")
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
        # توليد اسم ملف فريد باستخدام UUID
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(app.static_folder, audio_filename)
        
        # توليد ملف صوتي باستخدام gTTS
        tts = gTTS(text=text, lang='ar', slow=False)
        tts.save(audio_path)

        # إرجاع رابط الملف الصوتي
        return jsonify({"audio_url": f"/static/{audio_filename}"})
    except Exception as e:
        print(f"Error in speak: {str(e)}")
        return jsonify({"error": f"خطأ: {str(e)}"}), 500

# تنظيف الملفات القديمة (اختياري)
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
    app.run(debug=True, port=5000)