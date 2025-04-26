# 🧏‍♂️ Deafbot: AI-powered Sign Language Recognition and Communication with LLM

**Deafbot** is an AI-powered application designed to bridge the communication gap for individuals with hearing impairments. It recognizes sign language gestures in real-time, translates them into text, and uses Large Language Models to generate intelligent responses — which are then converted into speech for full accessibility.

---

## 🚀 Key Features

- 🔤 **Real-Time Sign Language Recognition**  
  Uses YOLO and OpenCV to detect sign language gestures from live video feed.

- 🧠 **Context-Aware AI Responses**  
  Interacts with users via smart replies generated using the OpenRouter API (GPT/Gemini/...).

- 🔊 **Speech Output**  
  Converts replies into speech using Google Text-to-Speech (gTTS) for audio-based communication.

- 💬 **Interactive Interface**  
  A simple web interface with live camera feed and chat-style interaction.

---

## 🛠️ Technologies Used

- **YOLO** – Real-time object detection.
- **OpenCV** – Video stream handling & preprocessing.
- **OpenRouter API** – For intelligent LLM-based response generation.
- **gTTS** – Google Text-to-Speech engine.
- **Flask** – Lightweight backend web server.
- **HTML/CSS/JavaScript** – For frontend UI.

---

## 🎯 Project Objective

The aim of this project is to enable effective two-way communication for deaf or mute individuals using sign language and AI, by:

1. Detecting gestures via webcam.
2. Translating them to natural language text.
3. Responding contextually using LLMs.
4. Generating clear speech responses.

---

## 📂 Project Structure

```bash
project/
├── app.py                 # Main Flask app handling backend logic
├── templates/
│   └── index.html         # Frontend HTML template
├── static/
│   ├── style.css          # Custom styles
│   ├── script.js          # Frontend interactivity scripts
│   └── Images/
│       └── deaf.png       # Interface or illustration image
```

---
## Get Clone
 1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ELHALOSE/GraduationProject.git
```
