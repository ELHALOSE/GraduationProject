# Deafbot: AI-powered Sign Language Recognition and Communication with LLM

**Deafbot** is an AI-powered application designed to help bridge the communication gap for individuals with hearing impairments. The system recognizes sign language gestures in real-time, translates them into text, and provides context-aware responses using Artificial Intelligence. Additionally, it converts these text responses into speech, ensuring full accessibility for users.

## Key Features

- **Sign Language Gesture Recognition**: Utilizes the YOLO model and OpenCV to detect sign language gestures in real-time from a video stream.
- **Intelligent Responses**: Powered by OpenRouter API, the system generates context-aware replies to user inputs.
- **Speech Output**: Uses Google Text-to-Speech (gTTS) to convert the AI-generated text responses into speech, enabling communication with users who prefer audio responses.
- **Interactive User Interface**: The interface includes a live video feed for sign detection and a chat window for user interaction.

## Technologies Used

- **YOLO (You Only Look Once)**: Real-time object detection for identifying sign language gestures from video input.
- **OpenCV**: A computer vision library for image processing and video handling.
- **OpenRouter API**: AI model that generates relevant, context-aware responses based on user input.
- **gTTS (Google Text-to-Speech)**: Converts text responses into speech.
- **Flask**: A lightweight Python web framework used for the backend development of the application.

## Project Objective

The main objective of this project is to provide a platform where individuals with hearing impairments can effectively communicate through sign language. The system aims to:

1. Detect and recognize sign language gestures.
2. Translate gestures into text.
3. Generate context-aware replies using AI.
4. Provide audio responses to ensure accessibility for all users.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ELHALOSE/GraduationProject.git
