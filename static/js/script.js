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
            const audio = new Audio(`${data.audio_url}?t=${new Date().getTime()}`);
            audio.oncanplaythrough = () => audio.play();
            audio.onerror = () => console.error('Failed to load audio');
            audio.onended = () => {
                fetch('/cleanup', { method: 'POST' });
            };
        } catch (error) {
            console.error('Error fetching or playing audio:', error);
            addMessage('خطأ: تعذر تشغيل الصوت', 'bot-message');
        }
    }
});

setInterval(sendFrameToServer, 800);
