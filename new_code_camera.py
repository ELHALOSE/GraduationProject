import cv2
import threading
import queue
import time
from ultralytics import YOLO

# تحميل نموذج YOLO المدرب على لغة الإشارة
model = YOLO("E:\\EELU\\GP\\project\\Data\\datasets\\best.pt")  # استبدل بمسار النموذج المدرب
print("Model loaded. Classes:", model.names)  # طباعة التصنيفات المدربة

# متغيرات لتخزين آخر إشارة مكتشفة
last_detected_sign = None
last_confidence = 0.0
last_box = None
detection_timeout = 15  # عدد الإطارات التي يجب أن تمر قبل مسح الإشارة القديمة
frames_since_last_detection = 0  # عداد الإطارات منذ آخر كشف ناجح

# دالة لاكتشاف الإشارة باستخدام YOLO
def detect_sign(frame):
    global last_detected_sign, last_confidence, last_box, frames_since_last_detection
    
    results = model(frame)  # إرسال الإطار للنموذج
    
    if len(results[0].boxes) > 0:
        detected_sign = results[0].names[int(results[0].boxes[0].cls)]  # اسم الإشارة
        confidence = float(results[0].boxes[0].conf)  # درجة الثقة
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])  # إحداثيات الصندوق
        
        last_detected_sign = detected_sign
        last_confidence = confidence
        last_box = (x1, y1, x2, y2)
        frames_since_last_detection = 0  # إعادة تعيين العداد
    else:
        frames_since_last_detection += 1  # زيادة العداد عند عدم الكشف

    # مسح الإشارة المخزنة إذا مر وقت طويل بدون كشف جديد
    if frames_since_last_detection > detection_timeout:
        last_detected_sign = None
        last_box = None

# دالة لتشغيل الكاميرا في thread منفصل
def camera_thread(frame_queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)  # إضافة الإطار إلى الـ queue
        else:
            break
    cap.release()

# دالة لمعالجة الإطارات وعرض النتائج على الشاشة
def processing_thread(frame_queue):
    frame_counter = 0  # عداد للإطارات

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()  # أخذ الإطار من الـ queue

            if frame_counter % 5 == 0:  # معالجة كل 5 إطارات (لتحسين الأداء)
                detect_sign(frame)  # اكتشاف الإشارة وتحديث آخر القيم المكتشفة

            # رسم المستطيل والنص في حال وجود آخر إشارة مكتشفة
            if last_box:
                x1, y1, x2, y2 = last_box
                color = (0, 255, 0)  # اللون الأخضر للصندوق
                thickness = 2  # سمك الخط
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)  # رسم المستطيل

                # عرض اسم الإشارة فوق الصندوق
                text_to_display = f"{last_detected_sign} ({last_confidence:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text_to_display, (x1, y1 - 10), font, 0.8, color, 2, cv2.LINE_AA)

            frame_counter += 1
            cv2.imshow("Sign Language Recognition", frame)  # عرض الصورة

        # خروج بالضغط على زر 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# إنشاء queue لتخزين الإطارات
frame_queue = queue.Queue()

# بدء thread الكاميرا
camera_thread = threading.Thread(target=camera_thread, args=(frame_queue,))
camera_thread.daemon = True
camera_thread.start()

# بدء thread المعالجة
processing_thread = threading.Thread(target=processing_thread, args=(frame_queue,))
processing_thread.daemon = True
processing_thread.start()

# الانتظار حتى انتهاء الـ threads
camera_thread.join()
processing_thread.join()

# تنظيف
cv2.destroyAllWindows()
