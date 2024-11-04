import cv2 
import face_recognition
import numpy as np
import sqlite3

# Ma'lumotlar bazasini yaratish
def create_database():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    # Agar 'persons' jadvali mavjud bo'lmasa, uni yaratamiz
    c.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Encodinglarni ma'lumotlar bazasiga qo'shish
def insert_person(name, encoding):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO persons (name, encoding) VALUES (?, ?)
    ''', (name, encoding.tobytes()))
    conn.commit()
    conn.close()

# Encodinglarni ma'lumotlar bazasidan yuklash
def load_encodings():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute('SELECT name, encoding FROM persons')
    data = c.fetchall()
    conn.close()

    names = []
    encodings = []
    for name, encoding in data:
        names.append(name)
        encodings.append(np.frombuffer(encoding, dtype=np.float64))  # Yuz encodingini qayta tiklash
    
    return names, encodings

# === 1. Rasmlardan yuz encodinglarini olish ===
def load_image_encodings(image_paths):
    encodings = []
    for img_path in image_paths:
        image = face_recognition.load_image_file(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb_image)
        if enc:
            encodings.append(enc[0])  # Faqat birinchi yuzni olish
        else:
            print(f"Yuz topilmadi: {img_path}")
    return encodings

# === 2. Videoni qayta ishlash va yuzlarni solishtirish ===
def process_video(video_path):
    known_names, known_encodings = load_encodings()  # Ma'lumotlar bazasidan yuklash
    new_person_index = len(known_names) + 1  # Yangi odamlarni raqamlash uchun

    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = known_names[index]
            else:
                # Yangi odamni saqlash
                filename = f"person_{new_person_index}.jpg"
                cv2.imwrite(filename, frame)  # Yuzni saqlash
                insert_person(f"Person {new_person_index}", face_encoding)  # Yuz encodingini ma'lumotlar bazasiga saqlash
                known_encodings.append(face_encoding)  # Yuz encodingini saqlash
                known_names.append(f"Person {new_person_index}")  # Odamni nomi
                new_person_index += 1

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# === 3. Rasmlar va video yo‘llarini berish ===
image_paths = [
    # r"C:\\Users\\hp-se\\Pictures\\5264904665764597917.jpg",
    # r"C:\\Users\\hp-se\\OneDrive\\Pictures\\WIN_20241029_23_15_19_Pro.jpg",
]
video_path = r"C:\Users\hp-se\Downloads\document_5280868788164649217.mp4"

# === 4. Ma'lumotlar bazasini yaratish ===
create_database()

# === 5. Rasmlarni yuklab, ularning encodinglarini olish ===
known_encodings = load_image_encodings(image_paths)

# === 6. Video ichidagi yuzlarni rasmlar bilan taqqoslash ===
process_video(video_path)
