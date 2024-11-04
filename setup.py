import cv2
import face_recognition
import os

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
def process_video(video_path, known_encodings, known_names):
    video_capture = cv2.VideoCapture(video_path)
    new_person_index = len(known_names) + 1  # Yangi odamlarni raqamlash uchun

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
    r"C:\\Users\\hp-se\\Pictures\\5264904665764597917.jpg",
    # r"C:\\Users\\hp-se\\OneDrive\\Pictures\\WIN_20241029_23_15_19_Pro.jpg",
]
video_path = r"C:\\Users\\hp-se\\Pictures\\WIN_20241029_23_14_31_Pro.mp4"

# === 4. Rasmlarni yuklab, ularning encodinglarini olish ===
known_encodings = load_image_encodings(image_paths)
known_names = [f"Person {i + 1}" for i in range(len(known_encodings))]  # Odamlar nomlari

# === 5. Video ichidagi yuzlarni rasmlar bilan taqqoslash ===
process_video(video_path, known_encodings, known_names)
