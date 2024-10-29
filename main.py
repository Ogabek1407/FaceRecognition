import cv2
import face_recognition
import os

# === 1. Rasmlardan yuz encodinglarini olish ===
def load_image_encodings(image_paths):
    encodings = []
    for img_path in image_paths:
        # Rasmni yuklash va RGB formatga o‘tkazish
        image = face_recognition.load_image_file(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Rasm ichidagi yuz encodingini olish
        enc = face_recognition.face_encodings(rgb_image)
        if enc:
            encodings.append(enc[0])
        else:
            print(f"Yuz topilmadi: {img_path}")
    return encodings

# === 2. Videoni qayta ishlash va yuzlarni solishtirish ===
def process_video(video_path, known_encodings):
    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Freymdan yuz joylashuvlarini aniqlash
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Har bir freymdagi yuzni rasmdagi yuzlar bilan solishtirish
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                index = matches.index(True)
                top, right, bottom, left = face_location

                # Moslik topilgan yuzga kvadrat chizish
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Match: Image {index+1}", (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Freymni ko‘rsatish
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# === 3. Rasmlar va video yo‘llarini berish ===
image_paths = [
    r"C:\\Users\\hp-se\\OneDrive\\Pictures\WIN_20241029_22_26_33_Pro.jpg",
    r"C:\\Users\\hp-se\\OneDrive\\Pictures\WIN_20241029_22_26_31_Pro.jpg",
    r"C:\\Users\\hp-se\\OneDrive\\Pictures\WIN_20241029_22_26_22_Pro.jpg",
    r"C:\\Users\\hp-se\\OneDrive\\Pictures\WIN_20241029_22_26_21_Pro.jpg"
]
video_path = r"C:\\Users\\hp-se\\OneDrive\\Pictures\WIN_20241029_22_25_25_Pro.mp4"

# === 4. Rasmlarni yuklab, ularning encodinglarini olish ===
known_encodings = load_image_encodings(image_paths)

# === 5. Video ichidagi yuzlarni rasmlar bilan taqqoslash ===
process_video(video_path, known_encodings)
