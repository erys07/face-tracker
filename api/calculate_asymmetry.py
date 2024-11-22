import json
import numpy as np
import cv2
import mediapipe as mp

def calculate_face_asymmetry(landmarks):
    left_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    right_indices = [9, 10, 11, 12, 13, 14, 15, 16]

    left_points = np.array([[landmarks[i].x, landmarks[i].y] for i in left_indices])
    right_points = np.array([[landmarks[i].x, landmarks[i].y] for i in right_indices])

    left_y_avg = np.mean(left_points[:, 1])
    right_y_avg = np.mean(right_points[:, 1])

    difference = abs(left_y_avg - right_y_avg)
    asymmetry_percentage = (difference / max(left_y_avg, right_y_avg)) * 100
    return asymmetry_percentage

def handler(request):
    if request.method == "POST":
        file = request.files["image"]
        file_path = "/tmp/face_image.jpg"
        file.save(file_path)

        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        result = face_mesh.process(rgb_image)

        response = {}
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                asymmetry_percentage = calculate_face_asymmetry(face_landmarks.landmark)
                if asymmetry_percentage > 20:
                    response = {
                        "asymmetry_percentage": round(asymmetry_percentage, 2),
                        "message": "Assimétrico"
                    }
                else:
                    response = {
                        "asymmetry_percentage": round(asymmetry_percentage, 2),
                        "message": "Normal"
                    }
        else:
            response = {"error": "Nenhum rosto detectado na imagem."}

        return json.dumps(response)
    else:
        return json.dumps({"error": "Método não permitido"}), 405
