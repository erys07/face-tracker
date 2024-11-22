import json
import numpy as np
import cv2
import dlib
from imutils import face_utils

def calculate_face_asymmetry(shape):
    left_points = shape[0:9]
    right_points = shape[9:17]

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

        # Carregar imagem e detectar rosto
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        rects = detector(gray, 0)

        result = {}
        if len(rects) > 0:
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                asymmetry_percentage = calculate_face_asymmetry(shape)
                if asymmetry_percentage > 20:
                    result = {
                        "asymmetry_percentage": round(asymmetry_percentage, 2),
                        "message": "Assimétrico"
                    }
                else:
                    result = {
                        "asymmetry_percentage": round(asymmetry_percentage, 2),
                        "message": "Normal"
                    }
        else:
            result = {"error": "Nenhum rosto detectado na imagem."}

        return json.dumps(result)
    else:
        return json.dumps({"error": "Método não permitido"}), 405
