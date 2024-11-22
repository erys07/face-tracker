from flask import Flask, request, jsonify
import dlib
import cv2
import numpy as np
import os
from imutils import face_utils

app = Flask(__name__)

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def calculate_face_asymmetry(shape):
    left_points = shape[0:9]
    right_points = shape[9:17]

    left_y_avg = np.mean(left_points[:, 1])
    right_y_avg = np.mean(right_points[:, 1])

    difference = abs(left_y_avg - right_y_avg)
    asymmetry_percentage = (difference / max(left_y_avg, right_y_avg)) * 100
    return asymmetry_percentage


@app.route('/calculate_asymmetry', methods=['POST'])
def calculate_asymmetry():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem fornecida!"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Nenhuma imagem selecionada!"}), 400

    try:
        img_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Erro ao carregar a imagem!"}), 500

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
