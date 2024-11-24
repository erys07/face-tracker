from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import mediapipe as mp
import requests

app = Flask(__name__)


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


@app.route('/face-asymmetry', methods=['POST'])
def handler():
    if request.method == "POST":
        image_url = request.json.get("image_url")

        if not image_url:
            return jsonify({"error": "URL da imagem não fornecida."}), 400

        # Baixar a imagem da URL
        try:
            response = requests.get(image_url)
            image = np.array(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Falha ao baixar a imagem: {str(e)}"}), 500

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        result = face_mesh.process(rgb_image)

        response = {}
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                asymmetry_percentage = calculate_face_asymmetry(face_landmarks.landmark)
                if asymmetry_percentage > 10:
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

        return jsonify(response)
    else:
        return jsonify({"error": "Método não permitido"}), 405


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
