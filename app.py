import streamlit as st
import cv2
import numpy as np

# Загрузка модели для обнаружения объектов
# from tensorflow.keras.models import load_model
# model = load_model('your_model.h5')

# Функция для обнаружения объектов на кадре
def detect_objects(frame):
    # Ваш код для обнаружения объектов на кадре
    # Изображение с нарисованными баундбоксами

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def main():
    st.title("Обнаружение объектов на видео")

    # Загрузка видео с помощью Streamlit
    uploaded_file = st.file_uploader("Выберите видео", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Чтение видео с использованием OpenCV
        video = cv2.VideoCapture(uploaded_file)

        # Отображение видеопотока и баундбоксов на каждом кадре
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Обработка кадра
            processed_frame = detect_objects(frame)

            # Отображение кадра с баундбоксами
            st.image(processed_frame, channels="BGR")

        video.release()

if __name__ == "__main__":
    main()
