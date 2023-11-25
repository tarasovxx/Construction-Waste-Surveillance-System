import asyncio

import cv2
import streamlit as st
import plotly.express as px

from process import save_cropped_contours
from inference_model import classification_trash

fkko = {
    "tree": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ЛЕСНОГО ХОЗЯЙСТВА|ОТХОДЫ СТРОИТЕЛЬСТВА|Отходы при уборке урожая зерновых и зернобобовых культур|зелень древесная|ОТХОДЫ ПРОИЗВОДСТВА БУМАГИ И БУМАЖНЫХ ИЗДЕЛИЙ".lower(),
    "bricks": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ|ОТХОДЫ ДОБЫЧИ ПОЛЕЗНЫХ ИСКОПАЕМЫХ".lower(),
    "concrete": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ|ОТХОДЫ ДОБЫЧИ ПОЛЕЗНЫХ ИСКОПАЕМЫХ".lower(),
    "earth": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ПРОИЗВОДСТВА ПИЩЕВЫХ ПРОДУКТОВ, НАПИТКОВ, ТАБАЧНЫХ ИЗДЕЛИЙ|ОТХОДЫ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ".lower()
}



def display_object_info(obj, mat):


    # Display additional information
    # Отображение графика
    fig = px.bar(x=list(obj.keys()), y=list(obj.values()),
                 labels={'x': 'Material', 'y': 'Probability'}, title='Material Probabilities')
    st.plotly_chart(fig)

    # Обработка события при клике на колонку
    sorted_attr = sorted(obj.items(), key=lambda item: item[1], reverse=True)
    selected_material = st.selectbox('Наиболее вероятный материал отходов:', sorted_attr[0])
    related_categories = fkko[selected_material].split('|')[0:3]

    # Отображение связанных категорий
    st.write(f"Соответствующие категории отходов по ФККО (Федеральный классификационный каталог отходов):")
    for category in related_categories:
        st.write(f"- {category}")
    st.write("---")



def main():
    st.title("Users. Система контроля за строительными отходами.")

    inference_msg = st.empty()
    st.sidebar.title("Конфиграция")

    input_source = st.sidebar.radio(
        "Выберите источник входных данных",
        ('RTSP', 'Локальное видео'))

    save_output_video = st.sidebar.radio("Сохранить выходное видео?", ('Да', 'Нет'))
    if save_output_video == 'Да':
        nosave = False
        display_labels = False
    else:
        nosave = True
        display_labels = True

    save_poor_frame = st.sidebar.radio("Сохранить кадры с плохой производительностью?", ('Да', 'Нет'))
    if save_poor_frame == "Да":
        save_poor_frame__ = True
    else:
        save_poor_frame__ = False

    # ------------------------- LOCAL VIDEO ------------------------------
    if input_source == "Локальное видео":
        video_bytes = st.sidebar.file_uploader("Выберите входное видео", type=["mp4", "avi"],
                                               accept_multiple_files=False)
        video_capture = None
        if video_bytes is not None:
            # Сохраняем видео на диск
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as file:
                file.write(video_bytes.read())

            video_capture = cv2.VideoCapture(video_path)

        if st.sidebar.button("Начало обработки"):
            output_video_path, frame_, milliseconds = asyncio.run(save_cropped_contours(video_path))
            seconds, milliseconds = divmod(milliseconds, 1000)
            minutes, seconds = divmod(seconds, 60)

            st.video(video_bytes, start_time=int(seconds + minutes * 60 + bool(milliseconds)))

            # Обработка нейросетью
            # detect(source=video.name, stframe=stframe, kpi1_text=kpi1_text, kpi2_text=kpi2_text, kpi3_text=kpi3_text,
            #        js1_text=js1_text, js2_text=js2_text, js3_text=js3_text, conf_thres=float(conf_thres), nosave=nosave,
            #        display_labels=display_labels, conf_thres_drift=float(conf_thres_drift),
            #        save_poor_frame__=save_poor_frame__, inf_ov_1_text=inf_ov_1_text, inf_ov_2_text=inf_ov_2_text,
            #        inf_ov_3_text=inf_ov_3_text, inf_ov_4_text=inf_ov_4_text, fps_warn=fps_warn,
            #        fps_drop_warn_thresh=float(fps_drop_warn_thresh))

            inference_msg.success("Успешно обработан!")

            st.subheader("Обнаруженные объекты")
            st.image(frame_, channels="RGB")
            vero, mater = classification_trash(frame_)
            display_object_info(vero, mater)


            # Function to display object information

    # -------------------------- RTSP ------------------------------
    if input_source == "RTSP":

        rtsp_input = st.sidebar.text_input("IP Address", "rtsp://192.168.0.1")
        if st.sidebar.button("Start tracking"):
            stframe = st.empty()

            st.title("Здесь обрабатывается видео с камер")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass