import asyncio

import cv2
import io
import streamlit as st
import plotly.express as px

from process import save_cropped_contours

fkko = {
    "wood": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ЛЕСНОГО ХОЗЯЙСТВА|ОТХОДЫ РЫБОВОДСТВА И РЫБОЛОВСТВА|Отходы при уборке урожая зерновых и зернобобовых культур|зелень древесная|ОТХОДЫ ПРОИЗВОДСТВА БУМАГИ И БУМАЖНЫХ ИЗДЕЛИЙ".lower(),
    "stone": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ|ОТХОДЫ ДОБЫЧИ ПОЛЕЗНЫХ ИСКОПАЕМЫХ".lower(),
    "concrete": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ|ОТХОДЫ ДОБЫЧИ ПОЛЕЗНЫХ ИСКОПАЕМЫХ".lower(),
    "ground": "ОТХОДЫ СЕЛЬСКОГО ХОЗЯЙСТВА|ОТХОДЫ ПРОИЗВОДСТВА ПИЩЕВЫХ ПРОДУКТОВ, НАПИТКОВ, ТАБАЧНЫХ ИЗДЕЛИЙ|ОТХОДЫ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ".lower()
}

def display_object_info(obj):
    # Display image
    st.image(obj["image"], caption=f"Detected at {obj['time']}", use_column_width=True)

    # Display additional information
    materials_data = obj['material']
    # Отображение графика
    fig = px.bar(x=list(materials_data.keys()), y=list(materials_data.values()),
                 labels={'x': 'Material', 'y': 'Probability'}, title='Material Probabilities')
    st.plotly_chart(fig)

    # Обработка события при клике на колонку
    sorted_attr = sorted(materials_data.items(), key=lambda item: item[1], reverse=True)
    selected_material = st.selectbox('Наиболее вероятный материал отходов:', sorted_attr[0])
    related_categories = fkko[selected_material].split('|')[0:3]

    # Отображение связанных категорий
    st.write(f"Соответствующие категории отходов по ФККО (Федеральный классификационный каталог отходов):")
    for category in related_categories:
        st.write(f"- {category}")
    st.write("---")

# Function to simulate getting detected objects
def get_detected_objects():
    # Replace this with your logic to get detected objects
    # For now, just returning a sample list
    return [
        {"image": "src/image/scr1.png", "time": "0:05", "material": {
    "wood": 0.7,
    "stone": 0.1,
    "concrete": 0.1,
    "ground": 0.1
    }},
        {"image": "src/image/scr2.png", "time": "0:15", "material": {
    "wood": 0.2,
    "stone": 0.3,
    "concrete": 0.4,
    "ground": 0.1
    }},
        # Add more objects as needed
    ]

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
        video_bytes = st.sidebar.file_uploader("Выберите входное видео", type=["mp4", "avi"], accept_multiple_files=False)
        if video_bytes is not None:
            # Сохраняем видео на диск
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as file:
                file.write(video_bytes.read())
            video_capture = cv2.VideoCapture(video_path)
            print("video_capture", video_capture)

        if st.sidebar.button("Начало обработки"):
            output_video_path, preview, time = asyncio.run(save_cropped_contours(
                io.BytesIO(video_bytes.read())))
            print(f"!!!!!!!!!!! {output_video_path, preview, time}")
            st.video(video_bytes, start_time=0)

            # Обработка нейросетью
            # detect(source=video.name, stframe=stframe, kpi1_text=kpi1_text, kpi2_text=kpi2_text, kpi3_text=kpi3_text,
            #        js1_text=js1_text, js2_text=js2_text, js3_text=js3_text, conf_thres=float(conf_thres), nosave=nosave,
            #        display_labels=display_labels, conf_thres_drift=float(conf_thres_drift),
            #        save_poor_frame__=save_poor_frame__, inf_ov_1_text=inf_ov_1_text, inf_ov_2_text=inf_ov_2_text,
            #        inf_ov_3_text=inf_ov_3_text, inf_ov_4_text=inf_ov_4_text, fps_warn=fps_warn,
            #        fps_drop_warn_thresh=float(fps_drop_warn_thresh))

            inference_msg.success("Успешно обработан!")


            st.subheader("Обнаруженные объекты")
            detected_objects = get_detected_objects()  # Replace with your logic to get detected objects
            for obj in detected_objects:
                display_object_info(obj)

            # Function to display object information

    # -------------------------- RTSP ------------------------------
    if input_source == "RTSP":

        rtsp_input = st.sidebar.text_input("IP Address", "rtsp://192.168.0.1")
        if st.sidebar.button("Start tracking"):
            stframe = st.empty()

            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results

            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0")
                fps_warn = st.empty()

            with kpi2:
                st.markdown("**Detected objects in curret Frame**")
                kpi2_text = st.markdown("0")

            with kpi3:
                st.markdown("**Total Detected objects**")
                kpi3_text = st.markdown("0")

            # Updating System stats

            with js1:
                st.markdown("**Memory usage**")
                js1_text = st.markdown("0")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown("**Poor performing classes (Conf < {0})**".format(conf_thres_drift))
                inf_ov_1_text = st.markdown("0")

            with inf_ov_2:
                st.markdown("**No. of poor peforming frames**")
                inf_ov_2_text = st.markdown("0")

            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0")

            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0")

            detect(source=rtsp_input, stframe=stframe, kpi1_text=kpi1_text, kpi2_text=kpi2_text, kpi3_text=kpi3_text,
                   js1_text=js1_text, js2_text=js2_text, js3_text=js3_text, conf_thres=float(conf_thres), nosave=nosave,
                   display_labels=display_labels, conf_thres_drift=float(conf_thres_drift),
                   save_poor_frame__=save_poor_frame__, inf_ov_1_text=inf_ov_1_text, inf_ov_2_text=inf_ov_2_text,
                   inf_ov_3_text=inf_ov_3_text, inf_ov_4_text=inf_ov_4_text, fps_warn=fps_warn,
                   fps_drop_warn_thresh=float(fps_drop_warn_thresh))

    # torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass