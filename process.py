import asyncio
import cv2
import numpy as np
import os
import shutil
from math import ceil
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_inversed_gaussian_mean(frames, visualize_weights=False):
    num_frames = len(frames)
    if num_frames == 0:
        return None

    # Gaussian weights centered and normalized
    x = np.arange(num_frames)
    gaussian_weights = norm.pdf(x, loc=num_frames / 2, scale=num_frames / 5)

    # Invert weights (1 - normalized weight)
    inverted_weights = 1 - gaussian_weights / gaussian_weights.max()

    # Visualize weight distribution
    if visualize_weights:
        plt.plot(x, inverted_weights, label='Inversed Gaussian Weights')
        plt.xlabel('Frame Index')
        plt.ylabel('Weight')
        plt.title('Inversed Gaussian Weight Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Apply inverted weights to frames
    weighted_sum = np.zeros_like(frames[0], dtype=np.float64)
    for frame, weight in zip(frames, inverted_weights):
        weighted_sum += frame * weight

    # Calculate weighted mean frame
    total_weight = inverted_weights.sum()
    mean_frame = (weighted_sum / total_weight).astype(np.uint8)

    return mean_frame


async def save_cropped_contours(video_path, contour_width_threshold=140, contour_height_threshold=140,
                          start_time='1:47', end_time='2:27', cropx=125, cropy=175,
                          video_name='output.mp4', preview_name="{0}_final.jpg"):
    print(f"video_path = {video_path}")
    cap = cv2.VideoCapture(video_path)
    print(f"cap = {cap}")
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")

    # Trimming and cropping setup
    start_min, start_sec = map(int, start_time.split(':'))
    end_min, end_sec = map(int, end_time.split(':'))
    start_frame_time = (start_min * 60 + start_sec) * 1000  # Convert to milliseconds
    end_frame_time = (end_min * 60 + end_sec) * 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_frame_time)
    
    # Get a mean frame across the duration of the video
    total_frames = int((end_frame_time - start_frame_time) * fps / 1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_frame_time)
    frames = [cap.retrieve()[1][cropy:-cropy, cropx:-cropx] for _ in range(total_frames) if cap.grab()]
    mean_frame = calculate_inversed_gaussian_mean(frames)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_frame_time)

    # Setup for processing
    frames_with_contours = []
    debug_frames = []
    cnt = 0
    final_contour = None
    final_time = 0
    final_area = 0
    frame_ = None

    while True:
        ret, frame = cap.read()
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret or current_time > end_frame_time:
            break
        if current_time >= start_frame_time:
            cropped_frame = frame[cropy:-cropy, cropx:-cropx]

            # Compare with mean frame
            diff_frame = cv2.absdiff(mean_frame, cropped_frame)
            gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            max_area = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if w >= contour_width_threshold and h >= contour_height_threshold and area > max_area:
                    largest_contour = contour
                    max_area = area
            if max_area > final_area:
                final_area = max_area
                final_contour = largest_contour
                frame_ = cropped_frame
                final_time = current_time

            # Debug visualization
            debug_frame = cropped_frame.copy()
            if largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frames_with_contours.append(debug_frame)

            debug_frames.append(debug_frame)
        cnt += 1
    x, y, w, h = cv2.boundingRect(final_contour)
    output_directory = os.getcwd()  # Получить текущую рабочую директорию
    output_picture_path = os.path.join(output_directory, preview_name.format(video_path.split('/')[-1].split('.')[0]))
    cv2.rectangle(frame_, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_picture_path, frame_) # [y:y+h, x:x+w])

    # Указать путь к сохраняемому видеофайлу
    height, width, _ = debug_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_directory = os.getcwd()  # Получить текущую рабочую директорию
    output_video_path = os.path.join(output_directory, video_name)

    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    # Сохранить каждый кадр в видеофайл
    for frame in debug_frames:
        output_video.write(frame)

    # Закрыть видеофайл
    output_video.release()

    cap.release()

    return output_video_path if os.path.isfile(output_video_path) else None, output_picture_path if os.path.isfile(output_picture_path) else None, final_time

# if __name__ == "__main__":
#     import os
#     # a = os.listdir()
#     # packs = [a[i:i+100] for i in range(0, len(a), 100)] # len is 10
#     print(asyncio.run(save_cropped_contours('K260BM977_09_18_2023 13_14_06.mp4')))
#     print("Finished!!!!!!!! 2")


