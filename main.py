

from pipeline import process_video

if __name__ == "__main__":
    # Укажи путь к своему видеофайлу
    video_file = 

    result = process_video(video_file)

    print("Обработка завершена.")
    print("Длительность аудио:", result["audio_info"]["duration_hms"])
    print("Количество сегментов:", len(result["segments"]))
