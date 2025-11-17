

from pipeline import process_video

if __name__ == "__main__":
    # Укажи путь к своему видеофайлу
    video_file = "/Users/artursaharov/Desktop/снимки экрана/Запись экрана 2025-10-15 в 11.02.45.mov"

    result = process_video(video_file)

    print("Обработка завершена.")
    print("Длительность аудио:", result["audio_info"]["duration_hms"])
    print("Количество сегментов:", len(result["segments"]))