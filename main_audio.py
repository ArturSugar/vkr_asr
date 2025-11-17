# main_audio.py
import sys
from pipeline import process_audio

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python main_audio.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = process_audio(audio_path)

    print("Обработка завершена.")
    print("Длительность аудио:", result["audio_info"]["duration_hms"])
    print("Количество сегментов:", len(result["segments"]))