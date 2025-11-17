# Общие настройки проекта диаризации и распознавания

# Путь к локальной модели Pyannote для диаризации
MODEL_DIR = "./models/pyannote_speaker-diarization-3.1"

# Временный файл для аудио, извлечённого из видео
AUDIO_TMP = "./tmp_audio.wav"

# Длительность фрагмента при диаризации (в секундах)
CHUNK_DURATION_SEC = 60

# Настройки модели для faster-whisper.
# Можно указать название стандартной модели ("tiny", "base", "small", "medium", "large-v2")
# или путь/репозиторий кастомной модели, например: "Systran/faster-whisper-small-ru"
ASR_MODEL_NAME = "small"

# Устройство: на Mac без CUDA обычно "cpu"
ASR_DEVICE = "cpu"

# Тип вычислений: для CPU разумно использовать "int8" или "int16"
ASR_COMPUTE_TYPE = "int8"
