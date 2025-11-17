

"""
Модуль диаризации аудио (определение спикеров и таймкодов)
на базе Pyannote. Использует прямой waveform input,
без внутренних декодеров.
"""

from typing import List, Dict
import gc

import librosa
import torch
from pyannote.audio import Pipeline
from tqdm import tqdm

from config import MODEL_DIR, CHUNK_DURATION_SEC


def run_diarization(audio_path: str, chunk_duration_sec: int = CHUNK_DURATION_SEC) -> List[Dict]:
    """
    Выполняет диаризацию аудиофайла и возвращает список сегментов:

    [
        {
            "speaker": str,
            "start": float,
            "end": float,
            "duration": float,
            "chunk_id": int,
        },
        ...
    ]
    """

    # Загружаем модель Pyannote
    pipeline = Pipeline.from_pretrained(MODEL_DIR)

    # Загружаем аудио целиком
    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)
    print(f"Длительность аудио: {total_duration / 60:.2f} мин")

    chunk_samples = int(chunk_duration_sec * sr)
    all_segments: List[Dict] = []

    # Идем по кускам аудио
    for i, start_sample in enumerate(
        tqdm(range(0, len(y), chunk_samples), desc="Диаризация по фрагментам")
    ):
        end_sample = min(start_sample + chunk_samples, len(y))
        chunk_y = y[start_sample:end_sample]

        # Передаем waveform напрямую
        waveform = torch.from_numpy(chunk_y).unsqueeze(0)
        file_obj = {"waveform": waveform, "sample_rate": sr}

        diarization = pipeline(file_obj)

        # Определяем формат выходных данных Pyannote
        if hasattr(diarization, "speaker_diarization"):
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, "annotation"):
            annotation = diarization.annotation
        else:
            raise TypeError(f"Unsupported diarization output type: {type(diarization)}")

        # Извлекаем интервалы
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start_time = turn.start + start_sample / sr
            end_time = turn.end + start_sample / sr
            all_segments.append(
                {
                    "speaker": speaker,
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "duration": round(turn.end - turn.start, 2),
                    "chunk_id": i,
                }
            )

        gc.collect()

    # Сортируем сегменты по времени
    all_segments.sort(key=lambda s: s["start"])
    print(f"Найдено сегментов: {len(all_segments)}")

    return all_segments