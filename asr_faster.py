"""
Модуль распознавания речи (ASR) на базе faster-whisper.
"""

from typing import List, Dict
import os

import librosa
import soundfile as sf
from faster_whisper import WhisperModel
from tqdm import tqdm

from config import ASR_MODEL_NAME, ASR_DEVICE, ASR_COMPUTE_TYPE


def load_asr_model() -> WhisperModel:
    """Создаёт и возвращает экземпляр модели faster-whisper."""
    print(f"Загрузка faster-whisper модели: {ASR_MODEL_NAME} ({ASR_DEVICE}, {ASR_COMPUTE_TYPE})")
    model = WhisperModel(
        ASR_MODEL_NAME,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )
    return model


def transcribe_by_speaker(
    audio_path: str,
    segments: List[Dict],
    language: str = "ru",
) -> List[Dict]:
    """Распознаёт речь по спикерам и добавляет текст в сегменты.

    Для каждого спикера берётся общий промежуток времени (от первой до последней реплики),
    аудио вырезается одной порцией, прогоняется через faster-whisper.
    Полученный текст записывается во все сегменты этого спикера.
    """
    if not segments:
        return segments

    model = load_asr_model()

    grouped_segments: Dict[str, List[Dict]] = {}
    for seg in segments:
        grouped_segments.setdefault(seg["speaker"], []).append(seg)

    for speaker, segs in tqdm(grouped_segments.items(), desc="Распознавание по спикерам"):
        segs = sorted(segs, key=lambda x: x["start"])
        start_time = segs[0]["start"]
        end_time = segs[-1]["end"]
        duration = end_time - start_time

        # Вырезаем аудио для конкретного спикера
        y_spk, sr_spk = librosa.load(
            audio_path,
            sr=16000,
            offset=start_time,
            duration=duration,
        )
        tmp_speaker_path = f"./tmp_spk_{speaker}.wav"
        sf.write(tmp_speaker_path, y_spk, sr_spk)

        # faster-whisper возвращает генератор сегментов и объект info
        asr_segments, info = model.transcribe(
            tmp_speaker_path,
            language=language,
            beam_size=5,
        )

        # Собираем текст в одну строку
        full_text_parts = [seg.text for seg in asr_segments]
        full_text = " ".join(full_text_parts).strip()

        os.remove(tmp_speaker_path)

        for seg in segs:
            seg["text"] = full_text

    return segments