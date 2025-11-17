from typing import Dict, List
from datetime import timedelta, datetime
import os

from moviepy import VideoFileClip
import librosa
import soundfile as sf

from config import AUDIO_TMP
from diarization import run_diarization
from asr_faster import transcribe_by_speaker


def extract_audio_from_video(video_path: str, audio_path: str = AUDIO_TMP) -> str:
    """Извлекает аудиодорожку из видео и сохраняет её в WAV."""
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path


def get_audio_info(audio_path: str) -> Dict:
    """Возвращает базовую информацию об аудио."""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "path": audio_path,
        "sample_rate": sr,
        "duration_sec": round(duration, 2),
        "duration_hms": str(timedelta(seconds=int(duration))),
        "channels": sf.info(audio_path).channels,
        "format": sf.info(audio_path).format,
    }


def save_result_json(source_path: str, segments: List[Dict], audio_info: Dict) -> str:
    """Сохраняет результат в JSON с уникальным именем файла."""
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diarization_{base_name}_{timestamp}.json"

    result = {
        "source_path": source_path,
        "audio_info": audio_info,
        "num_speakers": len({s["speaker"] for s in segments}),
        "segments": segments,
    }

    output_file = os.path.join(os.getcwd(), filename)
    with open(output_file, "w", encoding="utf-8") as f:
        import json
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Результат сохранён в {output_file}")
    return output_file


# Internal helper to avoid code duplication
def _run_pipeline(audio_path: str) -> Dict:
    """Внутренний помощник: диаризация + ASR + инфо об аудио."""
    segments = run_diarization(audio_path)
    segments = transcribe_by_speaker(audio_path, segments)
    audio_info = get_audio_info(audio_path)
    return {
        "segments": segments,
        "audio_info": audio_info,
    }


def process_video(video_path: str) -> Dict:
    """Полный цикл обработки видео: audio → диаризация → ASR → JSON."""
    audio_path = extract_audio_from_video(video_path)
    try:
        result = _run_pipeline(audio_path)
        save_result_json(video_path, result["segments"], result["audio_info"])
        return {
            "video_path": video_path,
            "audio_info": result["audio_info"],
            "segments": result["segments"],
        }
    finally:
        # Удаляем временный аудиофайл, если он был создан
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                # Если вдруг не получилось удалить, просто молча продолжаем
                pass


def process_audio(audio_path: str) -> Dict:
    """Полный цикл обработки уже существующего аудиофайла (без удаления его)."""
    result = _run_pipeline(audio_path)
    save_result_json(audio_path, result["segments"], result["audio_info"])
    return {
        "audio_path": audio_path,
        "audio_info": result["audio_info"],
        "segments": result["segments"],
    }