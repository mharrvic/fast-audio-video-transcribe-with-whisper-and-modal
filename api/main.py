"""
Uses OpenAI's Whisper modal to do speech-to-text transcription
of audio.
"""
import json
import os
import pathlib
from typing import Iterator, Tuple

from modal import Dict, Image, SharedVolume, Stub, asgi_app

from . import audio, config, video

logger = config.get_logger(__name__)
volume = SharedVolume().persist("dataset-cache-vol")

app_image = (
    Image.debian_slim()
    .pip_install(
        "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
        "dacite",
        "jiwer",
        "ffmpeg-python",
        "gql[all]~=3.0.0a5",
        "pandas",
        "loguru==0.6.0",
        "torchaudio==0.12.1",
        "yt-dlp",
    )
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python")
)

stub = Stub(
    "whisper-audio-video-transcriber-api-v2",
    image=app_image,
)

stub.in_progress = Dict()


def get_audio_metadata_path(audio_url: str, title_slug: str) -> pathlib.Path:
    return config.AUDIO_METADATA_DIR / audio_url / f"{title_slug}.json"


def get_transcript_path(title_slug: str) -> pathlib.Path:
    return config.TRANSCRIPTIONS_DIR / f"{title_slug}.json"


@stub.function(
    shared_volumes={config.CACHE_DIR: volume},
    keep_warm=1,
)
@asgi_app()
def fastapi_app():
    from .api import web_app

    return web_app


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[Tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds."""

    import re

    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    logger.info(f"Split {path} into {num_segments} segments")


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    cpu=2,
)
def transcribe_segment(
    start: float,
    end: float,
    audio_filepath: pathlib.Path,
    model: config.ModelSpec,
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(audio_filepath))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(
            model.name, device=device, download_root=config.MODEL_DIR
        )
        result = model.transcribe(f.name, language="en", fp16=use_gpu)  # type: ignore

    logger.info(
        f"Transcribed segment {start:.2f} to {end:.2f} of {end - start:.2f} in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    timeout=900,
)
def transcribe_audio(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model: config.ModelSpec,
):
    segment_gen = split_silences(str(audio_filepath))

    output_text = ""
    output_segments = []
    for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_filepath=audio_filepath, model=model)
    ):
        output_text += result["text"]
        output_segments += result["segments"]

    result = {
        "text": output_text,
        "segments": output_segments,
        "language": "en",
    }

    logger.info(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    timeout=900,
)
def process_audio(src_url: str, title_slug: str, is_video: bool, password: str):
    import dacite
    import whisper
    import yt_dlp
    from modal import container_app

    destination_path = config.RAW_AUDIO_DIR / title_slug

    # Video files are converted to mp3, so we need to pass the mp3 file path.
    audio_filepath = f"{destination_path}.mp3" if is_video else destination_path

    try:
        transcription_path = get_transcript_path(title_slug)

        # pre-download the model to the cache path, because the _download fn is not
        # thread-safe.
        model = config.DEFAULT_MODEL
        whisper._download(whisper._MODELS[model.name], config.MODEL_DIR, False)

        config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        config.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

        if is_video:
            video.download_convert_video_to_audio(
                yt_dlp, src_url, password, destination_path
            )
        else:
            audio.store_original_audio(
                url=src_url,
                destination=destination_path,
            )

        logger.info(
            f"Using the {model.name} model which has {model.params} parameters."
        )

        transcribe_audio.call(
            audio_filepath=audio_filepath,
            result_path=transcription_path,
            model=model,
        )
    except Exception as e:
        logger.exception(e)
        raise dacite.DaciteError("Failed to process audio") from e

    finally:
        del container_app.in_progress[title_slug]
        logger.info(f"Deleting the audio file in '{destination_path}'")
        os.remove(audio_filepath)
        logger.info(f"Deleted the audio file in '{destination_path}'")

    return title_slug
