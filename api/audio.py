import dataclasses
import os
import pathlib
import urllib.request
from typing import NamedTuple, Optional, TypedDict, Union

from . import config

logger = config.get_logger(__name__)
Segment = TypedDict("Segment", {"text": str, "start": float, "end": float})


@dataclasses.dataclass
class AudioMetadata:
    title_slug: str
    audio_url: str


class DownloadResult(NamedTuple):
    data: bytes
    # Helpful to store and transmit when uploading to cloud bucket.
    content_type: str


def download_audio_file(url: str) -> DownloadResult:
    req = urllib.request.Request(
        url,
        data=None,
        # Set a user agent to avoid 403 response from some audio audio servers.
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response:
        return DownloadResult(
            data=response.read(),
            content_type=response.headers["content-type"],
        )


def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def store_original_audio(
    url: str, destination: pathlib.Path, overwrite: bool = False
) -> None:
    if destination.exists():
        if overwrite:
            logger.info(
                f"Audio file exists at {destination} but overwrite option is specified."
            )
        else:
            logger.info(f"Audio file exists at {destination}, skipping download.")
            return

    audio_download_result = download_audio_file(url=url)
    humanized_bytes_str = sizeof_fmt(num=len(audio_download_result.data))
    logger.info(f"Downloaded {humanized_bytes_str} audio from URL.")
    with open(destination, "wb") as f:
        f.write(audio_download_result.data)
    logger.info(f"Stored audio at {destination}.")


def coalesce_short_transcript_segments(
    segments: list[Segment],
) -> list[Segment]:
    """
    Some extracted transcript segments from openai/whisper are really short, like even just one word.
    This function accepts a minimum segment length and combines short segments until the minimum is reached.
    """
    minimum_transcript_len = 200  # About 2 sentences.
    previous = None
    long_enough_segments = []
    for current in segments:
        if previous is None:
            previous = current
        elif len(previous["text"]) < minimum_transcript_len:
            previous = _merge_segments(left=previous, right=current)
        else:
            long_enough_segments.append(previous)
            previous = current
    if previous:
        long_enough_segments.append(previous)
    return long_enough_segments


def _merge_segments(left: Segment, right: Segment) -> Segment:
    return {
        "text": left["text"] + " " + right["text"],
        "start": left["start"],
        "end": right["end"],
    }
