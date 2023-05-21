import asyncio
import json
import time
from typing import List, NamedTuple

from fastapi import FastAPI, Request

from . import config
from .audio import coalesce_short_transcript_segments
from .main import get_audio_metadata_path, get_transcript_path, process_audio

logger = config.get_logger(__name__)
web_app = FastAPI()

# A transcription taking > 10 minutes should be exceedingly rare.
MAX_JOB_AGE_SECS = 10 * 60


class InProgressJob(NamedTuple):
    call_id: str
    start_time: int


@web_app.delete("/api/audio/{title_slug}")
async def delete_audio_info(title_slug: str):
    transcription_path = get_transcript_path(title_slug)
    if transcription_path.exists():
        transcription_path.unlink()
        return dict(message="transcription deleted")
    else:
        return dict(message="transcription not found")


@web_app.get("/api/audio/{title_slug}")
async def get_audio_info(title_slug: str):
    transcription_path = get_transcript_path(title_slug)

    if not transcription_path.exists():
        return dict(message="transcription not yet available")

    with open(transcription_path, "r") as f:
        data = json.load(f)

    return dict(
        segments=coalesce_short_transcript_segments(data["segments"]),
    )


@web_app.post("/api/transcribe")
async def transcribe_job(
    src_url: str, title_slug: str, is_video: bool = False, password: str = None
):
    from modal import container_app

    transcription_path = get_transcript_path(title_slug)
    if transcription_path.exists():
        logger.info(
            f"Transcription already exists for '{title_slug}' with URL {src_url}."
        )
        logger.info("Skipping transcription.")
        return {"call_id": "transcription_already_exists"}

    now = int(time.time())
    try:
        inprogress_job = container_app.in_progress[title_slug]
        # NB: runtime type check is to handle present of old `str` values that didn't expire.
        if (
            isinstance(inprogress_job, InProgressJob)
            and (now - inprogress_job.start_time) < MAX_JOB_AGE_SECS
        ):
            existing_call_id = inprogress_job.call_id
            logger.info(
                f"Found existing, unexpired call ID {existing_call_id} for title {title_slug}"
            )
            return {"call_id": existing_call_id}
    except KeyError:
        pass

    call = process_audio.spawn(src_url, title_slug, is_video, password)
    container_app.in_progress[title_slug] = InProgressJob(
        call_id=call.object_id, start_time=now
    )

    return {"call_id": call.object_id}


@web_app.get("/api/status/{call_id}")
async def poll_status(call_id: str):
    from modal._call_graph import InputInfo, InputStatus
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    graph: List[InputInfo] = function_call.get_call_graph()

    try:
        function_call.get(timeout=0.1)
    except TimeoutError:
        pass
    except Exception as exc:
        if exc.args:
            inner_exc = exc.args[0]
            if "HTTPError 403" in inner_exc:
                return dict(error="permission denied on audio download")
        return dict(error="unknown job processing error")

    try:
        map_root = graph[0].children[0].children[0]
    except IndexError:
        return dict(finished=False)

    assert map_root.function_name == "transcribe_audio"

    leaves = map_root.children
    tasks = len(set([leaf.task_id for leaf in leaves]))
    done_segments = len([leaf for leaf in leaves if leaf.status == InputStatus.SUCCESS])
    total_segments = len(leaves)
    finished = map_root.status == InputStatus.SUCCESS

    return dict(
        finished=finished,
        total_segments=total_segments,
        tasks=tasks,
        done_segments=done_segments,
    )
