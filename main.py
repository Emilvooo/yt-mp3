import asyncio
import json
import os
import shutil
import tempfile
import re
import uuid
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

YT_URL_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)[\w-]{11}(\b|$)"
)
PROGRESS_RE = re.compile(r"\[download\]\s+([\d.]+)%")
OUT_TIME_RE = re.compile(r"out_time_ms=(\d+)")
TASK_TTL = 300
YTDLP_AUDIO_FORMAT = "bestaudio/best"
YTDLP_CONCURRENT_FRAGMENTS = "4"

tasks: dict[str, dict] = {}


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots():
    return "User-agent: *\nAllow: /\nDisallow: /download\nDisallow: /progress/\nDisallow: /file/\nDisallow: /info\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


async def _fetch_video_info(url: str) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp", "-j", "--no-playlist", url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Could not fetch video info")
    return json.loads(stdout.decode())


def _pick_thumbnail_url(data: dict) -> str:
    thumbnail = data.get("thumbnail") or ""
    for thumb in reversed(data.get("thumbnails", [])):
        if thumb.get("url"):
            return thumb["url"]
    return thumbnail


async def _download_thumbnail(url: str, tmp_dir: str) -> Path | None:
    if not url:
        return None
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    thumb_path = Path(tmp_dir) / f"thumb{suffix}"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            res = await client.get(url)
        if res.status_code >= 400 or not res.content:
            return None
        thumb_path.write_bytes(res.content)
        return thumb_path
    except Exception:
        return None


async def _download_thumbnail_fallback(url: str, tmp_dir: str) -> Path | None:
    output_template = os.path.join(tmp_dir, "thumb.%(ext)s")
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "--skip-download",
        "--write-thumbnail",
        "--no-playlist",
        "-o", output_template,
        url,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    if proc.returncode != 0:
        return None
    files = [f for f in Path(tmp_dir).iterdir() if f.is_file() and f.name.startswith("thumb.")]
    if not files:
        return None
    return files[0]


async def _get_thumbnail_path(url: str, tmp_dir: str) -> Path | None:
    thumb_path = await _download_thumbnail(url, tmp_dir)
    if thumb_path:
        return thumb_path
    return await _download_thumbnail_fallback(url, tmp_dir)


def _effective_duration_ms(duration_ms: int, trim_start, trim_end) -> int:
    effective_duration_ms = duration_ms
    if trim_start is not None and trim_end is not None:
        effective_duration_ms = int(max(trim_end - trim_start, 0) * 1_000_000)
    elif trim_end is not None:
        effective_duration_ms = int(max(trim_end, 0) * 1_000_000)
    elif trim_start is not None and duration_ms:
        effective_duration_ms = max(duration_ms - int(trim_start * 1_000_000), 0)
    return effective_duration_ms


@app.get("/info")
async def info(url: str = ""):
    url = url.strip()
    if not url or not YT_URL_PATTERN.match(url):
        return JSONResponse({"error": "Invalid YouTube URL"}, status_code=400)
    try:
        data = await _fetch_video_info(url)
    except Exception:
        return JSONResponse({"error": "Could not fetch video info"}, status_code=400)
    return JSONResponse({
        "title": data.get("title", ""),
        "uploader": data.get("uploader") or data.get("channel", ""),
        "duration": int(float(data.get("duration", 0))),
        "thumbnail": _pick_thumbnail_url(data),
    })


VALID_FORMATS = {"mp3", "aac", "ogg"}
VALID_BITRATES = {128, 192, 320}
FORMAT_MIME = {
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
}


@app.post("/download")
async def download(request: Request):
    body = await request.json()
    url = body.get("url", "").strip()

    if not url or not YT_URL_PATTERN.match(url):
        return JSONResponse({"error": "Invalid YouTube URL"}, status_code=400)

    fmt = body.get("format", "mp3").lower()
    if fmt not in VALID_FORMATS:
        fmt = "mp3"
    bitrate = body.get("bitrate", 192)
    if bitrate not in VALID_BITRATES:
        bitrate = 192
    trim_start = body.get("start")
    trim_end = body.get("end")

    task_id = uuid.uuid4().hex[:12]
    tasks[task_id] = {
        "status": "starting",
        "progress": 0,
        "stage": "Starting...",
        "error": None,
        "filename": None,
        "title": None,
        "artist": None,
        "tmp_dir": None,
        "output_path": None,
        "output_format": fmt,
        "created": time.time(),
    }

    asyncio.create_task(_run_download(task_id, url, fmt, bitrate, trim_start, trim_end))
    return JSONResponse({"task_id": task_id})


METADATA_PROMPT = """Given this YouTube video info, extract the artist name and clean song/track title.

YouTube title: "{title}"
Uploader: "{uploader}"

Rules:
- Remove noise like (Official Video), (Official Audio), (Lyrics), (Music Video), [Official Music Video], (4K Remaster), (Remastered XXXX), (Official Visualizer), (Audio), [HD], [HQ], etc.
- Keep meaningful qualifiers like (Live in ...), (feat. ...), (Remix), (Acoustic), (Deluxe), etc.
- The artist is usually before " - " in the title, or matches the uploader/channel name
- For compilations/mixes, the artist is usually the uploader
- Return ONLY valid JSON: {{"artist": "...", "title": "..."}}"""


async def _parse_metadata(title: str, uploader: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"artist": "", "title": title}

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "google/gemini-3-flash-preview",
                    "messages": [{"role": "user", "content": METADATA_PROMPT.format(title=title, uploader=uploader)}],
                },
                timeout=10,
            )
        content = res.json()["choices"][0]["message"]["content"]
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(content)
        return {"artist": parsed.get("artist", ""), "title": parsed.get("title", title)}
    except Exception:
        return {"artist": "", "title": title}


def _build_ffmpeg_cmd(
    audio_input: str,
    out_path: Path,
    fmt: str,
    bitrate: int,
    title: str,
    artist: str,
    thumb_path: Path | None,
    trim_start,
    trim_end,
    allow_input_seek: bool,
) -> list[str]:
    cmd = ["ffmpeg", "-y"]
    if allow_input_seek and trim_start is not None:
        cmd.extend(["-ss", str(trim_start)])
    cmd.extend(["-i", audio_input])
    if thumb_path and fmt == "mp3":
        cmd.extend(["-i", str(thumb_path)])
    if not allow_input_seek and trim_start is not None:
        cmd.extend(["-ss", str(trim_start)])
    if trim_end is not None:
        if trim_start is not None:
            cmd.extend(["-t", str(max(trim_end - trim_start, 0))])
        else:
            cmd.extend(["-to", str(trim_end)])
    if fmt == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-b:a", f"{bitrate}k"])
    elif fmt == "aac":
        cmd.extend(["-codec:a", "aac", "-b:a", f"{bitrate}k"])
    elif fmt == "ogg":
        cmd.extend(["-codec:a", "libvorbis", "-b:a", f"{bitrate}k"])
    cmd.extend(["-metadata", f"title={title}", "-metadata", f"artist={artist}", "-metadata", "comment="])
    if thumb_path and fmt == "mp3":
        cmd.extend([
            "-map", "0:a", "-map", "1:0",
            "-c:v", "png",
            "-id3v2_version", "3",
            "-metadata:s:v", "title=Album cover",
            "-metadata:s:v", "comment=Cover (front)",
        ])
    cmd.extend(["-progress", "pipe:1", str(out_path)])
    return cmd


async def _drain_lines(stream, task: dict | None = None, parse_progress: bool = False) -> str:
    if stream is None:
        return ""
    tail: list[str] = []
    async for line in stream:
        text = line.decode(errors="replace").strip()
        if parse_progress and task:
            match = PROGRESS_RE.search(text)
            if match:
                task["progress"] = max(task["progress"], float(match.group(1)))
        if text:
            tail.append(text)
            if len(tail) > 25:
                tail.pop(0)
    return "\n".join(tail)


async def _drain_ffmpeg_progress(stream, task: dict, effective_duration_ms: int) -> str:
    if stream is None:
        return ""
    tail: list[str] = []
    async for line in stream:
        text = line.decode(errors="replace").strip()
        match = OUT_TIME_RE.search(text)
        if match and effective_duration_ms > 0:
            current_ms = int(match.group(1))
            task["progress"] = max(task["progress"], min(current_ms / effective_duration_ms * 100, 99))
        if text:
            tail.append(text)
            if len(tail) > 25:
                tail.pop(0)
    return "\n".join(tail)


async def _probe_duration_ms(audio_path: Path) -> int:
    probe = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(audio_path),
        stdout=asyncio.subprocess.PIPE,
    )
    probe_out, _ = await probe.communicate()
    try:
        return int(float(probe_out.decode().strip()) * 1_000_000)
    except (ValueError, AttributeError):
        return 0


async def _download_audio_file(url: str, tmp_dir: str, task: dict) -> tuple[Path | None, str | None]:
    output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")
    dl_proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "-f", YTDLP_AUDIO_FORMAT,
        "-N", YTDLP_CONCURRENT_FRAGMENTS,
        "--no-playlist",
        "--newline",
        "-o", output_template,
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_task = asyncio.create_task(_drain_lines(dl_proc.stdout, task, True))
    stderr_task = asyncio.create_task(_drain_lines(dl_proc.stderr, task, True))
    await dl_proc.wait()
    stdout_tail, stderr_tail = await asyncio.gather(stdout_task, stderr_task)
    if dl_proc.returncode != 0:
        return None, stderr_tail or stdout_tail or "Download failed"
    audio_files = [
        f for f in Path(tmp_dir).iterdir()
        if f.is_file()
        and f.suffix.lower() in {".opus", ".webm", ".m4a", ".ogg", ".mp3", ".aac", ".wav", ".flac"}
        and not f.name.startswith("output.")
        and not f.name.startswith("thumb.")
    ]
    if not audio_files:
        return None, "No audio file found"
    return audio_files[0], None


async def _pump_stream(src: asyncio.StreamReader | None, dst: asyncio.StreamWriter | None):
    if src is None or dst is None:
        return
    try:
        while True:
            chunk = await src.read(64 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            dst.write_eof()
        except Exception:
            try:
                dst.close()
            except Exception:
                pass


async def _convert_from_file(
    audio_path: Path,
    out_path: Path,
    fmt: str,
    bitrate: int,
    title: str,
    artist: str,
    thumb_path: Path | None,
    trim_start,
    trim_end,
    task: dict,
    effective_duration_ms: int,
) -> str | None:
    ffmpeg_cmd = _build_ffmpeg_cmd(
        str(audio_path), out_path, fmt, bitrate, title, artist, thumb_path, trim_start, trim_end, True
    )
    ff_proc = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_task = asyncio.create_task(_drain_ffmpeg_progress(ff_proc.stdout, task, effective_duration_ms))
    stderr_task = asyncio.create_task(_drain_lines(ff_proc.stderr))
    await ff_proc.wait()
    _, stderr_tail = await asyncio.gather(stdout_task, stderr_task)
    if ff_proc.returncode != 0 or not out_path.exists():
        return stderr_tail or "Conversion failed"
    return None


async def _convert_from_stream(
    url: str,
    out_path: Path,
    fmt: str,
    bitrate: int,
    title: str,
    artist: str,
    thumb_path: Path | None,
    task: dict,
    effective_duration_ms: int,
) -> str | None:
    yt_proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "-f", YTDLP_AUDIO_FORMAT,
        "-N", YTDLP_CONCURRENT_FRAGMENTS,
        "--no-playlist",
        "--newline",
        "-o", "-",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ffmpeg_cmd = _build_ffmpeg_cmd("pipe:0", out_path, fmt, bitrate, title, artist, thumb_path, None, None, False)
    ff_proc = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pump_task = asyncio.create_task(_pump_stream(yt_proc.stdout, ff_proc.stdin))
    yt_log_task = asyncio.create_task(_drain_lines(yt_proc.stderr))
    ff_out_task = asyncio.create_task(_drain_ffmpeg_progress(ff_proc.stdout, task, effective_duration_ms))
    ff_err_task = asyncio.create_task(_drain_lines(ff_proc.stderr))
    yt_code = await yt_proc.wait()
    await pump_task
    ff_code = await ff_proc.wait()
    yt_tail, _, ff_err = await asyncio.gather(yt_log_task, ff_out_task, ff_err_task)
    if yt_code != 0:
        return yt_tail or "Download failed"
    if ff_code != 0 or not out_path.exists():
        return ff_err or "Conversion failed"
    return None


async def _run_download(task_id: str, url: str, fmt: str = "mp3", bitrate: int = 192, trim_start=None, trim_end=None):
    task = tasks[task_id]
    tmp_dir = tempfile.mkdtemp()
    task["tmp_dir"] = tmp_dir
    try:
        task["status"] = "downloading"
        task["stage"] = "Fetching info..."
        info = await _fetch_video_info(url)
        raw_title = info.get("title") or "audio"
        uploader = info.get("uploader") or info.get("channel", "")
        duration_ms = int(float(info.get("duration", 0)) * 1_000_000)
        thumb_url = _pick_thumbnail_url(info)
        task["stage"] = "Parsing metadata..."
        parse_task = asyncio.create_task(_parse_metadata(raw_title, uploader))
        thumb_task = None
        if fmt == "mp3" and thumb_url:
            thumb_task = asyncio.create_task(_get_thumbnail_path(thumb_url, tmp_dir))
        if thumb_task:
            parsed, thumb_path = await asyncio.gather(parse_task, thumb_task)
        else:
            parsed = await parse_task
            thumb_path = None
        title = parsed.get("title", raw_title)
        artist = parsed.get("artist", "")
        effective_duration_ms = _effective_duration_ms(duration_ms, trim_start, trim_end)
        out_path = Path(tmp_dir) / f"output.{fmt}"
        if trim_start is None and trim_end is None:
            task["status"] = "converting"
            task["stage"] = "Downloading + converting..."
            task["progress"] = 0
            stream_error = await _convert_from_stream(
                url, out_path, fmt, bitrate, title, artist, thumb_path, task, effective_duration_ms
            )
            if stream_error:
                out_path.unlink(missing_ok=True)
                task["status"] = "downloading"
                task["stage"] = "Retrying download..."
                task["progress"] = 0
                audio_path, download_error = await _download_audio_file(url, tmp_dir, task)
                if download_error or audio_path is None:
                    task["status"] = "error"
                    task["error"] = f"Download failed: {download_error}"
                    return
                if not duration_ms:
                    duration_ms = await _probe_duration_ms(audio_path)
                    effective_duration_ms = _effective_duration_ms(duration_ms, trim_start, trim_end)
                task["status"] = "converting"
                task["stage"] = "Converting..."
                task["progress"] = 0
                convert_error = await _convert_from_file(
                    audio_path, out_path, fmt, bitrate, title, artist, thumb_path, trim_start, trim_end, task,
                    effective_duration_ms,
                )
                audio_path.unlink(missing_ok=True)
                if convert_error:
                    task["status"] = "error"
                    task["error"] = f"Conversion failed: {convert_error}"
                    return
        else:
            task["status"] = "downloading"
            task["stage"] = "Downloading..."
            task["progress"] = 0
            audio_path, download_error = await _download_audio_file(url, tmp_dir, task)
            if download_error or audio_path is None:
                task["status"] = "error"
                task["error"] = f"Download failed: {download_error}"
                return
            if not duration_ms:
                duration_ms = await _probe_duration_ms(audio_path)
                effective_duration_ms = _effective_duration_ms(duration_ms, trim_start, trim_end)
            task["status"] = "converting"
            task["stage"] = "Converting..."
            task["progress"] = 0
            convert_error = await _convert_from_file(
                audio_path, out_path, fmt, bitrate, title, artist, thumb_path, trim_start, trim_end, task,
                effective_duration_ms,
            )
            audio_path.unlink(missing_ok=True)
            if convert_error:
                task["status"] = "error"
                task["error"] = f"Conversion failed: {convert_error}"
                return
        filename = f"{artist} - {title}.{fmt}" if artist else f"{title}.{fmt}"
        filename_safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
        task["status"] = "complete"
        task["progress"] = 100
        task["stage"] = "Done"
        task["output_path"] = str(out_path)
        task["filename"] = filename_safe
        task["title"] = title
        task["artist"] = artist
    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)


@app.get("/progress/{task_id}")
async def progress(task_id: str):
    if task_id not in tasks:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    async def event_stream():
        while True:
            task = tasks.get(task_id)
            if not task:
                yield f"event: error\ndata: {json.dumps({'error': 'Task expired'})}\n\n"
                break

            if task["status"] == "error":
                yield f"event: error\ndata: {json.dumps({'error': task['error']})}\n\n"
                break

            if task["status"] == "complete":
                yield f"event: progress\ndata: {json.dumps({'percent': 100, 'stage': 'Done'})}\n\n"
                yield f"event: complete\ndata: {json.dumps({'filename': task['filename'], 'title': task['title'], 'artist': task['artist']})}\n\n"
                break

            yield f"event: progress\ndata: {json.dumps({'percent': task['progress'], 'stage': task['stage']})}\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/file/{task_id}")
async def file(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "complete":
        return JSONResponse({"error": "File not available"}, status_code=404)

    out_path = task["output_path"]
    filename = task["filename"]
    tmp_dir = task["tmp_dir"]
    mime = FORMAT_MIME.get(task.get("output_format", "mp3"), "audio/mpeg")

    async def cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tasks.pop(task_id, None)

    return FileResponse(
        path=out_path,
        filename=filename,
        media_type=mime,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
        },
        background=BackgroundTask(cleanup),
    )


@app.on_event("startup")
async def startup():
    asyncio.create_task(_cleanup_stale_tasks())


async def _cleanup_stale_tasks():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = [tid for tid, t in tasks.items() if now - t["created"] > TASK_TTL]
        for tid in stale:
            t = tasks.pop(tid, None)
            if t and t.get("tmp_dir"):
                shutil.rmtree(t["tmp_dir"], ignore_errors=True)
