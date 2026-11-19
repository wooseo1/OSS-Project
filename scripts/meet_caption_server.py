"""Stream microphone transcription to the local Google Meet captions extension."""
import argparse
import asyncio
import json
import queue
import threading
from pathlib import Path


def join_segment_text(segments):
    return " ".join(text for segment in segments
                    if (text := segment.text.strip()))


async def run_server(args):
    import sounddevice as sd
    from faster_whisper import WhisperModel
    from websockets.asyncio.server import serve

    if args.list_devices:
        print(sd.query_devices())
        return

    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")

    model = WhisperModel(str(model_path), device="cpu", compute_type=args.compute_type,
                         cpu_threads=args.cpu_threads, local_files_only=True)
    audio_queue = queue.Queue(maxsize=16)
    connected_clients = set()
    loop = asyncio.get_running_loop()
    stopping = threading.Event()
    chunk_frames = int(args.sample_rate * args.chunk_seconds)

    async def handler(websocket):
        connected_clients.add(websocket)
        await websocket.send(json.dumps({"type": "status", "text": "Whisper 연결됨"},
                                        ensure_ascii=False))
        try:
            await websocket.wait_closed()
        finally:
            connected_clients.discard(websocket)

    async def broadcast(message):
        stale = set()
        for client in connected_clients.copy():
            try:
                await client.send(message)
            except Exception:
                stale.add(client)
        connected_clients.difference_update(stale)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Microphone status: {status}", flush=True)
        block = indata[:, 0].copy()
        try:
            audio_queue.put_nowait(block)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            audio_queue.put_nowait(block)

    def transcribe_audio():
        import numpy as np

        buffered = []
        buffered_frames = 0
        while not stopping.is_set():
            try:
                block = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            buffered.append(block)
            buffered_frames += len(block)
            if buffered_frames < chunk_frames:
                continue

            audio = np.concatenate(buffered)
            buffered.clear()
            buffered_frames = 0
            try:
                segments, _ = model.transcribe(
                    audio,
                    language="ko",
                    task="transcribe",
                    beam_size=1,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                )
                text = join_segment_text(segments)
                if text:
                    payload = json.dumps({"type": "transcript", "text": text},
                                         ensure_ascii=False)
                    loop.call_soon_threadsafe(
                        lambda message=payload: asyncio.create_task(broadcast(message)))
                    print(text, flush=True)
            except Exception as error:
                print(f"Transcription error: {error}", flush=True)

    async with serve(handler, "127.0.0.1", args.port, max_size=2**20):
        stream = sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=args.input_device,
            callback=audio_callback,
        )
        worker = threading.Thread(target=transcribe_audio, daemon=True)
        worker.start()
        try:
            with stream:
                print(f"Listening on microphone; WebSocket: ws://127.0.0.1:{args.port}",
                      flush=True)
                print("Press Ctrl+C to stop. Audio is not saved.", flush=True)
                await asyncio.Future()
        finally:
            stopping.set()
            worker.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local faster-whisper CTranslate2 model folder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.chunk_seconds <= 0 or args.sample_rate <= 0 or args.cpu_threads <= 0:
        parser.error("chunk duration, sample rate, and CPU threads must be positive")
    if not args.list_devices and args.model is None:
        parser.error("--model is required unless --list-devices is used")
    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()