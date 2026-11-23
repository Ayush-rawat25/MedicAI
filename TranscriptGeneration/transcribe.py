
import json
import time
from datetime import datetime
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
import os


# ==========================================
# CONFIGURATION
# ==========================================

audio_path = "audio/sample.wav"

transcript_path = "output/transcript1.json"
dialogue_path = "output/dialogue.json"

MODEL_SIZE = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

NUM_SPEAKERS = 2


# ==========================================
# TIME MONITORING
# ==========================================

total_start = time.perf_counter()


def log_time(message, start=None):
    current_time = datetime.now().strftime("%H:%M:%S")

    if start is not None:
        elapsed = time.perf_counter() - start
        print(f"[{current_time}] {message}: {elapsed:.2f} seconds")
    else:
        print(f"\n[{current_time}] {message}")


os.makedirs("output", exist_ok=True)


# ==========================================
# STEP 1: LOAD FASTER-WHISPER
# ==========================================

log_time("Loading Faster-Whisper Medium model")

start = time.perf_counter()

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

log_time("Model loaded", start)


# ==========================================
# STEP 2: TRANSCRIPTION
# ==========================================

log_time("Starting transcription")

start = time.perf_counter()

segments, info = model.transcribe(
    audio_path,
    beam_size=5,
    language=None,          # Automatic language detection
    vad_filter=True,
    word_timestamps=True
)

# IMPORTANT: Transcription actually runs
# when the generator is consumed.
segments = list(segments)

transcription_time = time.perf_counter() - start

print("Detected language:", info.language)
print("Language probability:", info.language_probability)
print("Total segments:", len(segments))

transcript = {
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": []
}

for segment in segments:

    transcript["segments"].append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip()
    })


with open(transcript_path, "w", encoding="utf-8") as f:
    json.dump(transcript, f, indent=4, ensure_ascii=False)

log_time("Transcription completed", start)


# ==========================================
# STEP 3: LOAD PYANNOTE
# ==========================================

log_time("Loading pyannote diarization model")

start = time.perf_counter()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
)

pipeline.to(torch.device(DEVICE))

log_time("Diarization model loaded", start)


# ==========================================
# STEP 4: SPEAKER DIARIZATION
# ==========================================

log_time("Starting speaker diarization")

start = time.perf_counter()

diarization = pipeline(
    audio_path,
    num_speakers=NUM_SPEAKERS
)

speaker_turns = []

for turn, _, speaker in (
    diarization.exclusive_speaker_diarization.itertracks(
        yield_label=True
    )
):

    speaker_turns.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

diarization_time = time.perf_counter() - start

print("Detected speaker turns:", len(speaker_turns))

log_time("Diarization completed", start)


# ==========================================
# STEP 5: ALIGN SPEAKERS WITH TRANSCRIPT
# ==========================================

log_time("Starting transcript-speaker alignment")

start = time.perf_counter()

dialogue = []

for segment in segments:

    seg_start = segment.start
    seg_end = segment.end

    best_speaker = "UNKNOWN"
    max_overlap = 0

    for turn in speaker_turns:

        overlap = max(
            0,
            min(seg_end, turn["end"]) -
            max(seg_start, turn["start"])
        )

        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = turn["speaker"]

    dialogue.append({
        "start": seg_start,
        "end": seg_end,
        "speaker": best_speaker,
        "text": segment.text.strip()
    })


with open(dialogue_path, "w", encoding="utf-8") as f:
    json.dump(dialogue, f, indent=4, ensure_ascii=False)

log_time("Speaker alignment completed", start)


# ==========================================
# STEP 6: FINAL RESULTS
# ==========================================

total_time = time.perf_counter() - total_start

print("\n" + "=" * 45)
print("          PROCESSING SUMMARY")
print("=" * 45)

print(f"Transcription time : {transcription_time:.2f} sec")
print(f"Diarization time   : {diarization_time:.2f} sec")
print(f"Total time         : {total_time:.2f} sec")

print(f"\nDetected language  : {info.language}")
print(f"Transcript segments: {len(segments)}")
print(f"Speaker turns      : {len(speaker_turns)}")

print("\nFiles generated:")
print(transcript_path)
print(dialogue_path)

print("\nProcessing completed successfully!")
