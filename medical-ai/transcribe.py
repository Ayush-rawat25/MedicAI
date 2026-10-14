import whisper
import json

model = whisper.load_model("base")

result = model.transcribe(
    "audio/sample.wav",
    language="en"
)

output = {
    "text": result["text"].strip(),
    "segments": [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        }
        for segment in result["segments"]
    ]
}

with open(
    "output/transcript.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False
    )

print("Transcription completed.")
print(output["text"])