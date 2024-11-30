import json
import wave
import os

from fastapi import UploadFile
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer


model = Model(lang="en-us")

async def get_dialog_as_text(file: UploadFile) -> list[dict]:
    #Convert mp3 в wav
    audio = AudioSegment.from_file(file.file, format="mp3")
    temp_wav = "temp.wav"
    audio.export(temp_wav, format="wav")

    wf = wave.open(temp_wav, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    dialog = []
    total_duration = {"receiver": 0, "transmitter": 0}
    current_speaker = "receiver"

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "text" in result and result["text"]:
                duration = 0
                for res in result["result"]:
                    duration += res['end'] - res['start'] 
                dialog.append({
                    "source": current_speaker,
                    "text": result["text"],
                    "duration": duration,
                })
                total_duration[current_speaker] += duration

                current_speaker = "transmitter" if current_speaker == "receiver" else "receiver"

    os.remove(temp_wav)

    return {
        "dialog": dialog,
        "total_duration": total_duration,
    }
