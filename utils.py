import json
import wave
import os
import librosa

import numpy as np

from struct import pack
from fastapi import UploadFile
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def create_model(vector_length: int=128):
    """5 hidden dense layers from 256 units to 64, not the best model, but not bad."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model

def extract_feature(file_name: str, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    start = kwargs.get("start")
    duration = kwargs.get("duration")

    X, sample_rate = librosa.core.load(file_name, offset=start, duration=duration)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

def get_gender(file: str, start: float, duration: float) -> str:
    """
    Fuction for return gender
    Params:
        file: path to wav audio file
        start: wait start secons before start read
        duration: read duration seconds
    """
    model = create_model()
    # load the saved/trained weights
    model.load_weights("model.h5")
    # extract features and reshape it
    features = extract_feature(file, mel=True, start=start, duration=duration).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    return gender

# I want use this model, but my laptop can't load this model. Sorry..
# model = Model(model_path="./vosk-model-ru-0.42")
model = Model(lang="ru")
async def get_dialog_as_text(file: UploadFile) -> list[dict]:
    # Model for get text from audio
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
        data = wf.readframes(100)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "text" in result and result["text"]:
                duration = 0
                for res in result["result"]:
                    duration += res['end'] - res['start'] 
                gender = get_gender(temp_wav, res['start'], duration)
                dialog.append({
                    "source": current_speaker,
                    "text": result["text"],
                    "duration": duration,
                    "gender": gender,
                })
                total_duration[current_speaker] += duration

                current_speaker = "transmitter" if current_speaker == "receiver" else "receiver"

    os.remove(temp_wav)

    return {
        "dialog": dialog,
        "total_duration": total_duration,
    }
