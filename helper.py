import subprocess
import librosa
import os
import math
import numpy as np
from scipy.signal import find_peaks

TMP_AUDIO_FILE = "__tmp__.wav"


def onset_clip(audio, sr, onset, duration=0.1):
    start = int(onset * sr)
    end = int((onset + duration) * sr)
    return audio[start:end]


def get_onsets(audio, sr, hop_length, delta):
    onset_frames = librosa.onset.onset_detect(y=audio, hop_length=hop_length, delta=delta, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    return onset_times


def features(audio, sr):
    n_fft = min(len(audio), 1024)
    if n_fft <= 0:
        return []
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=32)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    mfcc_max = np.max(mfcc, axis=1)
    delta_max = np.max(delta_mfcc, axis=1)
    delta2_max = np.max(delta2_mfcc, axis=1)

    return np.concatenate([mfcc_max, delta_max, delta2_max], axis=0)


def extract_audio_from_video(video_file, start_time=0, chunk_size=600):
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",  # Suppress ffmpeg
                "-i",
                video_file,  # Input video file.
                "-vn",  # Disable video recording.
                "-acodec",
                "pcm_s16le",  # Use PCM 16-bit little endian codec for WAV.
                "-ar",
                "22050",  # Set the audio sampling rate to 22050 Hz.
                TMP_AUDIO_FILE,  # Output WAV file.
            ]
        )
        # Split up due to memory issues on large audio files
        total_duration = librosa.get_duration(path=TMP_AUDIO_FILE)
        nchunks = math.ceil((total_duration - start_time) / chunk_size)
        chunks = np.array_split(np.arange(start_time, total_duration), nchunks)
        chunks = [(c[0], c[-1] + 1) for c in chunks]

        # Load and yield audio chunks
        sr = librosa.get_samplerate(TMP_AUDIO_FILE)
        for iter, (offset, end) in enumerate(chunks):
            audio, _ = librosa.load(TMP_AUDIO_FILE, sr=sr, offset=offset, duration=(end - offset), mono=True)
            yield sr, audio, offset, end

    except Exception as e:
        raise
    finally:
        if os.path.exists(TMP_AUDIO_FILE):
            os.remove(TMP_AUDIO_FILE)
