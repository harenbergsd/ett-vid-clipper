"""
Extract onsets from video file for training
"""

import librosa.util
import soundfile as sf
import argparse
import matplotlib.pyplot as plt

from helper import *

parser = argparse.ArgumentParser(description="Extract onsets from video file")
parser.add_argument("video_file", help="Path to the video file")
parser.add_argument("--outdir", default="data/onsets", help="Output directory for onset audio files")
parser.add_argument("--hop-length", type=int, default=32, help="hop length for onset detection")
parser.add_argument("--delta", type=float, default=0.02, help="delta for onset detection")
args = parser.parse_args()


def main():
    times = []
    for (
        sr,
        audio,
        offset,
        end,
    ) in extract_audio_from_video(args.video_file):
        onset_times = get_onsets(audio, sr, args.hop_length, delta=args.delta)

        for i, onset in enumerate(onset_times):
            onset_audio = onset_clip(audio, sr, onset)
            times.append((i, onset))

            # save audio
            filename = os.path.basename(args.video_file)
            outname = os.path.splitext(filename)[0] + f"_{i}"
            sf.write(f"{args.outdir}/{outname}.wav", onset_audio, sr)

            # save extended audio for training augmentation
            start = int((onset - 0.2) * sr)
            end = int((onset + 0.2) * sr)
            sf.write(f"{args.outdir}/extended/{outname}.wav", audio[start:end], sr)

    with open(f"{args.outdir}/onsets.txt", "w") as f:
        for i, onset in times:
            f.write(f"{i} {onset}\n")

    # Plot a subset of the onsets
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(audio, sr=sr, alpha=0.6)
    xmin = 0
    xmax = min(len(audio) // sr, 10)
    ymin = np.min(audio[xmin * sr : xmax * sr])
    ymax = np.max(audio[xmin * sr : xmax * sr])
    plt.vlines(onset_times, ymin=ymin, ymax=ymax, colors="r", linestyle="dashed", label="Onsets")
    plt.title("Waveform with Onsets")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(np.arange(xmin, xmax, step=1))
    plt.yticks(np.arange(ymin, ymax, step=0.1))
    plt.legend()
    plt.savefig(f"{args.outdir}/viz.png")


if __name__ == "__main__":
    main()
