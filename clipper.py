import os
import glob
import math
import bisect
import argparse
import subprocess
import librosa
import pandas as pd
import numpy as np
import time

TMP_AUDIO_FILE = "__tmp__.wav"
TMP_CONCAT_FILE = "__tmp_concat__.txt"

argparser = argparse.ArgumentParser(description="Extract clips of ETT points from video based on sounds.")
argparser.add_argument("video_file", help="Path to the video file.")
argparser.add_argument(
    "--buffer",
    type=float,
    default=1,
    help="buffer of time to add at the start and end of each clip",
)
argparser.add_argument(
    "--outcsv",
    action="store_true",
    help="whether to output the point timestamps to a CSV file",
)
argparser.add_argument(
    "--outclips",
    action="store_true",
    help="whether to create an output video for each clip and a combined video of all clips",
)
argparser.add_argument(
    "--outname",
    default="clips",
    help="prefix to use for the output video files and data",
)
argparser.add_argument(
    "--orderby",
    default="chrono",
    choices=["chrono", "duration"],
    help="order for outputting the clips by the given metric",
)
argparser.add_argument(
    "--nclips",
    type=int,
    help="number of clips to extract",
)
argparser.add_argument(
    "--starttime",
    type=int,
    default=0,
    help="seconds from the start of the video to start extracting clips",
)
argparser.add_argument(
    "--reverse-clips",
    action="store_true",
    help="reverse the order of the clips in the compiled video, useful for top-10 style videos.",
)
argparser.add_argument(
    "--delta",
    type=float,
    default=0.2,
    help="threshold for onset detection",
)
argparser.add_argument(
    "--min_centroid",
    type=float,
    default=1500,
    help="minimum spectral centroid for onset detection",
)


def main():
    args = argparser.parse_args()

    try:
        preclean(args.outname, args.outclips)  # ffmpeg hangs if output files already exist
        audio_file = extract_audio_from_video(args.video_file)
        timestamps = detect_onsets(
            audio_file, start_time=args.starttime, delta=args.delta, min_centroid=args.min_centroid
        )
        points_df = get_points(timestamps)

        # Do any sorting or limiting of the clips
        if args.orderby == "duration":
            points_df = points_df.sort_values("duration", ascending=False)

        if args.nclips:
            points_df = points_df.head(args.nclips)

        points_df = points_df.round(2)
        points_df.index.name = "point"

        # Print and optionally write out the data
        print()
        print("Extracted points:")
        print(points_df.to_markdown(index=True))
        if args.outcsv:
            points_df.to_csv(f"{args.outname}.csv", index=True)
        print()

        # Create clips and a combined video
        if args.outclips:
            print("Creating video clips ...")
            st = time.time()
            keyframes = get_keyframes(args.video_file)  # can only cut on a keyframe with re-encoding
            segment_files = create_clips(
                args.video_file,
                points_df[["start", "end"]].values,
                keyframes,
                prefix=args.outname,
                buffer=args.buffer,
            )
            if args.reverse_clips:
                segment_files = segment_files[::-1]
            concat_clips(segment_files, f"{args.outname}.mp4")
            print(f"Done! Created {len(segment_files)} video clips in {round(time.time()-st)}s")

    except:
        raise
    finally:
        cleanup()


def detect_onsets(audio_file, start_time=0, delta=0.2, min_centroid=1500, hop_length=32, chunk_size=600):
    sr = librosa.get_samplerate(audio_file)
    total_duration = librosa.get_duration(path=audio_file)

    # Split up due to memory issues on large audio files
    nchunks = math.ceil((total_duration - start_time) / chunk_size)
    chunks = np.array_split(np.arange(start_time, total_duration), nchunks)
    chunks = [(c[0], c[-1] + 1) for c in chunks]

    filtered_onsets = []
    for iter, (offset, end) in enumerate(chunks):
        print(f"Processing chunk {iter+1} of {len(chunks)}, start:{offset}s, end:{end}s ... ", end="", flush=True)
        st = time.time()
        # Perform onset detection
        audio, _ = librosa.load(audio_file, sr=sr, offset=offset, duration=(end - offset), mono=True)
        onset_frames = librosa.onset.onset_detect(y=audio, backtrack=True, delta=delta, hop_length=hop_length)

        # Convert frames to timestamps
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # Spectral Centroid (brightness measure)
        # Filter out low-frequency (floor hit) onsets
        centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]

        for i, onset in enumerate(onset_frames):
            if centroids[onset] > min_centroid:  # Only keep high-frequency (sharp) sounds
                filtered_onsets.append(onset_times[i] + offset)

        print(f"completed in {round(time.time()-st)}s", flush=True)

    return filtered_onsets


def preclean(outname, outclips):
    for f in [TMP_AUDIO_FILE, TMP_CONCAT_FILE]:
        if os.path.exists(f):
            os.remove(f)
    if outclips:
        for f in glob.glob(f"{outname}_*.mp4") + [f"{outname}.mp4"]:
            if os.path.exists(f):
                os.remove(f)


def cleanup():
    for f in [TMP_AUDIO_FILE, TMP_CONCAT_FILE]:
        if os.path.exists(f):
            os.remove(f)


def extract_audio_from_video(video_file):
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
    return TMP_AUDIO_FILE


def create_groups(timestamps, min_size=3, max_time_diff=1):
    groups = []
    if len(timestamps) == 0:
        return groups
    group = [timestamps[0]]
    for i in range(1, len(timestamps)):
        if timestamps[i] - group[-1] <= max_time_diff:
            group.append(timestamps[i])
        else:
            if len(group) >= min_size:
                groups.append(group)
            group = [timestamps[i]]
    groups.append(group)
    return groups


def get_points(timestamps):
    groups = create_groups(timestamps)
    df = pd.DataFrame(columns=["start", "end", "duration"])
    for i, group in enumerate(groups):
        start = group[0]
        end = group[-1]
        duration = end - start
        df.loc[i] = [start, end, duration]

    return df


def concat_clips(files, output_file):
    with open(TMP_CONCAT_FILE, "w") as f:
        for file in files:
            f.write(f"file '{file}'\n")

    # Concatenate the segments
    concat_cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        TMP_CONCAT_FILE,
        "-c",
        "copy",
        output_file,  # Uses copy mode to avoid re-encoding
    ]
    subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_keyframes(video_file):
    """Find the nearest keyframe before or equal to target_start using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-loglevel",
            "error",
            "-select_streams",
            "v",
            "-skip_frame",
            "nokey",
            "-show_frames",
            "-show_entries",
            "frame=pkt_pts_time",
            "-of",
            "csv",
            video_file,
        ],
        capture_output=True,
        text=True,
    )

    keyframe_times = []
    for line in result.stdout.split("\n"):
        if not "frame," in line:
            continue
        time = line.split(",")[1]
        keyframe_times.append(float(time))

    return keyframe_times


def create_clips(video_file, points, keyframes, prefix="clip", buffer=1):
    segment_files = []
    file_type = video_file.split(".")[-1]
    # Without doing some re-encoding, the start of clips get messed up due to not matching a keyframe
    for i, (start, end) in enumerate(points):
        start = max(0, start - buffer)
        end += buffer
        segment_file = f"{prefix}_{i}.{file_type}"

        # Get closest keyframe before start
        # Otherwise, would need to re-encode, which is very slow
        start_keyframe = bisect.bisect_left(keyframes, start) - 1
        if start_keyframe < 0:
            start_keyframe = 0
        start = keyframes[start_keyframe]

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            str(start),  # Fast seek to nearest keyframe
            "-i",
            video_file,
            "-t",
            str(end - start),  # Duration of the clip
            "-c",
            "copy",  # No re-encoding
            segment_file,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        segment_files.append(segment_file)

    return segment_files


if __name__ == "__main__":
    main()
