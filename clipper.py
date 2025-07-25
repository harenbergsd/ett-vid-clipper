import os
import av
import glob
import pickle
import bisect
import argparse
import subprocess
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import time
import datetime
from helper import *

TMP_CONCAT_FILE = "__tmp_concat__.txt"

argparser = argparse.ArgumentParser(description="Extract clips of ETT points from video based on sounds.")
argparser.add_argument("video_file", help="Path to the video file.")
argparser.add_argument(
    "--buffer",
    type=float,
    default=1.5,
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
    choices=["chrono", "shots", "duration"],
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
    "--max-time-diff",
    type=float,
    default=2.5,
    help="maximum time difference between subsequent onsets to group them into the same point (clip)",
)
argparser.add_argument(
    "--delta",
    type=float,
    default=0.02,
    help="delta for the onset detection",
)
argparser.add_argument(
    "--skip-clips",
    nargs="+",
    type=int,
    default=[],
    help="list of clip indices to skip (e.g., 0 1 2 will skip the first three clips)",
)
argparser.add_argument(
    "--reverse-clips",
    action="store_true",
    help="reverse the order of the clips in the compiled video, useful for top-10 style videos.",
)


def main():
    args = argparser.parse_args()
    run_clipper(
        video_file=args.video_file,
        buffer=args.buffer,
        outcsv=args.outcsv,
        outclips=args.outclips,
        outname=args.outname,
        orderby=args.orderby,
        nclips=args.nclips,
        starttime=args.starttime,
        reverse_clips=args.reverse_clips,
        max_time_diff=args.max_time_diff,
        delta=args.delta,
        skip_clips=args.skip_clips,
        log_callback=print,
    )


def run_clipper(
    video_file,
    buffer=1.5,
    outcsv=False,
    outclips=False,
    outname="clips",
    orderby="chrono",
    nclips=None,
    starttime=0,
    reverse_clips=False,
    max_time_diff=2.5,
    delta=0.02,
    skip_clips=None,
    log_callback=None,
):
    if skip_clips is None:
        skip_clips = []
    if log_callback is None:
        log_callback = print
    timestamps = detect_hits(video_file, start_time=starttime, delta=delta, log_callback=log_callback)
    points_df = get_points(timestamps, max_time_diff=max_time_diff)
    if orderby == "duration":
        points_df = points_df.sort_values(by=["duration", "shots"], ascending=False)
    elif orderby == "shots":
        points_df = points_df.sort_values(by=["shots", "duration"], ascending=[False, True])
    elif orderby == "chrono":
        points_df = points_df.sort_values(by=["start"], ascending=True)
    if nclips:
        points_df = points_df.head(nclips + len(skip_clips))
    points_df = points_df.reset_index()
    if len(skip_clips) > 0:
        points_df = points_df.drop(index=skip_clips, errors="ignore")
    points_df = points_df.round(2)
    points_df.index.name = "clip_id"
    points_df.rename(columns={"index": "point"}, inplace=True)

    # Format DataFrame for output/logging
    df = points_df.copy()
    df["start"] = pd.to_datetime(df["start"], unit="s").dt.strftime("%H:%M:%S")
    df["end"] = pd.to_datetime(df["end"], unit="s").dt.strftime("%H:%M:%S")
    log_callback("")
    log_callback("Extracted points:")
    log_callback(df.to_markdown(index=True))
    if outcsv:
        df.to_csv(f"{outname}.csv", index=True)
    log_callback("")

    if outclips:
        log_callback("Creating video clips ...")
        for f in glob.glob(f"{outname}_*.mp4") + [f"{outname}.mp4"]:
            if os.path.exists(f):
                os.remove(f)
        st = time.time()
        keyframes = get_keyframes(video_file)
        segment_files = create_clips(
            video_file,
            points_df[["start", "end"]].values,
            keyframes,
            prefix=outname,
            buffer=buffer,
        )
        if reverse_clips:
            segment_files = segment_files[::-1]
        concat_clips(segment_files, f"{outname}.mp4")
        log_callback(f"Done! Created {len(segment_files)} video clips in {round(time.time()-st)}s")
    return points_df


def detect_hits(video_file, start_time=0, hop_length=32, delta=0.02, log_callback=None):
    if log_callback is None:
        log_callback = print
    hits = []
    model = pickle.load(open("model.pkl", "rb"))
    for iter, (sr, audio, offset, end) in enumerate(
        extract_audio_from_video(video_file, start_time=start_time, chunk_size=1000)
    ):
        dtoffset = datetime.timedelta(seconds=offset)
        dtend = datetime.timedelta(seconds=end)
        log_callback(f"Processing chunk {iter+1}, start={dtoffset}, end={dtend} ... ")
        st = time.time()
        onset_times = get_onsets(audio, sr, hop_length, delta=delta)
        X = []
        invalid = set()
        for i, onset in enumerate(onset_times):
            clip = onset_clip(audio, sr, onset)
            feats = features(clip, sr)
            if len(feats) != model.nfeatures:
                invalid.add(i)
                continue
            X.append(feats)
        model_preds = model.predict(X)
        hits += [onset_times[i] + offset for i, pred in enumerate(model_preds) if pred == 1 and not i in invalid]
        log_callback(f"completed in {round(time.time()-st)}s")
    return hits


def create_groups(timestamps, min_size=1, max_time_diff=2):
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


def get_points(timestamps, max_time_diff=1):
    groups = create_groups(timestamps, max_time_diff=max_time_diff)
    df = pd.DataFrame(columns=["start", "end", "shots", "duration"])
    for i, group in enumerate(groups):
        start = group[0]
        end = group[-1]
        shots = len(group)
        for j in range(1, len(group)):
            if group[j] - group[j - 1] < 0.1:
                shots -= 1
        duration = end - start
        df.loc[i] = [start, end, shots, duration]

    return df


def concat_clips(files, output_file):
    try:
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
    except Exception as e:
        raise
    finally:
        if os.path.exists(TMP_CONCAT_FILE):
            os.remove(TMP_CONCAT_FILE)


def get_keyframes(video_file):
    container = av.open(video_file)

    stream = container.streams.video[0]  # Get first video stream
    keyframes = [float(packet.pts * stream.time_base) for packet in container.demux(stream) if packet.is_keyframe]

    # Normalize keyframes to start at 0
    # Might be off due to getting a clip of full vid from something like untwitch
    keyframes = [k - min(keyframes) for k in keyframes]

    return keyframes


def create_clips(video_file, points, keyframes, prefix="clip", buffer=1):
    segment_files = []
    file_type = video_file.split(".")[-1]
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
            "-avoid_negative_ts",
            "make_zero",  # Forces correct trim
            segment_file,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        segment_files.append(segment_file)

    return segment_files


if __name__ == "__main__":
    main()
