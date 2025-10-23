import sys, os
import av
import glob
import pickle
import bisect
import argparse
import subprocess
import pandas as pd
import time
import datetime
from pathlib import Path
from helper import *

try:
    from bundle_utils import get_ffmpeg_path

    FFMPEG_CMD = get_ffmpeg_path()
except ImportError:
    # Running in development environment
    FFMPEG_CMD = "ffmpeg"

# Define output directory at the same level as this script
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "ett_clipper_output"

TMP_CONCAT_FILE = "__tmp_concat__.txt"


def parse_time(time_str):
    """
    Parse time string in format MM:SS or HH:MM:SS to seconds.

    Args:
        time_str: Time string like "10:30" or "1:10:30"

    Returns:
        int: Total seconds

    Examples:
        "90" -> 90 (plain seconds still supported)
        "10:30" -> 630 (10 minutes 30 seconds)
        "1:10:30" -> 4230 (1 hour 10 minutes 30 seconds)
    """
    if time_str is None:
        return None

    time_str = str(time_str).strip()
    
    # Handle empty string
    if not time_str:
        return None

    # If it's just a number, treat as seconds
    if ":" not in time_str:
        return int(time_str)

    # Split by colons
    parts = time_str.split(":")

    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or HH:MM:SS")


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
    type=str,
    default="0",
    help="time to start extracting clips (formats: seconds, MM:SS, or HH:MM:SS)",
)
argparser.add_argument(
    "--endtime",
    type=str,
    default=None,
    help="time to stop extracting clips (formats: seconds, MM:SS, or HH:MM:SS)",
)
argparser.add_argument(
    "--reverse-clips",
    action="store_true",
    help="reverse the order of the clips in the compiled video, useful for top-10 style videos.",
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
    "--skip-clips-min-shots",
    type=int,
    default=0,
    help="minimum number of shots required for a clip to be included (clips with fewer shots will be filtered out)",
)


def process_video(
    video_file,
    buffer=1.5,
    output_csv=False,
    create_clips_flag=True,
    output_prefix="clips",
    sort_by="chrono",
    max_clips=None,
    start_time=0,
    end_time=None,
    skip_clips=None,
    skip_clips_min_shots=0,
    reverse_clips=False,
    max_time_diff=2.5,
    detection_sensitivity=0.02,
):
    """
    Process video to extract clips based on audio events.

    Args:
        video_file: Path to the video file
        buffer: Time buffer around clips (seconds)
        output_csv: Whether to export timestamps to CSV
        create_clips_flag: Whether to create video clips
        output_prefix: Prefix for output files
        sort_by: How to sort clips ("chrono", "shots", "duration")
        max_clips: Maximum number of clips to extract
        start_time: Start processing from this time (seconds)
        end_time: Stop processing at this time (seconds)
        skip_clips: List of clip indices to skip
        skip_clips_min_shots: Minimum shots required for a clip
        reverse_clips: Reverse the order of clips
        max_time_diff: Maximum time between hits to group into one clip
        detection_sensitivity: Sensitivity for audio onset detection

    Returns:
        dict: Processing results including points DataFrame and output messages
    """
    if skip_clips is None:
        skip_clips = []

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_messages = []

    timestamps = detect_hits(video_file, start_time=start_time, end_time=end_time, delta=detection_sensitivity)
    points_df = get_points(timestamps, max_time_diff=max_time_diff)

    # Do any sorting or limiting of the clips
    if sort_by == "duration":
        points_df = points_df.sort_values(by=["duration", "shots"], ascending=False)
    if sort_by == "shots":
        points_df = points_df.sort_values(
            by=["shots", "duration"], ascending=[False, True]
        )  # shorter duration means more exciting?

    if max_clips:
        points_df = points_df.head(max_clips + len(skip_clips))  # + skip clips
    points_df = points_df.reset_index()
    if len(skip_clips) > 0:
        points_df = points_df.drop(index=skip_clips, errors="ignore")

    # Filter out clips with fewer than the minimum required shots
    if skip_clips_min_shots > 0:
        initial_count = len(points_df)
        points_df = points_df[points_df["shots"] >= skip_clips_min_shots]
        filtered_count = initial_count - len(points_df)
        if filtered_count > 0:
            msg = f"Filtered out {filtered_count} clips with fewer than {skip_clips_min_shots} shots"
            output_messages.append(msg)
            print(msg)

    points_df = points_df.round(2)
    points_df.index.name = "clip_id"
    points_df.rename(columns={"index": "point"}, inplace=True)

    # Print and optionally write out the data
    output_messages.append("")
    output_messages.append("Extracted points:")
    df = points_df.copy()
    df["start"] = pd.to_datetime(df["start"], unit="s").dt.strftime("%H:%M:%S")
    df["end"] = pd.to_datetime(df["end"], unit="s").dt.strftime("%H:%M:%S")
    output_messages.append(df.to_markdown(index=True))
    if output_csv:
        csv_path = OUTPUT_DIR / f"{output_prefix}.csv"
        df.to_csv(csv_path, index=True)
        msg = f"CSV saved to: {csv_path}"
        output_messages.append(msg)
        print(msg)
    output_messages.append("")

    # Create clips and a combined video
    if create_clips_flag:
        output_messages.append("Creating video clips ...")
        # Must clean up any existing files to avoid ffmpeg hanging
        for f in glob.glob(str(OUTPUT_DIR / f"{output_prefix}_*.mp4")) + [str(OUTPUT_DIR / f"{output_prefix}.mp4")]:
            if os.path.exists(f):
                os.remove(f)

        st = time.time()
        keyframes = get_keyframes(video_file)  # can only cut on a keyframe to avoid re-encoding
        segment_files = create_clips(
            video_file,
            points_df[["start", "end"]].values,
            keyframes,
            prefix=output_prefix,
            buffer=buffer,
            clip_ids=points_df.index.tolist(),
        )
        if reverse_clips:
            segment_files = segment_files[::-1]
        combined_video_path = OUTPUT_DIR / f"{output_prefix}.mp4"
        concat_clips(segment_files, str(combined_video_path))
        msg = f"Done! Created {len(segment_files)} video clips in {round(time.time()-st)}s"
        output_messages.append(msg)
        print(msg)
        msg = f"Combined video saved to: {combined_video_path}"
        output_messages.append(msg)
        print(msg)

    return {"points_df": points_df, "output_messages": output_messages, "clip_count": len(points_df)}


def main():
    args = argparser.parse_args()

    result = process_video(
        video_file=args.video_file,
        buffer=args.buffer,
        output_csv=args.outcsv,
        create_clips_flag=args.outclips,
        output_prefix=args.outname,
        sort_by=args.orderby,
        max_clips=args.nclips,
        start_time=parse_time(args.starttime),
        end_time=parse_time(args.endtime),
        skip_clips=args.skip_clips,
        skip_clips_min_shots=args.skip_clips_min_shots,
        reverse_clips=args.reverse_clips,
        max_time_diff=args.max_time_diff,
        detection_sensitivity=args.delta,
    )

    # Print all output messages
    for msg in result["output_messages"]:
        print(msg)


def detect_hits(video_file, start_time=0, end_time=None, hop_length=32, delta=0.02):
    hits = []
    model_path = str(Path(__file__).parent / "model.pkl")
    backup_model_path = str(Path(__file__).parent / "model_orig.pkl")

    # Try to load model.pkl first, fallback to model_orig.pkl if it doesn't exist
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"model.pkl not found, trying model_orig.pkl...")
        try:
            with open(backup_model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Successfully loaded model from model_orig.pkl")
        except FileNotFoundError:
            raise FileNotFoundError(f"Neither model.pkl nor model_orig.pkl found in {Path(__file__).parent}")
    for iter, (sr, audio, offset, end) in enumerate(
        extract_audio_from_video(video_file, start_time=start_time, end_time=end_time, chunk_size=1000)
    ):
        dtoffset = datetime.timedelta(seconds=int(offset))
        dtend = datetime.timedelta(seconds=int(end))
        print(f"Processing chunk {iter+1}, start={dtoffset}, end={dtend} ... ", end="", flush=True)
        st = time.time()

        # Detect onsets
        onset_times = get_onsets(audio, sr, hop_length, delta=delta)

        # Classify onsets as paddle hits or not
        X = []
        invalid = set()
        for i, onset in enumerate(onset_times):
            clip = onset_clip(audio, sr, onset)
            feats = features(clip, sr)
            if len(feats) != model.nfeatures:  # because we are chunking, this might happen?
                invalid.add(i)
                continue
            X.append(feats)

        model_preds = model.predict(X)
        hits += [onset_times[i] + offset for i, pred in enumerate(model_preds) if pred == 1 and not i in invalid]

        print(f"completed in {round(time.time()-st)}s", flush=True)

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
            FFMPEG_CMD,
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


def create_clips(video_file, points, keyframes, prefix="clip", buffer=1, clip_ids=None):
    segment_files = []
    file_type = video_file.split(".")[-1]

    # Determine padding width based on total number of clips
    if clip_ids is not None:
        max_id = max(clip_ids) if clip_ids else 0
        padding = len(str(max_id))
    else:
        padding = len(str(len(points) - 1))

    for i, (start, end) in enumerate(points):
        start = max(0, start - buffer)
        end += buffer
        if clip_ids is not None:
            segment_file = OUTPUT_DIR / f"{prefix}_{str(clip_ids[i]).zfill(padding)}.{file_type}"
        else:
            segment_file = OUTPUT_DIR / f"{prefix}_{str(i).zfill(padding)}.{file_type}"

        # Get closest keyframe before start
        # Otherwise, would need to re-encode, which is very slow
        start_keyframe = bisect.bisect_left(keyframes, start) - 1
        if start_keyframe < 0:
            start_keyframe = 0
        start = keyframes[start_keyframe]

        cmd = [
            FFMPEG_CMD,
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
            str(segment_file),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        segment_files.append(str(segment_file))

    return segment_files


if __name__ == "__main__":
    main()
