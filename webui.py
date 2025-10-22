import gradio as gr
import os
import glob
import subprocess
import platform
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Import the clipper functions directly
from clipper import process_video

OUTPUT_DIR = Path("ett_clipper_output")


@dataclass
class ClipperConfig:
    """Configuration for video clipping parameters."""

    video_file: str
    buffer_time: float = 1.5
    output_csv: bool = False
    create_clips: bool = True
    output_prefix: str = "clips"
    sort_by: str = "chrono"
    max_clips: Optional[int] = None
    start_time: int = 0
    skip_clips_text: str = ""
    skip_clips_min_shots: int = 0
    reverse_order: bool = False
    max_time_diff: float = 2.5
    detection_sensitivity: float = 0.02


def open_output_folder():
    """Open the output folder in the system file browser."""
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Open folder based on operating system
    try:
        if platform.system() == "Windows":
            os.startfile(str(OUTPUT_DIR.absolute()))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(OUTPUT_DIR.absolute())])
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", str(OUTPUT_DIR.absolute())])
    except Exception:
        # Silently fail - user will notice if folder doesn't open
        pass


def get_generated_clips(output_prefix: str = "clips") -> List[Tuple[str, str]]:
    """
    Get list of generated clip files from the output folder with filenames as captions.
    The compiled video (clips.mp4) will appear first if it exists.

    Args:
        output_prefix: Prefix used for output files

    Returns:
        List of tuples containing (file_path, caption)
    """
    clip_files = []

    # First, add the compiled video if it exists (should appear first)
    compiled_video = f"{OUTPUT_DIR}/{output_prefix}.mp4"
    if os.path.exists(compiled_video):
        filename = os.path.basename(compiled_video)
        clip_files.append((compiled_video, filename))

    # Then add individual clip files (clips_0.mp4, clips_1.mp4, etc.)
    pattern = f"{OUTPUT_DIR}/{output_prefix}_*.mp4"
    for file_path in sorted(glob.glob(pattern)):
        filename = os.path.basename(file_path)
        clip_files.append((file_path, filename))

    return clip_files


def process_video_direct(config: ClipperConfig) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process the uploaded video file using direct function calls to clipper.py.

    Args:
        config: ClipperConfig object containing all processing parameters

    Returns:
        Tuple containing:
        - str: Combined output from the clipper processing
        - List[Tuple[str, str]]: List of tuples containing (file_path, caption) for generated clips
    """
    try:
        # Check if video file exists
        if not config.video_file or not os.path.exists(config.video_file):
            return f"Error: Video file not found: {config.video_file}", []

        # Parse skip clips from text input
        skip_clips = []
        if config.skip_clips_text and config.skip_clips_text.strip():
            try:
                skip_clips = [int(x.strip()) for x in config.skip_clips_text.split(",") if x.strip()]
            except ValueError:
                return "Error: Skip clips must be comma-separated integers", []

        print(f"Processing video: {config.video_file}")

        # Call the clipper function directly
        result = process_video(
            video_file=config.video_file,
            buffer=config.buffer_time,
            output_csv=config.output_csv,
            create_clips_flag=config.create_clips,
            output_prefix=config.output_prefix,
            sort_by=config.sort_by,
            max_clips=config.max_clips,
            start_time=config.start_time,
            skip_clips=skip_clips,
            skip_clips_min_shots=config.skip_clips_min_shots,
            reverse_clips=config.reverse_order,
            max_time_diff=config.max_time_diff,
            detection_sensitivity=config.detection_sensitivity,
        )

        # Combine output messages
        output = "\n".join(result["output_messages"])
        output += f"\n\n✅ Processing completed successfully! Created {result['clip_count']} clips."

        # Get the generated clip files with filenames as captions
        clip_files = get_generated_clips(config.output_prefix)

        return output, clip_files

    except Exception as e:
        return f"Error during processing: {str(e)}", []


def get_format(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json", video_path],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return data["format"]["format_name"]  # e.g., "mov,mp4,m4a,3gp,3g2,mj2"
    except Exception:
        return None


def fix_moov_if_needed(video_path):
    # Check if video format requires moov atom fix
    # This fixes some issues where gradio cannot play some MP4 files due to moov atom placement
    fixed_path = os.path.join(tempfile.gettempdir(), "fixed.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        video_path,
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "1",
        fixed_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fixed_path if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0 else video_path


def handle_upload(video):
    safe_video = fix_moov_if_needed(video)
    return safe_video


def gradio_interface(
    video_file,
    buffer_time,
    output_csv,
    create_clips,
    output_prefix,
    sort_by,
    max_clips,
    start_time,
    skip_clips_text,
    skip_clips_min_shots,
    reverse_order,
    max_time_diff,
    detection_sensitivity,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Gradio interface function that processes video and returns output and clip files with captions.
    """
    config = ClipperConfig(
        video_file=video_file,
        buffer_time=buffer_time,
        output_csv=output_csv,
        create_clips=create_clips,
        output_prefix=output_prefix,
        sort_by=sort_by,
        max_clips=max_clips,
        start_time=start_time,
        skip_clips_text=skip_clips_text,
        skip_clips_min_shots=skip_clips_min_shots,
        reverse_order=reverse_order,
        max_time_diff=max_time_diff,
        detection_sensitivity=detection_sensitivity,
    )
    return process_video_direct(config)


# Create the Gradio interface
with gr.Blocks(title="Eleven Table Tennis Video Clipper", theme=gr.themes.Soft()) as interface:
    # Custom CSS for transparent labels with borders
    gr.HTML(
        """
    <style>
    .gradio-container .has-info {
        background: #383640 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }
    </style>
    """
    )
    gr.Markdown("# 🎥 Eleven Table Tennis Video Clipper")
    gr.Markdown("Upload a video file to extract clips based on audio events with full parameter control.")

    with gr.Row():
        with gr.Column(scale=2):
            # Video Upload Section
            gr.Markdown("### 📹 Video Input")

            file_input = gr.File(label="Upload Video File (MP4, MOV, etc.)")

            preview_video = gr.Video(label="Input Video")
            file_input.change(handle_upload, inputs=file_input, outputs=preview_video)

            gr.Markdown(
                """
            **💡 Video Preview Note:** Some MP4 files may show an error due to browser codec limitations.
            This does not affect clip generation - the processing will work normally with VLC-compatible files.
            """
            )

            # Basic Settings Group
            with gr.Accordion("🔧 Basic Settings", open=True):
                with gr.Group():
                    with gr.Row():
                        buffer_time = gr.Slider(
                            label="Buffer Time",
                            minimum=0.1,
                            maximum=5.0,
                            value=1.5,
                            step=0.1,
                            info="Time buffer around clips (seconds)",
                        )
                        skip_clips_min_shots = gr.Number(
                            label="Skip Clips with Few Shots",
                            value=0,
                            minimum=0,
                            maximum=100,
                            info="Filter out clips with fewer than this many shots (0 = no filtering)",
                        )

                    output_prefix = gr.Textbox(
                        label="Output Prefix", value="clips", placeholder="clips", info="Prefix for output files"
                    )

            # Advanced Settings Accordion (collapsible)
            with gr.Accordion("🔧 Advanced Settings", open=False):
                with gr.Group():
                    with gr.Row():
                        start_time = gr.Slider(
                            label="Start Time",
                            minimum=0,
                            maximum=3600,
                            value=0,
                            step=1,
                            info="Start processing from this time (seconds)",
                        )
                        output_csv = gr.Checkbox(label="Export CSV", value=False, info="Export timestamps to CSV file")
                        create_clips = gr.Checkbox(
                            label="Create Video Clips", value=True, info="Create video clips and combined video"
                        )

                    with gr.Row():
                        sort_by = gr.Dropdown(
                            label="Sort By",
                            choices=["chrono", "shots", "duration"],
                            value="chrono",
                            info="How to sort the extracted clips",
                        )
                        max_clips = gr.Number(
                            label="Max Clips",
                            value=None,
                            minimum=None,
                            maximum=1000,
                            info="Maximum number of clips to extract (leave empty for unlimited)",
                        )

                    skip_clips_text = gr.Textbox(
                        label="Skip Clips",
                        placeholder="e.g., 0, 2, 5",
                        info="Comma-separated list of clip indices to skip",
                    )

                    with gr.Row():
                        reverse_order = gr.Checkbox(
                            label="Reverse Clip Order", value=False, info="Reverse the order of clips in final video"
                        )
                        max_time_diff = gr.Slider(
                            label="Max Time Between Hits",
                            minimum=0.5,
                            maximum=10.0,
                            value=2.5,
                            step=0.1,
                            info="Maximum time between onsets to group into one clip",
                        )

                    detection_sensitivity = gr.Slider(
                        label="Detection Sensitivity",
                        minimum=0.001,
                        maximum=0.1,
                        value=0.02,
                        step=0.001,
                        info="Sensitivity for audio onset detection",
                    )

        with gr.Column(scale=3):
            generate_btn = gr.Button("🚀 Generate Clips", variant="primary", size="lg")

            output_text = gr.Textbox(label="Console Output", lines=15, max_lines=30, show_copy_button=True)

            # Dynamic clip display area
            gr.Markdown("### 🎬 Generated Clips")

            # Open folder button
            open_folder_btn = gr.Button("📁 Open Output Folder", variant="primary", size="sm")

            clips_display = gr.Gallery(elem_id="gallery", columns=2, rows=2, height="auto", object_fit="contain")

    # Event handlers
    inputs_list = [
        file_input,
        buffer_time,
        output_csv,
        create_clips,
        output_prefix,
        sort_by,
        max_clips,
        start_time,
        skip_clips_text,
        skip_clips_min_shots,
        reverse_order,
        max_time_diff,
        detection_sensitivity,
    ]

    # Execute processing and show output + clips
    generate_btn.click(fn=gradio_interface, inputs=inputs_list, outputs=[output_text, clips_display])

    # Open folder button handler
    open_folder_btn.click(fn=open_output_folder)

    gr.Markdown(
        """
    ### 📖 Instructions:
    1. **Upload** an MP4 video file using the video input above
    2. **Configure** the processing parameters using the control groups:
       - **Basic Settings**: Buffer time, output prefix, and filtering clips with few shots
       - **Advanced Settings**: All other processing options including timing, output formats, and detection settings
    3. **Generate Clips** by clicking the button to start processing
    4. **Monitor Progress** in the output area above
    5. **Find Results** in the output directory (clips, CSV files, etc.)

    ### 💡 Tips:
    - Start with default settings for most use cases
    - Use **Buffer Time** to add padding around detected events
    - **Skip Clips with Few Shots** filters out short rallies automatically
    - **Skip Clips** is useful for excluding specific clips by index (see the file name for the index value)
    - To build a top-10 video, sort by **shots** and set **max clips to 10**
        - If you want the best point at the end of the video, enable **Reverse Clip Order**
    - **Detection Sensitivity** affects how many audio events are detected
    """
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", share=False, inbrowser=True)
