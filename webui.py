import gradio as gr
import subprocess
import os
from pathlib import Path
from typing import Tuple, Optional

CLIPPER_PATH = str(Path(__file__).parent / "clipper.py")

def run_command(cmd: list) -> Tuple[str, str, int]:
    """Run a command and return stdout, stderr, and return code."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", f"Error running command: {str(e)}", 1

def generate_command_string(
    video_file,
    buffer_time,
    output_csv,
    create_clips,
    output_prefix,
    sort_by,
    max_clips,
    start_time,
    skip_clips_text,
    reverse_order,
    max_time_diff,
    detection_sensitivity
) -> str:
    """
    Generate the command string that will be executed.
    
    Returns:
        str: Formatted command string
    """
    if not video_file:
        return "No video file provided"
    
    # Build command with all parameters
    cmd = ["uv", "run", "clipper.py", video_file]
    
    # Add parameters based on values
    if buffer_time != 1.5:  # Only add if different from default
        cmd.extend(["--buffer", str(buffer_time)])
    
    if output_csv:
        cmd.append("--outcsv")
    
    if create_clips:
        cmd.append("--outclips")
    
    if output_prefix != "clips":
        cmd.extend(["--outname", output_prefix])
    
    if sort_by != "chrono":
        cmd.extend(["--orderby", sort_by])
    
    if max_clips and max_clips > 0:
        cmd.extend(["--nclips", str(max_clips)])
    
    if start_time != 0:
        cmd.extend(["--starttime", str(start_time)])
    
    # Parse skip clips from text input
    if skip_clips_text and skip_clips_text.strip():
        try:
            skip_clips = [int(x.strip()) for x in skip_clips_text.split(",") if x.strip()]
            if skip_clips:
                cmd.extend(["--skip-clips"] + [str(x) for x in skip_clips])
        except ValueError:
            return "Error: Skip clips must be comma-separated integers"
    
    if reverse_order:
        cmd.append("--reverse-clips")
    
    if max_time_diff != 2.5:
        cmd.extend(["--max-time-diff", str(max_time_diff)])
    
    if detection_sensitivity != 0.02:
        cmd.extend(["--delta", str(detection_sensitivity)])
    
    return " ".join(cmd)

def process_video(
    video_file,
    buffer_time,
    output_csv,
    create_clips,
    output_prefix,
    sort_by,
    max_clips,
    start_time,
    skip_clips_text,
    reverse_order,
    max_time_diff,
    detection_sensitivity
) -> str:
    """
    Process the uploaded video file using the clipper script with full parameter support.

    Args:
        video_file: Path to the uploaded video file
        buffer_time: Time buffer around clips (seconds)
        output_csv: Whether to export CSV timestamps
        create_clips: Whether to create video clips
        output_prefix: Prefix for output files
        sort_by: How to sort clips ("chrono", "shots", "duration")
        max_clips: Maximum number of clips to extract (None for unlimited)
        start_time: Start processing from this time (seconds)
        skip_clips_text: Comma-separated list of clip indices to skip
        reverse_order: Whether to reverse clip order in final video
        max_time_diff: Maximum time between onsets to group into one clip
        detection_sensitivity: Sensitivity for onset detection

    Returns:
        str: Combined output from the clipper command
    """
    if not video_file:
        return "Error: No video file provided"

    if not os.path.exists(video_file):
        return f"Error: Video file not found: {video_file}"

    # Check if clipper.py exists
    if not os.path.exists(CLIPPER_PATH):
        return "Error: clipper.py not found in current directory"

    # Build command with all parameters
    cmd = ["uv", "run", CLIPPER_PATH, video_file]

    # Add parameters based on values
    if buffer_time != 1.5:  # Only add if different from default
        cmd.extend(["--buffer", str(buffer_time)])

    if output_csv:
        cmd.append("--outcsv")

    if create_clips:
        cmd.append("--outclips")

    if output_prefix != "clips":
        cmd.extend(["--outname", output_prefix])

    if sort_by != "chrono":
        cmd.extend(["--orderby", sort_by])

    if max_clips and max_clips > 0:
        cmd.extend(["--nclips", str(max_clips)])

    if start_time != 0:
        cmd.extend(["--starttime", str(start_time)])

    # Parse skip clips from text input
    if skip_clips_text and skip_clips_text.strip():
        try:
            skip_clips = [int(x.strip()) for x in skip_clips_text.split(",") if x.strip()]
            if skip_clips:
                cmd.extend(["--skip-clips"] + [str(x) for x in skip_clips])
        except ValueError:
            return "Error: Skip clips must be comma-separated integers"

    if reverse_order:
        cmd.append("--reverse-clips")

    if max_time_diff != 2.5:
        cmd.extend(["--max-time-diff", str(max_time_diff)])

    if detection_sensitivity != 0.02:
        cmd.extend(["--delta", str(detection_sensitivity)])

    print(f"Running command: {' '.join(cmd)}")

    stdout, stderr, returncode = run_command(cmd)

    # Combine outputs
    output = ""
    if stdout:
        output += f"STDOUT:\n{stdout}\n"
    if stderr:
        output += f"STDERR:\n{stderr}\n"

    if returncode == 0:
        output += "\n✅ Command completed successfully!"
    else:
        output += f"\n❌ Command failed with return code: {returncode}"

    return output

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
    reverse_order,
    max_time_diff,
    detection_sensitivity
) -> str:
    """
    Gradio interface function that processes video and returns output.
    """
    return process_video(
        video_file, buffer_time, output_csv, create_clips, output_prefix,
        sort_by, max_clips, start_time, skip_clips_text, reverse_order,
        max_time_diff, detection_sensitivity
    )

# Create the Gradio interface
with gr.Blocks(title="ETT Video Clipper", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎥 ETT Video Clipper")
    gr.Markdown("Upload an MP4 video file to extract clips based on audio events with full parameter control.")

    with gr.Row():
        with gr.Column(scale=2):
            # Video Upload Section
            gr.Markdown("### 📹 Video Input")
            video_input = gr.Video(
                label="Upload MP4 Video",
                sources=["upload"],
            )
            gr.Markdown("""
            **💡 Video Preview Note:** Some MP4 files may show "video not playable" in the preview due to browser codec limitations.
            This doesn't affect clip generation - the processing will work normally with VLC-compatible files.
            """)

            # Basic Output Settings
            gr.Markdown("### 📊 Basic Output Settings")
            with gr.Row():
                buffer_time = gr.Slider(
                    label="Buffer Time",
                    minimum=0.1,
                    maximum=5.0,
                    value=1.5,
                    step=0.1,
                    info="Time buffer around clips (seconds)"
                )
                start_time = gr.Slider(
                    label="Start Time",
                    minimum=0,
                    maximum=3600,
                    value=0,
                    step=1,
                    info="Start processing from this time (seconds)"
                )

            with gr.Row():
                output_csv = gr.Checkbox(
                    label="Export CSV",
                    value=False,
                    info="Export timestamps to CSV file"
                )
                create_clips = gr.Checkbox(
                    label="Create Video Clips",
                    value=True,
                    info="Create video clips and combined video"
                )

            output_prefix = gr.Textbox(
                label="Output Prefix",
                value="clips",
                placeholder="clips",
                info="Prefix for output files"
            )

            # Processing Settings
            gr.Markdown("### ⚙️ Processing Settings")
            with gr.Row():
                sort_by = gr.Dropdown(
                    label="Sort By",
                    choices=["chrono", "shots", "duration"],
                    value="chrono",
                    info="How to sort the extracted clips"
                )
                max_clips = gr.Number(
                    label="Max Clips",
                    value=None,
                    minimum=None,
                    maximum=1000,
                    info="Maximum number of clips to extract (leave empty for unlimited)"
                )

            skip_clips_text = gr.Textbox(
                label="Skip Clips",
                placeholder="e.g., 0, 2, 5",
                info="Comma-separated list of clip indices to skip"
            )

            # Advanced Settings
            gr.Markdown("### 🔧 Advanced Settings")
            with gr.Row():
                reverse_order = gr.Checkbox(
                    label="Reverse Clip Order",
                    value=False,
                    info="Reverse the order of clips in final video"
                )
                max_time_diff = gr.Slider(
                    label="Max Time Between Hits",
                    minimum=0.5,
                    maximum=10.0,
                    value=2.5,
                    step=0.1,
                    info="Maximum time between onsets to group into one clip"
                )

            detection_sensitivity = gr.Slider(
                label="Detection Sensitivity",
                minimum=0.001,
                maximum=0.1,
                value=0.02,
                step=0.001,
                info="Sensitivity for audio onset detection"
            )

            generate_btn = gr.Button("🚀 Generate Clips", variant="primary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("### 📋 Command Preview & Output")
            
            # Command Preview Section
            gr.Markdown("#### 🔍 Command Line Preview")
            command_preview = gr.Textbox(
                label="Command to be Executed",
                lines=3,
                max_lines=5,
                show_copy_button=True,
                interactive=False,
                info="Preview of the command that will be executed"
            )
            
            gr.Markdown("#### 📊 Processing Output")
            output_text = gr.Textbox(
                label="Command Output",
                lines=20,
                max_lines=50,
                show_copy_button=True,
                info="View the command execution results here"
            )

    # Event handlers
    inputs_list = [
        video_input, buffer_time, output_csv, create_clips, output_prefix,
        sort_by, max_clips, start_time, skip_clips_text, reverse_order,
        max_time_diff, detection_sensitivity
    ]
    
    # Show command preview immediately when button is clicked
    generate_btn.click(
        fn=generate_command_string,
        inputs=inputs_list,
        outputs=[command_preview]
    )
    
    # Then execute the command and show output
    generate_btn.click(
        fn=gradio_interface,
        inputs=inputs_list,
        outputs=[output_text]
    )

    gr.Markdown("""
    ### 📖 Instructions:
    1. **Upload** an MP4 video file using the video input above
    2. **Configure** the processing parameters using the control groups:
       - **Basic Settings**: Buffer time, output options, and file prefix
       - **Processing Settings**: Sorting, clip limits, and skip options
       - **Advanced Settings**: Fine-tuning for detection and grouping
    3. **Generate Clips** by clicking the button to start processing
    4. **Monitor Progress** in the output area above
    5. **Find Results** in the current directory (clips/, CSV files, etc.)

    ### 💡 Tips:
    - Start with default settings for most use cases
    - Use **Buffer Time** to add padding around detected events
    - **Max Clips** helps limit processing for very long videos
    - **Skip Clips** is useful for excluding problematic sections
    - **Detection Sensitivity** affects how many audio events are detected
    """)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)