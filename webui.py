import gradio as gr
import subprocess
import os
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

CLIPPER_PATH = str(Path(__file__).parent / "clipper.py")

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

def build_clipper_command(config: ClipperConfig) -> tuple:
    """
    Build the clipper command and return both list and string formats.
    
    Returns:
        tuple: (command_list, command_string)
    """
    if not config.video_file:
        raise ValueError("No video file provided")
    
    # Build command with all parameters
    cmd = ["uv", "run", CLIPPER_PATH, config.video_file]
    
    # Add parameters based on values
    if config.buffer_time != 1.5:  # Only add if different from default
        cmd.extend(["--buffer", str(config.buffer_time)])
    
    if config.output_csv:
        cmd.append("--outcsv")
    
    if config.create_clips:
        cmd.append("--outclips")
    
    if config.output_prefix != "clips":
        cmd.extend(["--outname", config.output_prefix])
    
    if config.sort_by != "chrono":
        cmd.extend(["--orderby", config.sort_by])
    
    if config.max_clips and config.max_clips > 0:
        cmd.extend(["--nclips", str(config.max_clips)])
    
    if config.start_time != 0:
        cmd.extend(["--starttime", str(config.start_time)])
    
    # Parse skip clips from text input
    if config.skip_clips_text and config.skip_clips_text.strip():
        try:
            skip_clips = [int(x.strip()) for x in config.skip_clips_text.split(",") if x.strip()]
            if skip_clips:
                cmd.extend(["--skip-clips"] + [str(x) for x in skip_clips])
        except ValueError:
            raise ValueError("Skip clips must be comma-separated integers")
    
    if config.skip_clips_min_shots and config.skip_clips_min_shots > 0:
        cmd.extend(["--skip-clips-min-shots", str(int(config.skip_clips_min_shots))])
    
    if config.reverse_order:
        cmd.append("--reverse-clips")
    
    if config.max_time_diff != 2.5:
        cmd.extend(["--max-time-diff", str(config.max_time_diff)])
    
    if config.detection_sensitivity != 0.02:
        cmd.extend(["--delta", str(config.detection_sensitivity)])
    
    return cmd, " ".join(cmd)


def generate_command_string(config: ClipperConfig) -> str:
    """
    Generate the command string that will be executed.
    
    Returns:
        str: Formatted command string
    """
    try:
        cmd_list, cmd_string = build_clipper_command(config)
        return cmd_string
    except ValueError as e:
        return f"Error: {str(e)}"

def process_video(config: ClipperConfig) -> str:
    """
    Process the uploaded video file using the clipper script with full parameter support.

    Args:
        config: ClipperConfig object containing all processing parameters

    Returns:
        str: Combined output from the clipper command
    """
    try:
        # Build command using centralized function
        cmd_list, cmd_string = build_clipper_command(config)

        # Check if video file exists
        if not os.path.exists(config.video_file):
            return f"Error: Video file not found: {config.video_file}"

        # Check if clipper.py exists
        if not os.path.exists(CLIPPER_PATH):
            return "Error: clipper.py not found in current directory"

        print(f"Running command: {cmd_string}")

        stdout, stderr, returncode = run_command(cmd_list)

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

    except ValueError as e:
        return f"Error: {str(e)}"

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
    detection_sensitivity
) -> str:
    """
    Gradio interface function that processes video and returns output.
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
        detection_sensitivity=detection_sensitivity
    )
    return process_video(config)

# Create the Gradio interface
with gr.Blocks(title="Eleven Table Tennis Video Clipper", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎥 Eleven Table Tennis Video Clipper")
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

            with gr.Row():
                skip_clips_min_shots = gr.Number(
                    label="Minimum Shots per Clip",
                    value=0,
                    minimum=0,
                    maximum=100,
                    info="Filter out clips with fewer than this many shots (0 = no filtering)"
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

        with gr.Column(scale=3):
            generate_btn = gr.Button("🚀 Generate Clips", variant="primary", size="lg")
            
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
        sort_by, max_clips, start_time, skip_clips_text, skip_clips_min_shots,
        reverse_order, max_time_diff, detection_sensitivity
    ]
    
    # Show command preview immediately when button is clicked
    generate_btn.click(
        fn=lambda *args: generate_command_string(ClipperConfig(
            video_file=args[0],
            buffer_time=args[1],
            output_csv=args[2],
            create_clips=args[3],
            output_prefix=args[4],
            sort_by=args[5],
            max_clips=args[6],
            start_time=args[7],
            skip_clips_text=args[8],
            skip_clips_min_shots=args[9],
            reverse_order=args[10],
            max_time_diff=args[11],
            detection_sensitivity=args[12]
        )),
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