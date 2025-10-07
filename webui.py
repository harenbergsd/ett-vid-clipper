import gradio as gr
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Import the clipper functions directly
from clipper import process_video

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


def process_video_direct(config: ClipperConfig) -> str:
    """
    Process the uploaded video file using direct function calls to clipper.py.

    Args:
        config: ClipperConfig object containing all processing parameters

    Returns:
        str: Combined output from the clipper processing
    """
    try:
        # Check if video file exists
        if not config.video_file or not os.path.exists(config.video_file):
            return f"Error: Video file not found: {config.video_file}"

        # Parse skip clips from text input
        skip_clips = []
        if config.skip_clips_text and config.skip_clips_text.strip():
            try:
                skip_clips = [int(x.strip()) for x in config.skip_clips_text.split(",") if x.strip()]
            except ValueError:
                return "Error: Skip clips must be comma-separated integers"

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
            detection_sensitivity=config.detection_sensitivity
        )

        # Combine output messages
        output = "\n".join(result["output_messages"])
        output += f"\n\n✅ Processing completed successfully! Created {result['clip_count']} clips."

        return output

    except Exception as e:
        return f"Error during processing: {str(e)}"

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
    return process_video_direct(config)

# Create the Gradio interface
with gr.Blocks(title="Eleven Table Tennis Video Clipper", theme=gr.themes.Soft()) as interface:
    # Custom CSS for transparent labels with borders
    gr.HTML("""
    <style>
    .gradio-container .has-info {
        background: #383640 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }
    </style>
    """)
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
                            info="Time buffer around clips (seconds)"
                        )
                        skip_clips_min_shots = gr.Number(
                            label="Skip Clips with Few Shots",
                            value=0,
                            minimum=0,
                            maximum=100,
                            info="Filter out clips with fewer than this many shots (0 = no filtering)"
                        )

                    output_prefix = gr.Textbox(
                        label="Output Prefix",
                        value="clips",
                        placeholder="clips",
                        info="Prefix for output files"
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
                            info="Start processing from this time (seconds)"
                        )
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
            
            output_text = gr.Textbox(
                label="Console Output",
                lines=25,
                max_lines=50,
                show_copy_button=True
            )

    # Event handlers
    inputs_list = [
        video_input, buffer_time, output_csv, create_clips, output_prefix,
        sort_by, max_clips, start_time, skip_clips_text, skip_clips_min_shots,
        reverse_order, max_time_diff, detection_sensitivity
    ]
    
    # Execute processing and show output
    generate_btn.click(
        fn=gradio_interface,
        inputs=inputs_list,
        outputs=[output_text]
    )

    gr.Markdown("""
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
    - **Max Clips** helps limit processing for very long videos
    - **Skip Clips** is useful for excluding problematic sections
    - **Detection Sensitivity** affects how many audio events are detected
    """)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)