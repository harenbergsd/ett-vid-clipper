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

def process_video(video_file) -> str:
    """
    Process the uploaded video file using the clipper script.

    Args:
        video_file: Path to the uploaded video file

    Returns:
        str: Combined output from the clipper command
    """
    if not video_file:
        return "Error: No video file provided"

    if not os.path.exists(video_file):
        return f"Error: Video file not found: {video_file}"

    # Check if uv and clipper.py exist
    if not os.path.exists(CLIPPER_PATH):
        return "Error: clipper.py not found in current directory"

    # Run the command: uv run clipper.py <video_file> --outclips
    cmd = ["uv", "run", CLIPPER_PATH, video_file, "--outclips"]

    print(video_file)

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

def gradio_interface(video_file) -> str:
    """
    Gradio interface function that processes video and returns output.
    """
    return process_video(video_file)

# Create the Gradio interface
with gr.Blocks(title="ETT Video Clipper", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎥 ETT Video Clipper")
    gr.Markdown("Upload an MP4 video file to extract clips based on audio events.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
            label="Upload MP4 Video",
            sources= ["upload"],
        )
            generate_btn = gr.Button("🚀 Generate Clips", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Command Output",
                lines=20,
                max_lines=50,
                show_copy_button=True
            )

    # Event handler
    generate_btn.click(
        fn=gradio_interface,
        inputs=[video_input],
        outputs=[output_text]
    )

    gr.Markdown("""
    ### Instructions:
    1. Upload an MP4 video file
    2. Click "Generate Clips" to start processing
    3. View the command output in the text area above
    4. Generated clips will be saved in the current directory
    """)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)