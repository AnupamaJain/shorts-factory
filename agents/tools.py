import os
import yaml
from langchain_core.tools import tool

def update_config(config_updates: dict, config_path: str = "core/config.yaml"):
    """Updates the config.yaml file with the given dictionary of updates."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Simple recursive update
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = deep_update(config, config_updates)
    
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
        
    return config

@tool
def render_video(script: str, emotion_theme: str = "neutral", output_filename: str = "agent_output.mp4"):
    """
    Renders a video using the shorts-factory engine.
    
    Args:
        script: The full voiceover script for the video.
        emotion_theme: The primary emotion (fear, greed, neutral) to guide the visuals.
        output_filename: The name of the final mp4 file.
    """
    # 1. Save the script to a temp file
    os.makedirs("temp", exist_ok=True)
    script_path = "temp/agent_script.txt"
    with open(script_path, "w") as f:
        f.write(script)

    # 2. Update config to use this script and AI TTS
    config_updates = {
        "video": {
            "output": f"outputs/{output_filename}"
        },
        "narrator": {
            "use_original_audio": False,
            "auto_extract_script": False,
            "script_file": script_path,
            "voice": "en-US-AriaNeural" # or whichever is preferred
        }
    }
    update_config(config_updates)

    # 3. Trigger the rendering script (we use os.system for simplicity, but could import main)
    # Since viral_storyteller.py uses asyncio, we can just run it as a subprocess
    print(f"🎬 Agent is triggering video rendering for {output_filename}...")
    exit_code = os.system("python core/viral_storyteller.py")
    
    if exit_code == 0:
        return f"Video successfully rendered to outputs/{output_filename}"
    else:
        return "Video rendering failed. Check the logs."
