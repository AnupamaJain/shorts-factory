from mcp.server.fastmcp import FastMCP
import sys
import os
import yaml

# Ensure project modules are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.tools import render_video
from agents.social_media_agent import generate_instagram_metadata

# Load Configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "shorts_config.yaml")
with open(config_path, 'r') as f:
    SHORTS_CONFIG = yaml.safe_load(f)
    
server_name = SHORTS_CONFIG.get("mcp_server", {}).get("name", "ShortsFactoryMCP")

# Initialize the Model Context Protocol Server
mcp = FastMCP(server_name)

@mcp.tool()
def render_video_mcp(script: str, emotion_theme: str = "neutral", output_filename: str = "agent_output.mp4") -> str:
    """
    Renders an mp4 video using the Shorts-Factory engine.
    Args:
        script: The spoken voiceover script.
        emotion_theme: The emotional pacing (neutral, aggressive, fear, poetic).
        output_filename: Output filename (e.g. video.mp4)
    """
    return render_video.invoke({
        "script": script, 
        "emotion_theme": emotion_theme, 
        "output_filename": output_filename
    })

@mcp.tool()
def generate_instagram_post(script: str) -> dict:
    """
    Generates Instagram metadata (title, description, hashtags) for a video.
    Args:
        script: The finalized video script.
    """
    return generate_instagram_metadata(script)

if __name__ == "__main__":
    print("🚀 Starting Shorts-Factory MCP Server on STDIO...")
    mcp.run()
