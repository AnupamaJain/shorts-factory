import os
from instagrapi import Client
from dotenv import load_dotenv

load_dotenv()

def post_video_to_instagram(video_path: str, caption: str, thumbnail_path: str = None):
    """
    Autonomously uploads a video to Instagram Reels.
    Requires IG_USERNAME and IG_PASSWORD in the .env file.
    Optionally accepts a custom thumbnail_path.
    """
    print("🚀 PUBLISHER AGENT: Preparing to post to Instagram...")
    
    username = os.getenv("IG_USERNAME")
    password = os.getenv("IG_PASSWORD")
    
    if not username or not password:
        print("❌ Error: IG_USERNAME or IG_PASSWORD not found in .env file.")
        print("Please add your credentials to the .env file to enable auto-posting.")
        return False
        
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return False

    try:
        # Initialize Instagrapi client
        cl = Client()
        
        # Login
        print(f"Logging into Instagram as {username}...")
        cl.login(username, password)
        
        # Upload as Reel
        print("Uploading video...")
        if thumbnail_path and os.path.exists(thumbnail_path):
            print(f"Applying custom SEO thumbnail: {thumbnail_path}")
            media = cl.clip_upload(video_path, caption, thumbnail=thumbnail_path)
        else:
            media = cl.clip_upload(video_path, caption)
        
        print(f"✅ Success! Video posted to Instagram. Media ID: {media.id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to post to Instagram: {e}")
        # Sometimes 2FA or suspicious login attempts block instagrapi
        print("Note: If you have 2FA enabled, or if Instagram flagged the login as suspicious, the automated upload will fail.")
        return False

if __name__ == "__main__":
    # Test script
    post_video_to_instagram("outputs/agent_output.mp4", "Test post! #trading")
