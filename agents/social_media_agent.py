import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import yaml
import random
import time

# Load Configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "shorts_config.yaml")
with open(config_path, 'r') as f:
    SHORTS_CONFIG = yaml.safe_load(f)

class SocialPost(BaseModel):
    title: str = Field(description="A catchy, short title for the Reel/Short.")
    description: str = Field(description="The main body of the Instagram caption, engaging and using emojis.")
    hashtags: str = Field(description="A string of 10-15 highly relevant, high-reach Instagram hashtags.")
    thumbnail_prompt: str = Field(description="A prompt to generate an engaging thumbnail image.")

def generate_instagram_metadata(script: str) -> dict:
    """
    Generates an optimized Instagram Reel caption, title, and hashtags
    based on the finalized video script.
    """
    print("📱 Social Media Agent: Crafting Instagram post metadata...")
    
    # Use LLaMA 3.3 for high-quality copywriting
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)
    
    # Extract config
    niche = SHORTS_CONFIG.get("social_media", {}).get("niche", "finance/trading")
    hashtag_count = SHORTS_CONFIG.get("social_media", {}).get("hashtag_count", 15)
    emojis = "Use emojis strategically." if SHORTS_CONFIG.get("social_media", {}).get("include_emojis", True) else "DO NOT use any emojis."

    prompt = f"""
    You are an expert Social Media Manager for a highly successful {niche} Instagram page.
    I have just created a new Instagram Reel/YouTube Short. Here is the exact script of the video:
    
    "{script}"
    
    Write the perfect Instagram post for this video. It must include:
    1. An explosive, click-worthy Hook/Title at the top.
    2. A highly engaging description that expands slightly on the script's lesson but tells them to "Watch the video for the secret." {emojis}
    3. A clear Call to Action (CTA) asking them to comment or follow.
    4. Exactly {hashtag_count} targeted hashtags for the {niche} niche (e.g., #TradingTips).
    5. A highly visual, photorealistic image prompt for an SEO-optimized, click-worthy video cover/thumbnail.
    
    Format your response EXACTLY like this:
    [TITLE]
    (Your Title)
    
    [DESCRIPTION]
    (Your Description & CTA)
    
    [HASHTAGS]
    (Your Hashtags)
    
    [THUMBNAIL]
    (Your Thumbnail Prompt)
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # Simple parser
    try:
        title = content.split("[TITLE]")[1].split("[DESCRIPTION]")[0].strip()
        description = content.split("[DESCRIPTION]")[1].split("[HASHTAGS]")[0].strip()
        hashtags = content.split("[HASHTAGS]")[1].split("[THUMBNAIL]")[0].strip()
        thumbnail_prompt = content.split("[THUMBNAIL]")[1].strip()
    except IndexError:
        # Fallback if the LLM doesn't format perfectly
        title = "New Trading Strategy 🚀"
        description = content
        hashtags = "#trading #finance"
        thumbnail_prompt = "A high contrast, dramatic cinematic shot of a glowing stock market chart, 8k resolution"
        
    post_data = {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "thumbnail_prompt": thumbnail_prompt,
        "full_post": f"{title}\n\n{description}\n\n{hashtags}"
    }
    
    # Generate the custom thumbnail using Pollinations
    print(f"🖼️ Generating SEO Thumbnail: {thumbnail_prompt[:50]}...")
    import urllib.parse
    import requests
    encoded_prompt = urllib.parse.quote(thumbnail_prompt)
    seed = random.randint(0, 999999)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920&nologo=true&seed={seed}"
    
    timestamp = int(time.time())
    thumbnail_path = f"outputs/thumbnail_{timestamp}.jpg"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            with open(thumbnail_path, 'wb') as f:
                f.write(res.content)
            post_data["thumbnail_path"] = thumbnail_path
            print("✅ Custom thumbnail generated successfully!")
        else:
            post_data["thumbnail_path"] = None
    except Exception as e:
        print(f"⚠️ Failed to generate thumbnail: {e}")
        post_data["thumbnail_path"] = None
    
    # Save the output for the user
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/instagram_post.txt", "w") as f:
        f.write(post_data["full_post"])
        f.write("\n\nThumbnail Prompt used: " + thumbnail_prompt)
        
    print("✅ Instagram post metadata generated and saved to outputs/instagram_post.txt")
    return post_data

if __name__ == "__main__":
    # Test execution
    test_script = "Don't let FOMO destroy your portfolio. Wait for the order block."
    generate_instagram_metadata(test_script)
