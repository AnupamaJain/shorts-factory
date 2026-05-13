import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import sys
import os
import yaml
import time

# Ensure core module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.text_to_video import create_new_video
from agents.social_media_agent import generate_instagram_metadata
from agents.publisher_agent import post_video_to_instagram

# DeepEval Integration
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

class GroqDeepEval(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model
    def load_model(self): return self.model
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
    def get_model_name(self): return "groq-llama3"

# Load Configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "shorts_config.yaml")
with open(config_path, 'r') as f:
    SHORTS_CONFIG = yaml.safe_load(f)

class ScenePrompts(BaseModel):
    prompts: list[str] = Field(description="List of exactly 4-6 image generation prompts describing the visual scenes.")

def generate_video_from_script(user_input: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    print("✍️ AI Scriptwriter: Expanding idea into a 35-second script...")
    script_prompt = f"""
    You are an expert YouTube Shorts scriptwriter. 
    The user provided this idea/script: "{user_input}"
    
    Expand this into a highly engaging, 30-35 second voiceover script (approximately 70-85 words).
    Make it punchy, authoritative, and focused on professional trading success and discipline.
    Use high-energy hooks that promise the "secret" or "blueprint" to success.
    
    CRITICAL RULES:
    1. ONLY output the exact spoken words.
    2. DO NOT include timestamps, sound effects, or visual cues.
    3. DO NOT include emojis.
    """
    script = ""
    for attempt in range(3):
        script = llm.invoke([HumanMessage(content=script_prompt)]).content.strip()
        
        print("🎯 Quality Evaluator (DeepEval): Scoring script...")
        try:
            viral_metric = GEval(
                name="Viral Potential and Authority",
                criteria=SHORTS_CONFIG.get("evaluator", {}).get("criteria", "Determine if the script has a strong hook."),
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                model=GroqDeepEval(ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2))
            )
            test_case = LLMTestCase(input=user_input, actual_output=script)
            viral_metric.measure(test_case)
            score = int(viral_metric.score * 10)
            reason = viral_metric.reason
            print(f"DeepEval Score: {score}/10 | Reason: {reason}")
            
            pass_threshold = SHORTS_CONFIG.get("evaluator", {}).get("pass_threshold", 8)
            if score >= pass_threshold:
                print("✅ Script passed DeepEval standards!")
                break
            else:
                print("⚠️ Script score too low. Revising...")
                script_prompt += f"\n\nFEEDBACK TO IMPROVE: {reason}. Make it punchier and fix this."
        except Exception as e:
            print(f"DeepEval failed ({e}). Proceeding with draft.")
            break

    print(f"\n✅ Final 35s Script:\n{script}\n")

    print("🤖 AI Director Agent: Analyzing script to plan scenes...")
    
    # Ask LLM to generate image prompts
    prompt = f"""
    You are an expert AI Video Director. 
    I am giving you a YouTube Short script. I need you to break it down into exactly 5 visual scenes.
    For each scene, write a highly detailed, cinematic image generation prompt.
    
    The prompts should:
    1. Be photorealistic, futuristic, and high-tech finance themed (8k, highly detailed, cinematic lighting).
    2. Focus on professional success: glowing stock charts, modern luxury trading desks, advanced AI data visualizations, and confident professional traders.
    3. NO SAD FACES, NO DESPAIR, NO CRYING. Use "Confident", "Successful", and "Visionary" expressions.
    4. NOT include any text overlays (the video engine will add text).
    5. Maintain a premium "Dark Mode" aesthetic (deep blues, neon greens, sleek blacks).
    
    Script:
    {script}
    
    Return ONLY a JSON array of 5 strings (the prompts). Example: ["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"]
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Extract JSON from response
    try:
        content = response.content
        # Find JSON array
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != 0:
            prompts = json.loads(content[start:end])
        else:
            raise ValueError("No JSON array found")
            
    except Exception as e:
        print(f"Failed to parse LLM response: {e}. Using fallback scenes.")
        prompts = [
            "A cinematic wide shot of a glowing neon green stock market chart trending upwards in a futuristic dark room, 8k, photorealistic",
            "A confident professional trader in a sleek suit looking at multiple holographic screens with complex financial data, cinematic lighting",
            "A high-tech luxury trading desk with advanced monitors displaying golden bull market signals, 8k resolution, photorealistic",
            "A futuristic AI brain processing streams of financial data and transforming them into gold coins, cyberpunk finance style",
            "A successful person standing on a penthouse balcony at night, looking at a glowing cityscape, symbolizing financial freedom and success"
        ]
        
    print(f"🎬 Generated {len(prompts)} scene prompts:")
    for i, p in enumerate(prompts):
        print(f"  Scene {i+1}: {p}")
        
    print("\n🚀 Initiating Text-to-Video Engine...")
    
    # Run the async video generation
    run_id = int(time.time())
    output_file = f"video_{run_id}.mp4"
    output_path = f"outputs/{output_file}"
    asyncio.run(create_new_video(script, prompts, output_file))
    print(f"\n🎉 Video Generation Complete! Saved to: {output_path}")
    
    # Generate Social Media Metadata
    print("\n--- SOCIAL MEDIA MANAGER ---")
    post_data = generate_instagram_metadata(script)
    caption = post_data.get("full_post", "")
    thumbnail_path = post_data.get("thumbnail_path", "")
    
    # Autonomously Post
    print("\n--- PUBLISHER AGENT ---")
    auto_post = SHORTS_CONFIG.get("publisher", {}).get("auto_post_instagram", False)
    
    if auto_post:
        if caption:
            post_video_to_instagram(output_path, caption, thumbnail_path)
        else:
            print("⚠️ Missing caption. Skipping upload.")
    else:
        print("⏸️ Auto-posting is disabled in shorts_config.yaml. Skipping Instagram upload.")

    print("\n🎉 Process Complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_script = sys.argv[1]
    else:
        user_script = "Don't let emotions ruin your trading account. The fear of missing out, or FOMO, is the number one reason beginners lose their capital. Instead of chasing green candles, wait for the price to retrace to a key order block. Let the market come to you. Discipline is what separates gamblers from professionals."
        
    generate_video_from_script(user_script)
