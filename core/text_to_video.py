import os
import asyncio
import urllib.parse
import requests
from moviepy import (
    ImageClip, AudioFileClip, CompositeVideoClip, TextClip,
    CompositeAudioClip, concatenate_videoclips, vfx
)
import edge_tts
from faster_whisper import WhisperModel
import numpy as np
import random
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def download_image(prompt, output_path):
    """Downloads an image from the free Pollinations AI API based on a prompt."""
    print(f"🎨 Generating image for prompt: {prompt[:50]}...")
    encoded_prompt = urllib.parse.quote(prompt)
    seed = random.randint(0, 999999)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1080&height=1920&nologo=true&seed={seed}"
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

async def generate_audio(script_text, output_path):
    """Generates TTS audio."""
    print("🎙️ Generating Voiceover...")
    communicate = edge_tts.Communicate(script_text, "en-US-AriaNeural", rate="-5%")
    await communicate.save(output_path)
    return output_path

def get_word_timestamps(audio_path):
    print("🧠 Analyzing audio timing for captions...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    words = []
    sentence_ends = []
    for s in segments:
        sentence_ends.append(s.end)
        for w in s.words:
            words.append({'word': w.word.strip(), 'start': w.start, 'end': w.end})
    return words, sentence_ends

def create_captions(words_data, target_w, target_h):
    print("📝 Generating captions...")
    caption_clips = []
    phrase = []
    start_t = None
    
    for w in words_data:
        if start_t is None: start_t = w['start']
        phrase.append(w['word'])
        if len(phrase) >= 4 or w['word'].endswith(('.', '!', '?')):
            end_t = w['end']
            duration = end_t - start_t
            if duration <= 0: continue
            
            # On macOS, use a safe system font path
            font_path = "/System/Library/Fonts/Helvetica.ttc"
            if not os.path.exists(font_path):
                font_path = "Arial" # Fallback
                
            text = " ".join(phrase).upper()
            txt = (TextClip(text=text, font=font_path, font_size=70, 
                            color="white", bg_color="black",
                            method='caption', size=(int(target_w * 0.8), None),
                            text_align='center', vertical_align='center')
                    .with_start(start_t).with_duration(duration)
                    .with_position(('center', target_h * 0.7)))
            caption_clips.append(txt)
            phrase = []
            start_t = None
    return caption_clips

def apply_ken_burns(clip, duration):
    """Applies a subtle zoom-in effect to make static images dynamic."""
    if duration <= 0: return clip
    w, h = clip.size
    scale_factor = 1.0 / duration if duration > 0 else 0
    # Zoom in by 10% over the duration
    clip = clip.resized(lambda t: 1 + 0.1 * (t * scale_factor))
    return clip.cropped(x_center=w/2, y_center=h/2, width=1080, height=1920)

async def create_new_video(script_text, image_prompts, output_filename="ai_generated_short.mp4"):
    # Create a unique scene directory for this specific run to avoid collisions
    run_id = output_filename.split('.')[0].replace('video_', '')
    scene_dir = f"temp/scenes_{run_id}"
    
    if os.path.exists(scene_dir):
        shutil.rmtree(scene_dir)
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Generate Audio
    audio_path = f"temp/audio_{run_id}.mp3"
    await generate_audio(script_text, audio_path)
    vo_audio = AudioFileClip(audio_path)
    total_dur = vo_audio.duration
    
    # 2. Get Timestamps to divide scenes
    words_data, sentence_ends = get_word_timestamps(audio_path)
    
    # 3. Download Images
    scene_clips = []
    num_scenes = len(image_prompts)
    last_t = 0
    
    for i, prompt in enumerate(image_prompts):
        img_path = f"{scene_dir}/scene_{i}.jpg"
        download_image(prompt, img_path)
        
        # Calculate duration for this image based on sentence ends or divide evenly
        if i < len(sentence_ends) and len(sentence_ends) >= num_scenes:
            end_t = sentence_ends[min(i * (len(sentence_ends)//num_scenes), len(sentence_ends)-1)]
        else:
            end_t = total_dur * ((i + 1) / num_scenes)
            
        if i == num_scenes - 1:
            end_t = total_dur
            
        duration = end_t - last_t
        if duration <= 0: duration = 1.0
        
        if os.path.exists(img_path):
            clip = ImageClip(img_path).with_duration(duration).resized(width=1080)
            if clip.h < 1920: clip = clip.resized(height=1920)
            clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=1080, height=1920)
            clip = apply_ken_burns(clip, duration)
            # Add crossfade
            clip = clip.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.5)])
            scene_clips.append(clip)
            
        last_t = end_t

    # 4. Assemble Video
    print("🎬 Assembling scenes...")
    if scene_clips:
        video_base = concatenate_videoclips(scene_clips, method="compose")
        # Ensure video base duration matches audio
        video_base = video_base.subclipped(0, total_dur)
    else:
        print("❌ No scenes generated. Using black background.")
        from moviepy import ColorClip
        video_base = ColorClip(size=(1080, 1920), color=(0,0,0)).with_duration(total_dur)
        
    captions = create_captions(words_data, 1080, 1920)
    
    final_video = CompositeVideoClip([video_base] + captions, size=(1080, 1920)).with_audio(vo_audio)
    
    output_path = f"outputs/{output_filename}"
    print(f"🎥 Rendering final video to {output_path}...")
    final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", threads=8, preset='ultrafast')
    print("✅ Video Generation Complete!")
    return output_path
