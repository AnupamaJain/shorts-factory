import os
import asyncio
import edge_tts
import numpy as np
from moviepy import (
    VideoFileClip, 
    TextClip, 
    CompositeVideoClip, 
    ColorClip, 
    ImageClip, 
    AudioFileClip, 
    CompositeAudioClip, 
    concatenate_videoclips,
    vfx
)
from faster_whisper import WhisperModel
import re

# --- CONFIG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_WARNINGS"] = "0"

IMAGE_DIR = "assets/video_images"
OUTPUT_VIDEO = "outputs/The_Sellers_Vow_Poetic.mp4"
BGM_PATH = "assets/bgm/bgm.mp3"
TARGET_SIZE = (1920, 1080)
FPS = 30

# The Story Script (36 Lines for 36 Images) - POETIC STYLE
STORY_LINES = [
    "In the garden of numbers, many come with a single spark of hope.",
    "Yet the shadows grow long when the questions outweigh the light.",
    "A soul divided, standing where the dream meets the mirror's edge.",
    "Searching the silence for a whisper of truth in a sea of curves.",
    "Reaching for the sun, assuming every rise is our salvation.",
    "Until the storm reminds us that the sky belongs to no one.",
    "There is a weight in the falling—a heavy silence that follows the rain.",
    "We cast our stones at fate, forgetting the ripples are our own.",
    "But look—see the golden thread you missed while fearing the loom.",
    "The sight of the shore is only for those who stop fighting the tide.",
    "To stand tall in the wreckage is to finally see the horizon.",
    "Data is no longer a ghost; it is the ink with which we write our peace.",
    "Release the anchor of old sorrows; your hands were meant for more.",
    "Shed the skin of the seeker; emerge as the anchor... the Seller.",
    "Mastery is not in the flame, but in the breath that keeps it steady.",
    "Let the ego fall like autumn leaves—quiet, necessary, and final.",
    "In the stillness between the breaths, the path becomes a road.",
    "Face the tempest with a heart made of stone and a mind made of light.",
    "Turn your gaze to the stars—where probability is the only law.",
    "Time does not take from us; it builds our cathedral, stone by stone.",
    "A sudden glow—the moment the lock yields to the right key.",
    "We stack the bricks of patience, building a house that cannot fall.",
    "The screen is but a mirror reflecting the calm of a steady hand.",
    "Two paths through the woods: the fevered run or the silent walk.",
    "To choose the silent walk is to walk alone... yet arrive first.",
    "Let the roar of the crowd be but a hum in the distance.",
    "Close the door on the hollow chase; you are home now.",
    "The vow is not a chain; it is the ground beneath your feet.",
    "Holding the secret of the seasons—to wait is to win.",
    "Faith is not for the outcome; it is for the bridge you built.",
    "Sowing the seeds where the earth is deep and the profit is slow.",
    "The light of a quiet mind is brighter than any winning candle.",
    "Looking out to the great expanse—unbounded, yet precise.",
    "The shift is complete; the chaos has found its order in you.",
    "Serene. Solemn. Sovereign.",
    "With a gentle breath, the Vow is signed... and the master is born."
]

EMOTION_WORDS = {
    "fear": ["shadow", "storm", "weight", "wreckage", "fate", "divided", "fear", "panic"],
    "greed": ["sun", "gold", "golden", "rise", "seeds", "horizon", "win"],
    "mistake": ["sorrows", "ego", "leaves", "hollow", "wrong"],
    "confidence": ["anchor", "steady", "road", "stone", "stars", "law", "cathedral", "vow", "mastery", "precise", "sovereign"],
    "neutral": []
}

async def generate_voiceover(text_list, output_path):
    print("🎙️ Generating poetic female voiceover...")
    full_text = " ".join(text_list)
    # Using 'en-US-AvaNeural' for a storytelling, lyrical female voice
    communicate = edge_tts.Communicate(full_text, "en-US-AvaNeural", rate="-8%", pitch="+1Hz")
    await communicate.save(output_path)
    return output_path

def detect_emotion(text):
    text = text.lower()
    for emotion, words in EMOTION_WORDS.items():
        for w in words:
            if w in text: return emotion
    return "neutral"

def apply_emotion_motion(clip, duration, emotion):
    """Applies motion logic with safety checks for zero duration."""
    if duration <= 0:
        return clip
        
    w, h = clip.size
    # Pre-calculate factor to avoid NaN in lambda
    scale_factor = 1.0 / duration if duration > 0 else 0

    if emotion == "fear":
        clip = clip.resized(lambda t: 1 + 0.12 * (t * scale_factor))
        clip = clip.with_position(lambda t: (
            int(w/2 - TARGET_SIZE[0]/2 + 4 * np.sin(15 * t)), 
            int(h/2 - TARGET_SIZE[1]/2 + 4 * np.cos(15 * t))
        ), relative=False)
    elif emotion == "greed":
        clip = clip.resized(lambda t: 1 + 0.08 * (t * scale_factor))
    elif emotion == "mistake":
        clip = clip.resized(lambda t: 1.12 - 0.08 * (t * scale_factor))
    elif emotion == "confidence":
        clip = clip.resized(lambda t: 1 + 0.02 * (t * scale_factor))
    else:
        clip = clip.resized(lambda t: 1 + 0.05 * (t * scale_factor))
    
    return clip.cropped(x_center=w/2, y_center=h/2, width=TARGET_SIZE[0], height=TARGET_SIZE[1])

def get_sentence_timestamps(audio_path):
    print("🧠 Analyzing voiceover timing with Whisper...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    all_words = []
    sentence_ends = []
    
    for segment in segments:
        sentence_ends.append(segment.end)
        for word in segment.words:
            all_words.append({'word': word.word.strip(), 'start': word.start, 'end': word.end})
            
    return all_words, sentence_ends

def style_caption(text):
    text_u = text.upper()
    if any(w in text_u for w in ["STORM", "FEAR", "SHADOW", "WEIGHT", "WRECKAGE"]): return {"color": "#FF3B3B", "size": 95}
    if any(w in text_u for w in ["SUN", "GOLD", "WIN", "RISE", "SEED"]): return {"color": "#00FFAA", "size": 90}
    return {"color": "white", "size": 75}

def create_captions(words, target_w, target_h):
    clips = []
    current_phrase = []
    start_t = None
    for w in words:
        if start_t is None: start_t = w['start']
        current_phrase.append(w['word'])
        if len(current_phrase) >= 4 or w['word'].endswith(('.', '!', '?')):
            end_t = w['end']
            text = " ".join(current_phrase).upper()
            style = style_caption(text)
            txt = (TextClip(text=text, font="Arial", font_size=style['size'], 
                            color=style['color'], stroke_color="black", stroke_width=2,
                            method='caption', size=(int(target_w * 0.8), None),
                            margin=(20, 40), text_align='center', vertical_align='center')
                    .with_start(start_t).with_duration(end_t - start_t)
                    .with_position(('center', target_h * 0.8)))
            clips.append(txt)
            current_phrase = []
            start_t = None
    return clips

def extract_num(f):
    m = re.search(r'(\d+)\.jpe?g$', f, re.IGNORECASE)
    if m: return int(m.group(1))
    nums = re.findall(r'\d+', f)
    return int(nums[-1]) if nums else 0

async def main():
    os.makedirs("video_generated", exist_ok=True)
    voice_file = "poetic_voice.mp3"
    await generate_voiceover(STORY_LINES, voice_file)
    
    words_data, sentence_ends = get_sentence_timestamps(voice_file)
    vo_audio = AudioFileClip(voice_file)
    total_dur = vo_audio.duration

    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    images.sort(key=extract_num)
    
    image_times = []
    num_images = len(images)
    num_sentences = len(sentence_ends)
    
    for i in range(num_images):
        idx = int((i + 1) * (num_sentences / num_images)) - 1
        idx = max(0, min(idx, num_sentences - 1))
        image_times.append(sentence_ends[idx])
    image_times[-1] = total_dur

    print("🖼️ Building poetic scenes...")
    image_clips = []
    last_t = 0
    energy_map = []
    for i, img_name in enumerate(images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        duration = image_times[i] - last_t
        emotion = detect_emotion(STORY_LINES[i % len(STORY_LINES)])
        
        clip_base = ImageClip(img_path).with_duration(duration).resized(width=TARGET_SIZE[0])
        if clip_base.h < TARGET_SIZE[1]: clip_base = clip_base.resized(height=TARGET_SIZE[1])
        clip_base = apply_emotion_motion(clip_base, duration, emotion)
        
        # Add smooth cinematic fade between segments
        clip_base = clip_base.with_effects([vfx.FadeIn(0.4), vfx.FadeOut(0.4)])
        
        image_clips.append(clip_base)
        energy_map.append({"start": last_t, "end": image_times[i], "emotion": emotion})
        last_t = image_times[i]

    main_video = concatenate_videoclips(image_clips, method="compose")
    caption_clips = create_captions(words_data, TARGET_SIZE[0], TARGET_SIZE[1])
    
    audio_layers = [vo_audio]
    if os.path.exists(BGM_PATH):
        bgm = AudioFileClip(BGM_PATH).with_duration(total_dur).audio_fadeout(3)
        def vol(t):
            for e in energy_map:
                if e['start'] <= t <= e['end']:
                    if e['emotion'] == "fear": return 0.18
                    return 0.1
            return 0.1
        audio_layers.append(bgm.with_volume_scaled(vol))
    
    final_video = CompositeVideoClip([main_video] + caption_clips, size=TARGET_SIZE)\
        .with_audio(CompositeAudioClip(audio_layers))
    
    print("🎥 Rendering Poetic Masterpiece...")
    final_video.write_videofile(OUTPUT_VIDEO, fps=FPS, codec="libx264", audio_codec="aac")
    print(f"✅ Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    asyncio.run(main())
