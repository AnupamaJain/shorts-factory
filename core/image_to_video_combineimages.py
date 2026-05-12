import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
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
# this combine images and add voice over and add some zoom effects and caption, taking multiple from assets
# --- PRO STORY ENGINE CONFIG ---
EMOTION_WORDS = {
    "fear": ["loss", "crash", "risk", "fear", "panic", "threat", "danger", "scared"],
    "greed": ["profit", "win", "money", "gain", "wealth", "million", "success"],
    "mistake": ["mistake", "wrong", "error", "fail", "failed", "blunder"],
    "confidence": ["strategy", "edge", "discipline", "mastery", "focus", "plan"],
    "neutral": []
}

# --- CONFIG ---
IMAGE_DIR = "video_images"
VOICEOVER_PATH = os.path.join(IMAGE_DIR, "ElevenLabs_2026-04-24T15_31_07_Alexandra - Conversational and Natural_pvc_sp100_s40_sb99_se38_b_m2.mp3")
OUTPUT_DIR = "video_generated"
BGM_PATH = "bgm.mp3" # Placeholder - change if you have a specific file
TARGET_SIZE = (1920, 1080)  # Landscape 16:9
FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_number(filename):
    """Extracts the sequence number from filenames like 'Woman_..._1.jpeg'."""
    match = re.search(r'(\d+)\.jpe?g$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Try finding any number if the above fails
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def detect_emotion(text):
    text = text.lower()
    for emotion, words in EMOTION_WORDS.items():
        for w in words:
            if w in text:
                return emotion
    return "neutral"

def apply_emotion_motion(clip, duration, emotion):
    """Applies motion based on the psychological intent of the scene."""
    w, h = clip.size
    
    if emotion == "fear":
        # Fear: Slight shake + aggressive zoom in
        # We use a lambda for position to create the 'shake'
        clip = clip.resized(lambda t: 1 + 0.12 * (t / duration))
        clip = clip.with_position(lambda t: (
            int(w/2 - TARGET_SIZE[0]/2 + 8 * np.sin(20 * t)), 
            int(h/2 - TARGET_SIZE[1]/2 + 8 * np.cos(20 * t))
        ), relative=False)

    elif emotion == "greed":
        # Greed: Slow, aspirational zoom-in
        clip = clip.resized(lambda t: 1 + 0.08 * (t / duration))
        clip = clip.with_position(('center', 'center'))

    elif emotion == "mistake":
        # Mistake: Zoom-out (creates a sense of realization/detachment)
        clip = clip.resized(lambda t: 1.15 - 0.1 * (t / duration))
        clip = clip.with_position(('center', 'center'))

    elif emotion == "confidence":
        # Confidence: Very stable, smooth crawling zoom
        clip = clip.resized(lambda t: 1 + 0.03 * (t / duration))
        clip = clip.with_position(('center', 'center'))

    else:
        # Neutral: Standard cinematic Ken Burns in/out
        clip = clip.resized(lambda t: 1 + 0.05 * (t / duration))
        clip = clip.with_position(('center', 'center'))

    # Final crop to ensure it stays 16:9 during motion
    return clip.cropped(x_center=w/2, y_center=h/2, width=TARGET_SIZE[0], height=TARGET_SIZE[1])

def create_cinematic_image_clip(img_path, duration, emotion):
    """Creates an image clip with emotion-based motion engine."""
    clip = ImageClip(img_path).with_duration(duration)
    
    # Resize to cover the horizontal screen
    clip = clip.resized(width=TARGET_SIZE[0])
    if clip.h < TARGET_SIZE[1]:
        clip = clip.resized(height=TARGET_SIZE[1])
    
    # Center and apply emotion motion
    clip = apply_emotion_motion(clip, duration, emotion)
    
    return clip

def get_sentence_timed_segments(audio_path):
    """Transcribes audio and returns segments grouped by sentences."""
    print("🧠 Transcribing voiceover for timing analysis...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    # We use segments instead of words for sentence-level timing
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    sentence_breaks = []
    all_words = []
    
    for segment in segments:
        segment_end = segment.end
        sentence_breaks.append(segment_end)
        for word in segment.words:
            all_words.append({
                'word': word.word.strip(),
                'start': word.start,
                'end': word.end
            })
            
    return all_words, sentence_breaks

def style_caption(text):
    """Applies dynamic sizing and coloring for power words."""
    text_upper = text.upper()
    if any(word in text_upper for word in ["LOSS", "TRUTH", "MISTAKE", "WRONG", "FAIL"]):
        return {"color": "#FF3B3B", "size": 95}  # Error/Fear Red
    elif any(word in text_upper for word in ["PROFIT", "WIN", "MONEY", "GAIN"]):
        return {"color": "#00FFAA", "size": 90}  # Profit Green
    else:
        return {"color": "white", "size": 75}

def create_stylish_captions(words_data, target_w, target_h):
    """Generates stylish captions with emotion-based styling."""
    print("🎨 Generating storytelling captions...")
    caption_clips = []
    phrase = []
    start_t = None
    
    for word_info in words_data:
        if start_t is None: start_t = word_info['start']
        phrase.append(word_info['word'])
        
        if len(phrase) >= 3 or word_info['word'].endswith(('.', '!', '?')):
            end_t = word_info['end']
            text = " ".join(phrase).upper()
            
            style = style_caption(text)
            
            txt = (TextClip(text=text, font="Arial", font_size=style['size'], 
                            color=style['color'], stroke_color="black", stroke_width=2.5,
                            method='caption', size=(int(target_w * 0.85), None),
                            margin=(20, 40), text_align='center', vertical_align='center')
                    .with_start(start_t)
                    .with_duration(end_t - start_t)
                    .with_position(('center', target_h * 0.75)))
            
            caption_clips.append(txt)
            phrase = []
            start_t = None
            
    return caption_clips

def generate_video():
    print("🎬 Starting PRO Story Engine Video Generation...")
    
    # 1. Get and sort images
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    images.sort(key=extract_number)
    
    if not images:
        print("❌ No images found in video_images folder!")
        return

    # 2. Transcription & Timing Analysis
    words_data, sentence_breaks = get_sentence_timed_segments(VOICEOVER_PATH)
    
    vo_audio = AudioFileClip(VOICEOVER_PATH)
    total_duration = vo_audio.duration
    
    # 3. Build Emotion Timeline
    print("🧠 Analyzing emotional tone of individual scenes...")
    emotion_timeline = []
    for word in words_data:
        emotion = detect_emotion(word['word'])
        if emotion != "neutral":
            emotion_timeline.append({"t": word['start'], "emotion": emotion})

    def get_dominant_emotion(start, end):
        relevant = [e['emotion'] for e in emotion_timeline if start <= e['t'] <= end]
        if not relevant: return "neutral"
        return max(set(relevant), key=relevant.count)

    # 4. Distribute Images across Sentence Breaks
    print(f"🎙️ Distributing {len(images)} images across {len(sentence_breaks)} narration segments...")
    
    image_clips = []
    num_images = len(images)
    num_segments = len(sentence_breaks)
    
    image_times = []
    if num_segments >= num_images:
        stride = num_segments / num_images
        for i in range(num_images):
            idx = int((i + 1) * stride) - 1
            idx = min(idx, num_segments - 1)
            image_times.append(sentence_breaks[idx])
    else:
        image_times = [(i + 1) * (total_duration / num_images) for i in range(num_images)]

    image_times[-1] = total_duration

    # 5. Create Image Sequence with Emotional Intelligence
    print("🖼️ Creating storytelling image sequence...")
    last_t = 0
    final_clips_for_music = [] # To sync volume later
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        duration = image_times[i] - last_t
        if duration <= 0: duration = 0.1
        
        scene_emotion = get_dominant_emotion(last_t, image_times[i])
        clip = create_cinematic_image_clip(img_path, duration, scene_emotion)
        image_clips.append(clip)
        final_clips_for_music.append({"start": last_t, "end": image_times[i], "emotion": scene_emotion})
        last_t = image_times[i]
    
    main_video = concatenate_videoclips(image_clips, method="compose")
    
    # 6. Captions
    caption_clips = create_stylish_captions(words_data, TARGET_SIZE[0], TARGET_SIZE[1])
    
    # 7. Audio: Dynamic BGM Energy Sync + Voiceover
    audio_layers = [vo_audio]
    if os.path.exists(BGM_PATH):
        # We create a volume-mapped BGM track
        bgm = AudioFileClip(BGM_PATH).with_duration(total_duration).audio_fadeout(3)
        
        # Mapping emotion to volume
        def volume_map(t):
            for scene in final_clips_for_music:
                if scene['start'] <= t <= scene['end']:
                    emotion = scene['emotion']
                    if emotion == "fear": return 0.19
                    if emotion == "greed": return 0.17
                    if emotion == "confidence": return 0.14
                    return 0.12
            return 0.12
            
        bgm = bgm.with_volume_scaled(volume_map)
        audio_layers.append(bgm)
        
    final_audio = CompositeAudioClip(audio_layers)
    
    # 8. Assembly
    print("🎥 Assembling final intelligent cinematic sequence...")
    final_video = CompositeVideoClip([main_video] + caption_clips, size=TARGET_SIZE).with_audio(final_audio)
    
    output_filename = os.path.join(OUTPUT_DIR, "pro_story_cinematic.mp4")
    final_video.write_videofile(output_filename, fps=FPS, codec="libx264", audio_codec="aac")
    
    print(f"\n🎉 BOOM! Your Pro Storytelling Engine has completed the render: {output_filename}")

if __name__ == "__main__":
    generate_video()
