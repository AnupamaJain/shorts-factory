import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx.all import crop, resize
from faster_whisper import WhisperModel

# --- CONFIG ---
model = WhisperModel("base", compute_type="int8")

def transcribe(video_path):
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    words = []
    for segment in segments:
        for word in segment.words:
            words.append({
                "word": word.word,
                "start": word.start,
                "end": word.end
            })
    return words

def to_vertical(clip):
    # This assumes the input is 16:9 landscape (1280x720) and crops to 9:16 vertical (1080x1920 or similar)
    clip = resize(clip, height=1920)
    return crop(clip, x_center=clip.w//2, width=1080, height=1920)

def create_dynamic_subtitles(words_data, clip_start, target_w, target_h):
    subtitle_clips = []
    pop_timestamps = [] # 🎵 Track when we need a 'pop' sound
    phrase = []
    start_t = None
    color_toggle = False 
    
    for word_info in words_data:
        if start_t is None: 
            start_t = word_info['start']
            
        text = word_info['word'].strip()
        phrase.append(text)
        
        if len(phrase) >= 3 or text.endswith(('.', '!', '?', ',')):
            end_t = word_info['end']
            rendered_text = " ".join(phrase).upper()
            
            text_color = '#00FF00' if color_toggle else 'white'
            
            if color_toggle:
                pop_timestamps.append(start_t - clip_start)
                
            color_toggle = not color_toggle
            
            # THE FIX: Added `size` and `method='caption'`
            # This forces the text to stay within 85% of the screen width and auto-wrap to a new line
            txt_clip = (TextClip(rendered_text, font="Arial", fontsize=85, 
                                 color=text_color, stroke_color='black', stroke_width=4.5,
                                 method='caption', size=(int(target_w * 0.85), None))
                        .set_position(('center', 'center')) # 'center' vertically helps if text wraps to 2 lines
                        .set_start(start_t - clip_start)
                        .set_duration(end_t - start_t))
            
            # We explicitly set the Y position *after* generation so multi-line text stays anchored properly
            txt_clip = txt_clip.set_position(('center', target_h - 600))
            
            subtitle_clips.append(txt_clip)
            phrase = []
            start_t = None
            
    return subtitle_clips, pop_timestamps

def generate_viral_shorts(video_file):
    print(f"🎬 Processing: {video_file}")
    
    # 1. Load Video
    video = VideoFileClip(video_file)
    
    # 2. Transcribe
    print("🎙️ Transcribing...")
    words = transcribe(video_file)
    
    # 3. Transform to Vertical
    print("📱 Converting to vertical...")
    vertical_video = to_vertical(video)
    target_w, target_h = vertical_video.size # 1080, 1920
    
    # 4. Generate Subtitles
    print("✍️ Adding dynamic subtitles...")
    subs, pops = create_dynamic_subtitles(words, 0, target_w, target_h)
    
    # 5. Composite
    print("🎞️ Compositing final video...")
    final = CompositeVideoClip([vertical_video] + subs)
    
    # 6. Save
    output_name = f"factory_output/viral_{os.path.basename(video_file)}"
    os.makedirs("factory_output", exist_ok=True)
    final.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac")
    print(f"✅ SUCCESS: Saved to {output_name}")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    video_file = "The_Seller_s_Vow.mp4"
    
    # Check if the file exists before trying to run
    if os.path.exists(video_file):
        generate_viral_shorts(video_file)
    else:
        # If you see this, your video isn't in the same folder as the script
        print(f"❌ ERROR: Could not find {video_file} in this directory.")