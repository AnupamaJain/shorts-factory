import os
import PIL.Image

# 1. The Mac MKL Memory Bypass
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_WARNINGS"] = "0"

# 2. The Pillow 10 Antialias Hack
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from faster_whisper import WhisperModel
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
# Note: In MoviePy 2.x, fx like 'crop' are accessed differently or as methods on the clip.

def transcribe_full_video(audio_path):
    print("🧠 Transcribing audio with Faster-Whisper...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    words_data = []
    for segment in segments:
        if segment.words:
            for word_obj in segment.words:
                words_data.append({
                    'word': word_obj.word,
                    'start': word_obj.start,
                    'end': word_obj.end
                })
    return words_data

def create_stylish_subtitles(words_data, target_w, target_h):
    print("🎨 Generating stylish captions...")
    subtitle_clips = []
    phrase = []
    start_t = None
    color_toggle = False 
    
    for word_info in words_data:
        if start_t is None: 
            start_t = word_info['start']
            
        text = word_info['word'].strip()
        phrase.append(text)
        
        # Build 3-word phrases or break on punctuation
        if len(phrase) >= 3 or text.endswith(('.', '!', '?', ',')):
            end_t = word_info['end']
            rendered_text = " ".join(phrase).upper()
            
            # Alternate colors: Yellow and White for a premium look
            text_color = '#FFD700' if color_toggle else 'white'
            color_toggle = not color_toggle
            
            txt_clip = (TextClip(text=rendered_text, font="Arial", font_size=55, 
                                 color=text_color, stroke_color='black', stroke_width=2,
                                 method='caption', size=(int(target_w * 0.8), None),
                                 margin=(20, 20), text_align='center', vertical_align='center')
                        .with_position(('center', 'center'))
                        .with_start(start_t)
                        .with_duration(end_t - start_t))
            
            # Drop the text to the lower quarter of the screen
            txt_clip = txt_clip.with_position(('center', target_h * 0.75))
            subtitle_clips.append(txt_clip)
            
            phrase = []
            start_t = None
            
    return subtitle_clips

def clean_and_caption(video_path, output_path):
    print(f"🎬 Loading video: {video_path}")
    main_clip = VideoFileClip(video_path)
    
    # --- 1. THE CUT (Remove the NotebookLM Outro) ---
    # We trim the last 3 seconds off the total duration
    safe_duration = main_clip.duration - 3.0
    trimmed_clip = main_clip.subclipped(0, safe_duration)
    target_w, target_h = trimmed_clip.size
    
    # Extract audio for the transcriber
    temp_audio_path = "temp_full_audio.wav"
    trimmed_clip.audio.write_audiofile(temp_audio_path, logger=None)
    
    # --- 3. THE CAPTIONS ---
    words_data = transcribe_full_video(temp_audio_path)
    subtitle_clips = create_stylish_subtitles(words_data, target_w, target_h)
                         
    # --- 4. ASSEMBLE & RENDER ---
    print("🎥 Assembling final cleaned video...")
    final_video = CompositeVideoClip([
        trimmed_clip, 
        *subtitle_clips
    ], size=(target_w, target_h)).with_audio(trimmed_clip.audio)
    
    final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")
    
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    print(f"\n🎉 Cleaned video saved to: {output_path}")

if __name__ == "__main__":
    input_video = "The_Seller_s_Vow.mp4"
    output_video = "The_Sellers_Vow_Cleaned.mp4"
    
    if os.path.exists(input_video):
        clean_and_caption(input_video, output_video)
    else:
        print(f"❌ ERROR: Could not find {input_video} in this directory.")