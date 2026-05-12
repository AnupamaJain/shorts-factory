import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from moviepy.video.fx.all import crop, resize
from faster_whisper import WhisperModel



def transcribe_and_chunk(audio_path, target_duration=30, max_shorts=5):
    print("🧠 Transcribing audio with Faster-Whisper...")
    # Using CPU mode for Mac compatibility
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    shorts_data = []
    current_short = {'start': 0, 'end': 0, 'words': []}
    
    for segment in segments:
        if segment.words:
            for word_obj in segment.words:
                # faster-whisper returns objects, we convert to dicts for our logic
                current_short['words'].append({
                    'word': word_obj.word,
                    'start': word_obj.start,
                    'end': word_obj.end
                })
            current_short['end'] = segment.end
            
            if (current_short['end'] - current_short['start']) >= target_duration:
                shorts_data.append(current_short)
                current_short = {'start': segment.end, 'end': segment.end, 'words': []}
                if len(shorts_data) == max_shorts:
                    break
                    
    return shorts_data

def create_dynamic_subtitles(words_data, clip_start, target_w, target_h):
    subtitle_clips = []
    pop_timestamps = [] 
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
            
            # MoviePy 2.x syntax
            txt_clip = (TextClip(text=rendered_text, font="Arial", font_size=85, 
                                 color=text_color, stroke_color='black', stroke_width=4.5,
                                 method='caption', size=(int(target_w * 0.85), None))
                        .with_position(('center', 'center'))
                        .with_start(start_t - clip_start)
                        .with_duration(end_t - start_t))
            
            txt_clip = txt_clip.with_position(('center', target_h - 600))
            subtitle_clips.append(txt_clip)
            
            phrase = []
            start_t = None
            
    return subtitle_clips, pop_timestamps

def generate_viral_shorts(video_path):
    print("🎬 Initializing viral batch process...")
    main_clip = VideoFileClip(video_path)
    target_w, target_h = 1080, 1920
    
    temp_audio_path = "full_temp_audio.wav"
    main_clip.audio.write_audiofile(temp_audio_path, logger=None)
    
    shorts_data = transcribe_and_chunk(temp_audio_path, target_duration=30, max_shorts=5)
    
    for i, short_meta in enumerate(shorts_data):
        print(f"\n✂️ Generating Short {i+1} (From {short_meta['start']:.1f}s to {short_meta['end']:.1f}s)...")
        
        clip = main_clip.subclipped(short_meta['start'], short_meta['end'])
        clip_duration = clip.duration
        
        # True 9:16 Fullscreen using MoviePy 2.x syntax
        clip_resized = resize(clip, height=target_h)
        vertical_clip = crop(clip_resized, x_center=clip_resized.w/2, y_center=target_h/2, 
                             width=target_w, height=target_h)
        
        # --- 1. VISUAL INSIGHT LOGIC ---
        insight_start = clip_duration * 0.60
        insight_end = min(insight_start + 3, clip_duration)
        
        part1 = vertical_clip.subclip(0, insight_start)
        part2_zoom = crop(resize(vertical_clip.subclip(insight_start, insight_end), 1.15),
                          x_center=(target_w*1.15)/2, y_center=(target_h*1.15)/2, 
                          width=target_w, height=target_h)
        
        highlight_box = (ColorClip(size=(target_w - 100, target_h - 600), color=(255, 255, 0))
                         .set_opacity(0.3)
                         .set_position('center')
                         .set_start(insight_start)
                         .set_duration(insight_end - insight_start))
        
        part3 = vertical_clip.subclip(insight_end, clip_duration)
        
        dynamic_video = CompositeVideoClip([
            part1.set_start(0),
            part2_zoom.set_start(insight_start),
            highlight_box,
            part3.set_start(insight_end)
        ], size=(target_w, target_h))

        # --- 2. SUBTITLES & AUDIO LOGIC ---
        subtitle_clips, pop_timestamps = create_dynamic_subtitles(short_meta['words'], short_meta['start'], target_w, target_h)
        audio_layers = [dynamic_video.audio]
        
        if os.path.exists("whoosh.mp3"):
            whoosh = AudioFileClip("whoosh.mp3").set_start(insight_start)
            audio_layers.append(whoosh)
            
        if os.path.exists("pop.mp3"):
            for pop_time in pop_timestamps:
                pop = AudioFileClip("pop.mp3").set_start(pop_time)
                audio_layers.append(pop)
                
        final_audio = CompositeAudioClip(audio_layers)
        final_video_assembled = CompositeVideoClip([dynamic_video, *subtitle_clips], size=(target_w, target_h)).set_audio(final_audio)

        # --- 3. THE PERFECT LOOP HACK ---
        print("🔁 Slicing timeline for the Perfect Loop...")
        if len(short_meta['words']) > 0:
            first_word_duration = short_meta['words'][0]['end'] - short_meta['start']
            loop_main_body = final_video_assembled.subclip(first_word_duration, clip_duration)
            loop_tail = final_video_assembled.subclip(0, first_word_duration)
            viral_loop_video = concatenate_videoclips([loop_main_body, loop_tail])
        else:
            viral_loop_video = final_video_assembled 

        # --- 4. RENDER ---
        output_filename = f"Viral_Short_{i+1}.mp4"
        viral_loop_video.write_videofile(output_filename, fps=30, codec="libx264", audio_codec="aac", bitrate="5000k")
        print(f"✅ Saved: {output_filename}")

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    print("\n🎉 Algorithm-ready Shorts generated successfully!")

if __name__ == "__main__":
    video_file = "The_Seller_s_Vow.mp4"
    if os.path.exists(video_file):
        generate_viral_shorts(video_file)
    else:
        print(f"❌ ERROR: Could not find {video_file} in this directory.")