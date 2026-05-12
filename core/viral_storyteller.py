import os
import asyncio
import edge_tts
import numpy as np
import yaml
from moviepy import (
    VideoFileClip, 
    TextClip, 
    ImageClip,
    CompositeVideoClip, 
    ColorClip, 
    AudioFileClip, 
    CompositeAudioClip, 
    vfx
)
from faster_whisper import WhisperModel
import re

# --- STABILITY CONFIG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_WARNINGS"] = "0"

def load_config(config_path="core/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def detect_emotion(text):
    text = text.lower()
    if any(w in text for w in ["debt", "risk", "danger", "trap", "nightmare", "fraud"]):
        return "fear"
    if any(w in text for w in ["profit", "win", "fortune", "wealth", "gold", "treasure"]):
        return "greed"
    return "neutral"

async def generate_voiceover(text, conf, output_path):
    print(f"🎙️ Generating AI voiceover using {conf['voice']}...")
    communicate = edge_tts.Communicate(
        text, 
        conf['voice'], 
        rate=conf['rate'], 
        pitch=conf['pitch']
    )
    await communicate.save(output_path)
    return output_path

def generate_cloned_voiceover(text, conf, output_path):
    print(f"🧬 Booting AI Voice Cloning (XTTS_v2) using reference: {conf['voice_clone_reference']}...")
    # Lazy import to avoid loading heavy Torch models if cloning is off
    from TTS.api import TTS
    import torch
    
    # Initialize the XTTS model (downloads automatically if first time)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    print("🎙️ Synthesizing cloned voiceover...")
    tts.tts_to_file(
        text=text, 
        speaker_wav=conf['voice_clone_reference'], 
        language="en", 
        file_path=output_path
    )
    return output_path

def get_word_timestamps(audio_path):
    print("🧠 Analyzing audio timing for captions...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    words = []
    for s in segments:
        for w in s.words:
            words.append({'word': w.word.strip(), 'start': w.start, 'end': w.end})
    return words

def extract_full_text(audio_path):
    print("📝 Auto-extracting full script text from audio...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    text = " ".join([s.text.strip() for s in segments])
    return text


async def main():
    # Load Config
    cfg = load_config()
    v_cfg = cfg['video']
    a_cfg = cfg['audio']
    n_cfg = cfg['narrator']
    e_cfg = cfg['editing']
    c_cfg = cfg['captions']

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # 1. Start with Visuals
    print(f"🎬 Loading source video: {v_cfg['source']}")
    source = VideoFileClip(v_cfg['source'])
    
    # 2. Decide Audio source
    voice_file = "temp/working_audio.mp3"
    
    if n_cfg.get('external_audio_path', "") and os.path.exists(n_cfg['external_audio_path']):
        ext_audio = n_cfg['external_audio_path']
        print(f"🔈 Using EXTERNAL audio from: {ext_audio}")
        if ext_audio.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            ext_clip = VideoFileClip(ext_audio)
            vo_audio = ext_clip.audio
            vo_audio.write_audiofile(voice_file, logger=None)
        else:
            vo_audio = AudioFileClip(ext_audio)
            vo_audio.write_audiofile(voice_file, logger=None)
    elif n_cfg.get('use_original_audio', False):
        print("🔈 Using ORIGINAL audio (Auto-Extracting Script from Video)...")
        source.audio.write_audiofile(voice_file, logger=None)
        vo_audio = source.audio
    else:
        # AI TTS Mode
        if n_cfg.get('auto_extract_script', True):
            print("📝 Auto-extracting script from source video for AI TTS...")
            temp_source_audio = "temp/source_audio.mp3"
            source.audio.write_audiofile(temp_source_audio, logger=None)
            script_text = extract_full_text(temp_source_audio)
            
            # Save it for reference
            script_file_path = n_cfg.get('script_file', 'temp/extracted_script.txt')
            with open(script_file_path, "w") as f:
                f.write(script_text)
            print(f"✅ Auto-extracted script saved to {script_file_path}")
        else:
            print(f"🎙️ AI TTS Mode: Reading script from {n_cfg['script_file']}")
            with open(n_cfg['script_file'], "r") as f:
                script_text = f.read()
                
        # Generate Voiceover
        if n_cfg.get('use_voice_cloning', False):
            # Blocking call for Coqui TTS
            generate_cloned_voiceover(script_text, n_cfg, voice_file)
        else:
            await generate_voiceover(script_text, n_cfg, voice_file)
            
        vo_audio = AudioFileClip(voice_file)

    # 3. Transcription
    words_data = get_word_timestamps(voice_file)
    total_dur = vo_audio.duration

    # 4. Filter & Crop Visuals
    print("✂️ Processing visuals...")
    crop_height = int(source.h * (1.0 - e_cfg['bottom_crop_percent']))
    clean_source = source.cropped(y2=crop_height).subclipped(0, source.duration - e_cfg['trim_outro_seconds'])
    
    # Proportional Scaling
    if e_cfg.get('maintain_original_resolution', False):
        print("🔍 Maintaining original resolution (skipping proportional scaling)")
        target_w, target_h = clean_source.w, clean_source.h
    else:
        target_w, target_h = v_cfg['resolution']
        if clean_source.w / clean_source.h > target_w / target_h:
            clean_source = clean_source.resized(height=target_h)
        else:
            clean_source = clean_source.resized(width=target_w)
        
        clean_source = clean_source.cropped(
            x_center=int(clean_source.w / 2), y_center=int(clean_source.h / 2), 
            width=target_w, height=target_h
        )
    
    # 5. Background Strategy
    print("🎭 Mapping visual flow (Time-stretching to match Audio)...")
    video_base = clean_source.with_effects([vfx.MultiplySpeed(final_duration=total_dur)])
    
    # 6. SFX Layers
    sfx_layers = []
    if a_cfg.get('enable_sfx', True):
        phrase = []
        for i, w_info in enumerate(words_data):
            phrase.append(w_info['word'])
            if len(phrase) >= 10 or w_info['word'].endswith(('.', '!', '?')):
                text_seg = " ".join(phrase).lower()
                if any(w in text_seg for w in ["profit", "money", "bagger"]):
                    if os.path.exists(a_cfg['sfx_coin']):
                        sfx = AudioFileClip(a_cfg['sfx_coin']).with_start(w_info['start']).with_volume_scaled(a_cfg['sfx_volume'])
                        sfx_layers.append(sfx)
                if any(w in text_seg for w in ["risk", "trap", "however"]):
                    if os.path.exists(a_cfg['sfx_whoosh']):
                        sfx = AudioFileClip(a_cfg['sfx_whoosh']).with_start(w_info['start']).with_volume_scaled(a_cfg['sfx_volume'] * 0.6)
                        sfx_layers.append(sfx)
                if any(w in text_seg for w in ["report", "analysis", "data", "screener", "annual"]):
                    if os.path.exists(a_cfg.get('sfx_paper', '')):
                        sfx = AudioFileClip(a_cfg['sfx_paper']).with_start(w_info['start']).with_volume_scaled(a_cfg['sfx_volume'] * 0.8)
                        sfx_layers.append(sfx)
                if any(w in text_seg for w in ["years", "time", "decade", "long", "wait", "history"]):
                    if os.path.exists(a_cfg.get('sfx_clock', '')):
                        sfx = AudioFileClip(a_cfg['sfx_clock']).with_start(w_info['start']).with_volume_scaled(a_cfg['sfx_volume'] * 0.7)
                        sfx_layers.append(sfx)
                phrase = []
    else:
        print("⏭️ SFX disabled in config, skipping sound effects.")
            
    # 7. Captions
    caption_clips = []
    if c_cfg.get('enable_captions', True):
        print("🎨 Adding stylish captions...")
        phrase = []
        start_t = None
        
        for w in words_data:
            if start_t is None: start_t = w['start']
            phrase.append(w['word'])
            if len(phrase) >= 4 or w['word'].endswith(('.', '!', '?')):
                end_t = w['end']
                duration = end_t - start_t
                if duration <= 0: continue
                
                text = " ".join(phrase).upper()
                color = c_cfg['color_danger'] if detect_emotion(text) == "fear" else c_cfg['color_normal']
                
                txt = (TextClip(text=text, font=c_cfg['font'], font_size=c_cfg['size'], 
                                color=color, stroke_color="black", stroke_width=2,
                                method='caption', size=(int(target_w * 0.9), None),
                                text_align='center', vertical_align='center', margin=tuple(c_cfg['margin']))
                        .with_start(start_t).with_duration(duration)
                        .with_position(('center', target_h * 0.85)))
                caption_clips.append(txt)
                phrase = []
                start_t = None
    else:
        print("⏭️ Captions disabled in config, skipping text rendering.")
            
    # 7.5 Visual Overlays (B-Roll)
    broll_clips = []
    if c_cfg.get('enable_broll', True):
        print("🖼️ Adding visual pattern interrupts (B-Roll)...")
        os.makedirs("assets/broll", exist_ok=True)
        phrase = []
        for w_info in words_data:
            phrase.append(w_info['word'])
            if len(phrase) >= 10 or w_info['word'].endswith(('.', '!', '?')):
                text_seg = " ".join(phrase).lower()
                overlay_path = None
                
                # Keyword matching for images
                if any(w in text_seg for w in ["profit", "money", "bagger", "wealth", "alpha", "growth"]):
                    overlay_path = "assets/broll/profit.png"
                elif any(w in text_seg for w in ["risk", "trap", "however", "danger", "chokehold", "fail", "destruction"]):
                    overlay_path = "assets/broll/danger.png"
                    
                if overlay_path and os.path.exists(overlay_path):
                    start_t = w_info['start']
                    dur = 1.5 # Show for 1.5 seconds
                    if start_t + dur > total_dur: dur = total_dur - start_t
                    try:
                        print(f"🌟 B-Roll Triggered: {overlay_path} at {start_t}s")
                        img_clip = (ImageClip(overlay_path)
                                    .with_duration(dur)
                                    .with_start(start_t)
                                    .resized(width=int(target_w * 0.4)) # 40% of screen width
                                    .with_position('center')
                                    .with_effects([vfx.CrossFadeIn(0.2), vfx.CrossFadeOut(0.2)]))
                        broll_clips.append(img_clip)
                    except Exception as e:
                        print(f"❌ Error adding B-Roll: {e}")
                phrase = []

    # 8. Render
    final_audio_elements = [vo_audio] + sfx_layers
    if os.path.exists(a_cfg['bgm_path']):
        print("🎚️ Applying dynamic Audio Ducking to Background Music...")
        ducking_vol = a_cfg.get('bgm_ducking_volume', 0.08)
        base_vol = a_cfg.get('bgm_base_volume', 0.25)
        
        resolution = 10 
        max_t_idx = int(total_dur * resolution) + 1
        vol_mask = np.full(max_t_idx, base_vol)
        
        pad = 0.3
        for w in words_data:
            start_idx = max(0, int((w['start'] - pad) * resolution))
            end_idx = min(max_t_idx - 1, int((w['end'] + pad) * resolution))
            vol_mask[start_idx:end_idx+1] = ducking_vol
            
        def duck_volume(t):
            t_arr = np.asarray(t)
            indices = np.clip((t_arr * resolution).astype(int), 0, max_t_idx - 1)
            vol = vol_mask[indices]
            if vol.ndim > 0:
                return vol[:, None]
            return vol

        bgm = AudioFileClip(a_cfg['bgm_path']).with_duration(total_dur)
        bgm = bgm.fl(lambda gf, t: duck_volume(t) * gf(t), keep_duration=True).audio_fadeout(3)
        final_audio_elements.insert(0, bgm)

    final_video = CompositeVideoClip([video_base] + broll_clips + caption_clips, size=(target_w, target_h)).with_audio(CompositeAudioClip(final_audio_elements))
    
    print(f"🎥 Rendering masterpiece to: {v_cfg['output']}")
    final_video.write_videofile(
        v_cfg['output'], 
        fps=v_cfg['fps'], 
        codec="libx264", 
        audio_codec="aac", 
        threads=8 if e_cfg['fast_render'] else 1, 
        preset='ultrafast' if e_cfg['fast_render'] else 'medium'
    )
    print(f"🎉 DONE!")

if __name__ == "__main__":
    asyncio.run(main())
