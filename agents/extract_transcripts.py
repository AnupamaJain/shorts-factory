import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
import time

def extract_video_scripts(video_dir="inputs/channelvideo", output_dir="inputs/rag_data"):
    print(f"Loading Whisper Model (base, int8)...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for filename in video_files:
        video_path = os.path.join(video_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_txt = os.path.join(output_dir, f"{base_name}.txt")
        
        # Skip if already extracted
        if os.path.exists(output_txt):
            print(f"⏭️ Skipping {filename} (already extracted)")
            continue
            
        print(f"🎬 Processing: {filename}")
        temp_audio = "temp/temp_extract.mp3"
        
        try:
            # 1. Extract audio using moviepy
            print("  -> Extracting audio...")
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                print("  -> No audio found in this video.")
                continue
                
            clip.audio.write_audiofile(temp_audio, logger=None)
            clip.close()
            
            # 2. Transcribe
            print("  -> Transcribing with Whisper...")
            segments, _ = model.transcribe(temp_audio)
            text = " ".join([s.text.strip() for s in segments])
            
            # 3. Save to text file
            with open(output_txt, "w") as f:
                f.write(text)
            print(f"✅ Saved transcript to {output_txt}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            
    print("\n🎉 All extractions complete!")

if __name__ == "__main__":
    extract_video_scripts()
