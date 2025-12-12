#python main.py --input "test2.mp3" --diarize
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#pip install Cython packaging
#pip install "nemo_toolkit[asr]"
#pip install git+https://github.com/m-bain/whisperx.git
#pip install google-generativeai deep-translator numpy
import os
import sys
import warnings
import logging
import gc 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           
os.environ["TRANSFORMERS_VERBOSITY"] = "error"     
os.environ["PYTORCH_LIGHTNING_CONSOLE_LOG_LEVEL"] = "0" 
os.environ["NEMO_CACHE_DIR"] = os.getcwd() 

warnings.filterwarnings("ignore")

try:
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=DeprecationWarning)
except: pass

LOGGERS_TO_SILENCE = [
    "whisperx", "whisperx.asr", "vads", "transformers", 
    "pytorch_lightning", "numba", "matplotlib", "urllib3", "speechbrain", "nemo_logger"
]

for logger_name in LOGGERS_TO_SILENCE:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# ==========================================

import argparse
import subprocess
import tempfile
import json
import shutil
import time
import numpy as np
from datetime import timedelta

import whisperx
import torch
import google.generativeai as genai

# --- THÊM NEMO ---
try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    print("❌ Thiếu thư viện NeMo. Hãy cài đặt: pip install nemo_toolkit[asr]")
    sys.exit(1)

# --- FIX LOAD MODEL PYTORCH 2.6+ ---
try:
    original_torch_load = torch.load
    def custom_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = custom_torch_load
except Exception as e:
    print(f"❌ Lỗi fix PyTorch: {e}")

# Thư viện dịch
try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("❌ Thiếu deep-translator. Pip install deep-translator")
    sys.exit(1)

# -------------------------
# CẤU HÌNH
# -------------------------
INPUT_LANG = None
OUTPUT_LANG = "en"

# -------------------------
# 1. NEMO DIARIZATION
# -------------------------
def run_nemo_diarization(wav_path, device="cpu"):
    print(f"👥 [NeMo] Đang tải model tách người nói (Device: {device})...")
    try:
        diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
        diar_model = diar_model.to(device)
        diar_model.eval()
        
        print("👥 [NeMo] Đang phân tích file audio...")
        # Sortformer cực mạnh về overlapping, nó sẽ trả về nhiều dòng chồng nhau
        raw_segments = diar_model.diarize(audio=wav_path, batch_size=1)
        
        speaker_segments = []
        if len(raw_segments) > 0:
            first_file_results = raw_segments[0]
            for line in first_file_results:
                if isinstance(line, str): parts = line.split()
                else: parts = line
                
                if len(parts) >= 3:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        speaker = str(parts[2])
                        speaker_segments.append({"start": start, "end": end, "speaker": speaker})
                    except ValueError: continue
        
        del diar_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        print(f"✅ [NeMo] Tìm thấy {len(speaker_segments)} đoạn hội thoại.")
        return speaker_segments

    except Exception as e:
        print(f"❌ Lỗi NeMo: {e}")
        return []

def merge_speaker_with_transcript(whisper_segments, nemo_segments):
    """
    Gán nhãn người nói từ NeMo sang text của Whisper dựa trên độ chồng lấn (overlap).
    """
    print("🔄 Đang đồng bộ người nói (NeMo) với văn bản (Whisper)...")
    
    for w_seg in whisper_segments:
        w_start = w_seg['start']
        w_end = w_seg['end']
        
        max_overlap = 0
        best_speaker = "UNKNOWN"
        
        # Tìm đoạn NeMo trùng khớp nhất
        for n_seg in nemo_segments:
            # Tính phần giao nhau (Intersection)
            start_overlap = max(w_start, n_seg['start'])
            end_overlap = min(w_end, n_seg['end'])
            overlap_duration = max(0, end_overlap - start_overlap)
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = n_seg['speaker']
        
        # Nếu không tìm thấy overlap đáng kể, giữ nguyên hoặc gán Unknown
        w_seg['speaker'] = best_speaker

    return whisper_segments

# -------------------------
# UTILS & TRANSLATE
# -------------------------
def correct_spelling_gemini(segments, api_key, batch_size):
    if not api_key: return segments
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except: return segments
    print("✨ Dùng Gemini để sửa lỗi chính tả...")
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        text_block = "\n".join(f"[{idx}] {s.get('text','').strip()}" for idx, s in enumerate(batch))
        prompt = f"Correct spelling and grammar for Vietnamese. Keep format [N]. Only output result. Text:\n{text_block}"
        try:
            resp = model.generate_content(prompt)
            lines = resp.text.strip().splitlines()
            for idx, seg in enumerate(batch):
                match = next((l for l in lines if l.strip().startswith(f"[{idx}]")), None)
                seg["corrected_text"] = match.split("]", 1)[-1].strip() if match else seg.get("text", "")
            time.sleep(1)
        except:
            for seg in batch: seg["corrected_text"] = seg.get("text", "")
    return segments

def translate_offline(segments):
    print(f"🌐 Đang dịch sang '{OUTPUT_LANG}'...")
    translator = GoogleTranslator(source='auto', target=OUTPUT_LANG)
    chunk_size = 2000
    current_chunk = []; seg_indices = []; current_len = 0
    for i, seg in enumerate(segments):
        text = seg.get("corrected_text", seg.get("text", "")).strip()
        if current_len + len(text) > chunk_size:
            try:
                translated = translator.translate_batch(current_chunk)
                for idx, t in zip(seg_indices, translated): segments[idx]["translated_text"] = t
            except: pass
            current_chunk = []; seg_indices = []; current_len = 0
        current_chunk.append(text)
        seg_indices.append(i)
        current_len += len(text)
    if current_chunk:
        try:
            translated = translator.translate_batch(current_chunk)
            for idx, t in zip(seg_indices, translated): segments[idx]["translated_text"] = t
        except: pass
    return segments

def run_ffmpeg_to_wav(in_path, out_path):
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", in_path, "-ar", "16000", "-ac", "1", out_path]
    subprocess.run(cmd, check=True)

def format_timestamp(seconds_float):
    td = timedelta(seconds=float(seconds_float))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds_float - int(seconds_float)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def segments_to_srt(segments, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"]).replace('.', ',')
            end = format_timestamp(seg["end"]).replace('.', ',')
            speaker = seg.get("speaker", "UNKNOWN")
            en = seg.get("corrected_text", seg.get("text", "")).strip()
            vi = seg.get("translated_text", en).strip()
            f.write(f"{i}\n{start} --> {end}\n[{speaker}] {en}\n[{speaker}] {vi}\n\n")

def segments_to_vtt(segments, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            speaker = seg.get("speaker", "UNKNOWN")
            en = seg.get("corrected_text", seg.get("text", "")).strip()
            vi = seg.get("translated_text", en).strip()
            f.write(f"{start} --> {end}\n[{speaker}] {en}\n[{speaker}] {vi}\n\n")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------------------
# MAIN
# -------------------------
def main(args):
    if shutil.which("ffmpeg") is None: sys.exit("❌ Cần cài ffmpeg")
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    tmp_wav = os.path.join(tempfile.gettempdir(), f"{base}_16k.wav")

    try:
        print("🔄 Chuẩn hóa Audio (FFmpeg)...")
        run_ffmpeg_to_wav(args.input, tmp_wav)

        # STEP 1: NEMO
        nemo_speaker_segments = []
        if args.diarize:
            nemo_speaker_segments = run_nemo_diarization(tmp_wav, device=args.device)
            if nemo_speaker_segments:
                nemo_txt_path = os.path.join(args.out_dir, f"{base}_nemo_timestamps.txt")
                print(f"📊 [NeMo] Lưu file timestamps raw: {nemo_txt_path}")
                with open(nemo_txt_path, "w", encoding="utf-8") as f:
                    for seg in nemo_speaker_segments:
                        start_str = format_timestamp(seg['start'])
                        end_str = format_timestamp(seg['end'])
                        f.write(f"[{start_str} --> {end_str}] {seg['speaker']}\n")
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print("🧹 [System] Đã dọn dẹp bộ nhớ NeMo.")

        # STEP 2: WHISPERX (FIXED CUT OFF)
        print(f"📦 [WhisperX] Transcribing ({args.model})...")
        
        # --- CẤU HÌNH VAD ĐỂ KHÔNG BỊ CẮT HỤT CHỮ ---
        # vad_onset: 0.500 -> 0.4 (Nhạy hơn lúc bắt đầu)
        # vad_offset: 0.363 -> 0.1 (Đợi lâu hơn mới ngắt sau khi hết tiếng)
        # chunk_size: Tăng lên 30s (Mặc định) để Whisper có nhiều ngữ cảnh hơn
        vad_options = {
            "vad_onset": 0.4, 
            "vad_offset": 0.1,  # <--- Quan trọng: Giảm xuống 0.1 để giữ âm đuôi
            "chunk_size": 30 
        }
        
        model = whisperx.load_model(
            args.model, 
            args.device, 
            compute_type=args.compute_type,
            vad_options=vad_options # Áp dụng cấu hình mới
        )
        audio = whisperx.load_audio(tmp_wav)
        
        result = model.transcribe(audio, batch_size=args.batch_size, language=INPUT_LANG)
        
        model_cpu = model; del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("⚡ [WhisperX] Alignment")
        
        align_model_name = None
        if result["language"] == "vi":
            align_model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"

        align_model, meta = whisperx.load_align_model(
            language_code=result["language"], 
            device=args.device,
            model_name=align_model_name
        )
        
        result = whisperx.align(result["segments"], align_model, meta, audio, args.device, return_char_alignments=False)
        
        del align_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        segments = result["segments"]

        # STEP 3: MERGE
        if args.diarize and nemo_speaker_segments:
            segments = merge_speaker_with_transcript(segments, nemo_speaker_segments)
        else:
            for s in segments: s["speaker"] = "SPEAKER_00"

        # STEP 4: CORRECTION & TRANSLATE
        if args.gemini_key:
            segments = correct_spelling_gemini(segments, args.gemini_key, args.gemini_batch_size)
        else:
            for s in segments: s["corrected_text"] = s.get("text", "")

        if OUTPUT_LANG != INPUT_LANG:
            segments = translate_offline(segments)
        else:
            for s in segments: s["translated_text"] = s.get("corrected_text", "")

        # STEP 5: EXPORT
        print("📝 Đang lưu kết quả...")
        with open(os.path.join(args.out_dir, f"{base}_raw.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([s.get("text", "").strip() for s in segments]))
        with open(os.path.join(args.out_dir, f"{base}_corrected.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([s.get("corrected_text", "").strip() for s in segments]))

        save_json({"segments": segments}, os.path.join(args.out_dir, f"{base}.json"))
        segments_to_srt(segments, os.path.join(args.out_dir, f"{base}.srt"))
        segments_to_vtt(segments, os.path.join(args.out_dir, f"{base}.vtt"))
        
        print(f"\n✅ Hoàn tất! Kiểm tra thư mục: {args.out_dir}")

    except Exception as e:
        print(f"❌ Lỗi Critical: {e}")
        import traceback; traceback.print_exc()
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compute_type", default="float16" if torch.cuda.is_available() else "int8") 
    p.add_argument("--batch_size", type=int, default=8)
    
    p.add_argument("--diarize", action="store_true", help="Bật tách người nói bằng NeMo")
    p.add_argument("--gemini_key", default=None)
    p.add_argument("--gemini_batch_size", type=int, default=30)
    
    args = p.parse_args()
    main(args)
