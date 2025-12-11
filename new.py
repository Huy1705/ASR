#python new.py --input "test2.mp3" --diarize
#pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#python new.py --input "F:\Projects\ASR(Sửa cái dịch phiên)\ASR\test2.mp3" --model large-v3 --batch_size 8 --diarize --gemini_batch_size 30 --out_dir "F:\Projects\ASR(Sửa cái dịch phiên)\ASR_outputs"#pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#pip install git+https://github.com/m-bain/whisperx.git
#pip install google-generativeai
import os
import sys
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          
os.environ["TRANSFORMERS_VERBOSITY"] = "error"     
os.environ["PYTORCH_LIGHTNING_CONSOLE_LOG_LEVEL"] = "0" 


warnings.filterwarnings("ignore")

try:
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=DeprecationWarning)
except: pass

LOGGERS_TO_SILENCE = [
    "whisperx",            # Chặn log "INFO - No language specified..."
    "whisperx.asr",        # Chặn log ASR cụ thể
    "vads",                # Chặn log VAD
    "transformers",        # Chặn log model weights
    "pytorch_lightning",   # Chặn log Lightning upgrade
    "numba",
    "matplotlib",
    "urllib3",
    "speechbrain"
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
import librosa
import soundfile as sf
import google.generativeai as genai

for logger_name in LOGGERS_TO_SILENCE:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
# Thư viện dịch miễn phí & Phân cụm
try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("❌ Thiếu thư viện dịch. Chạy: pip install deep-translator")
    sys.exit(1)

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    print("❌ Thiếu thư viện phân cụm. Chạy: pip install scikit-learn librosa")
    sys.exit(1)

# -------------------------
# CẤU HÌNH NGÔN NGỮ
# -------------------------
INPUT_LANG = "vi"      # Ngôn ngữ gốc
OUTPUT_LANG = "en"     # Ngôn ngữ đích
# -------------------------

# -------------------------
# 1. AUTO-DIARIZATION 
# -------------------------
def extract_embedding(wav_path, start, end, sr=16000):
    try:
        duration = end - start
        if duration < 0.5: return None
        y, _ = librosa.load(wav_path, sr=sr, offset=start, duration=duration)
        if len(y) == 0: return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.concatenate([np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0)])
    except: return None

def find_best_num_speakers(X, min_k=2, max_k=5):
    if len(X) < min_k + 1: return 1
    print(f"🔍 Đang tìm số lượng người nói ({min_k}-{max_k})...")
    best_k = 1; best_score = -1
    for k in range(min_k, max_k + 1):
        if k >= len(X): break
        try:
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score: best_score = score; best_k = k
        except: continue
    return best_k if best_score > 0.05 else 1

def perform_clustering_diarization(wav_path, segments, forced_num_speakers=0):
    print("👥 Đang phân tích giọng nói (Không cần Token)...")
    embeddings = []; valid_indices = []
    for idx, seg in enumerate(segments):
        emb = extract_embedding(wav_path, seg['start'], seg['end'])
        if emb is not None: embeddings.append(emb); valid_indices.append(idx)
    
    if not embeddings: return segments
    X = StandardScaler().fit_transform(np.array(embeddings))
    
    n_clusters = forced_num_speakers if forced_num_speakers > 0 else find_best_num_speakers(X)
    print(f"✅ Phát hiện: {n_clusters} người nói.")
    
    labels = [0]*len(X) if n_clusters == 1 else AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
    
    for i, label in enumerate(labels):
        segments[valid_indices[i]]["speaker"] = f"SPEAKER_{label:02d}"
    for seg in segments:
        if "speaker" not in seg: seg["speaker"] = "SPEAKER_00"
    return segments

# -------------------------
# 2. GEMINI (CHỈ SỬA LỖI CHÍNH TẢ)
# -------------------------
def correct_spelling_gemini(segments, api_key, batch_size):
    if not api_key: return segments
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except: return segments

    print("✨ Dùng Gemini để sửa lỗi chính tả (Correction only)...")
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        text_block = "\n".join(f"[{idx}] {s.get('text','').strip()}" for idx, s in enumerate(batch))
        prompt = f"Correct spelling and grammar. Keep format [N]. Text:\n{text_block}"
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

# -------------------------
# 3. GOOGLE TRANSLATE 
# -------------------------
def translate_offline(segments):
    print(f"🌐 Đang dịch sang '{OUTPUT_LANG}' bằng Google Translate (Free)...")
    translator = GoogleTranslator(source='auto', target=OUTPUT_LANG)
    
    chunk_size = 2000
    current_chunk = []; current_length = 0; seg_indices = []

    for i, seg in enumerate(segments):
        text = seg.get("corrected_text", seg.get("text", "")).strip()
        
        if current_length + len(text) > chunk_size:
            try:
                translated = translator.translate_batch(current_chunk)
                for idx, trans_text in zip(seg_indices, translated):
                    segments[idx]["translated_text"] = trans_text
            except Exception as e:
                print(f"⚠️ Lỗi dịch: {e}")
                for idx in seg_indices: segments[idx]["translated_text"] = segments[idx].get("corrected_text", "")
            current_chunk = []; seg_indices = []; current_length = 0

        current_chunk.append(text)
        seg_indices.append(i)
        current_length += len(text)

    if current_chunk:
        try:
            translated = translator.translate_batch(current_chunk)
            for idx, trans_text in zip(seg_indices, translated):
                segments[idx]["translated_text"] = trans_text
        except:
            for idx in seg_indices: segments[idx]["translated_text"] = segments[idx].get("corrected_text", "")

    return segments

# -------------------------
# UTILITIES & EXPORT FORMAT
# -------------------------
def run_ffmpeg_to_wav(in_path, out_path):
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", in_path, "-ar", "16000", "-ac", "1", out_path]
    subprocess.run(cmd, check=True)

def format_timestamp(seconds_float):
    td = timedelta(seconds=float(seconds_float))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    frac = seconds_float - int(seconds_float)
    milliseconds = int(frac * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def segments_to_srt(segments, out_file):
    """SRT song ngữ: dòng 1 tiếng Anh, dòng 2 tiếng Việt."""
    with open(out_file, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            start = format_timestamp(seg["start"]).replace('.', ',')
            end = format_timestamp(seg["end"]).replace('.', ',')
            speaker = seg.get("speaker", "UNKNOWN")
            
            # Lấy text đã sửa (nếu có), không thì lấy text gốc
            en = seg.get("corrected_text", "").strip()
            if not en: en = seg.get("text", "").strip()
            
            # Lấy text dịch, fallback là text tiếng Anh
            vi = seg.get("translated_text", en).strip()

            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"[{speaker}] {en}\n")
            f.write(f"[{speaker}] {vi}\n\n")
            idx += 1

def segments_to_vtt(segments, out_file):
    """VTT song ngữ: dòng 1 tiếng Anh, dòng 2 tiếng Việt."""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            speaker = seg.get("speaker", "UNKNOWN")
            
            en = seg.get("corrected_text", "").strip()
            if not en: en = seg.get("text", "").strip()
            
            vi = seg.get("translated_text", en).strip()

            f.write(f"{start} --> {end}\n")
            f.write(f"[{speaker}] {en}\n")
            f.write(f"[{speaker}] {vi}\n\n")

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
        print("🔄 Chuẩn hóa Audio...")
        run_ffmpeg_to_wav(args.input, tmp_wav)

        print(f"📦 Transcribing ({args.device})...")
        model = whisperx.load_model(args.model, args.device, compute_type=args.compute_type)
        audio = whisperx.load_audio(tmp_wav)
        result = model.transcribe(audio, batch_size=args.batch_size, language=INPUT_LANG)
        
        print("⚡ Alignment...")
        align_model, meta = whisperx.load_align_model(language_code=result["language"], device=args.device)
        result = whisperx.align(result["segments"], align_model, meta, audio, args.device, return_char_alignments=False)

        # 4. DIARIZATION (NO TOKEN)
        if args.diarize:
            result["segments"] = perform_clustering_diarization(tmp_wav, result["segments"], args.num_speakers)
        
        segments = result["segments"]

        # 5. GEMINI CORRECTION (OPTIONAL)
        if args.gemini_key:
            segments = correct_spelling_gemini(segments, args.gemini_key, args.gemini_batch_size)
        else:
            print("ℹ️ Không có Gemini Key: Bỏ qua bước sửa lỗi chính tả.")
            for s in segments: s["corrected_text"] = s.get("text", "")

        # 6. TRANSLATION (ALWAYS RUN)
        if OUTPUT_LANG != INPUT_LANG:
            segments = translate_offline(segments)
        else:
            for s in segments: s["translated_text"] = s.get("corrected_text", "")
        # 1. RAW TEXT (Gốc Whisper)
        raw_full = "\n".join([s.get("text", "").strip() for s in segments])
        with open(os.path.join(args.out_dir, f"{base}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw_full)
            
        # 2. CORRECTED TEXT (Đã sửa lỗi)
        corr_full = "\n".join([s.get("corrected_text", "").strip() for s in segments])
        with open(os.path.join(args.out_dir, f"{base}_corrected.txt"), "w", encoding="utf-8") as f:
            f.write(corr_full)
            
        # 3. TRANSLATED TEXT (Đã dịch)
        trans_full = "\n".join([s.get("translated_text", "").strip() for s in segments])
        with open(os.path.join(args.out_dir, f"{base}_translated_{OUTPUT_LANG}.txt"), "w", encoding="utf-8") as f:
            f.write(trans_full)
        print("📝 Lưu kết quả (SRT/VTT Song Ngữ)...")
        save_json({"segments": segments}, os.path.join(args.out_dir, f"{base}.json"))
        segments_to_srt(segments, os.path.join(args.out_dir, f"{base}.srt"))
        segments_to_vtt(segments, os.path.join(args.out_dir, f"{base}.vtt"))
        
        print(f"\n✅ Hoàn tất! File tại: {args.out_dir}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
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
    
    p.add_argument("--diarize", action="store_true", help="Bật phân biệt người nói")
    p.add_argument("--num_speakers", type=int, default=0, help="0 = Tự động tìm")
    
    p.add_argument("--gemini_key", default=None, help="Key sửa lỗi chính tả (không bắt buộc)")
    p.add_argument("--gemini_batch_size", type=int, default=30)
    args = p.parse_args()
    main(args)