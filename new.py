#python app.py --input "filepath" --model large-v3 --device cuda --compute_type float16 --batch_size 8 --diarize --token --gemini_key key --out_dir "E:/newwshiper" 
#pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#pip install git+https://github.com/m-bain/whisperx.git
#pip install google-generativeai
import os
import sys
import argparse
import subprocess
import tempfile
import json
import shutil
import time
import re
from datetime import timedelta

# whisperx + dependencies
import whisperx
import torch

# optional: diarization pipeline & gemini
try:
    from whisperx.diarize import DiarizationPipeline
except Exception:
    DiarizationPipeline = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# -------------------------
# GLOBAL LANGUAGE SETTINGS
# -------------------------
INPUT_LANG = "en"      # Ngôn ngữ đầu vào (Whisper)
OUTPUT_LANG = "vi"     # Ngôn ngữ mong muốn cho output
# -------------------------

# -------------------------
# Utilities
# -------------------------
def run_ffmpeg_to_wav(in_path, out_path, sample_rate=16000, channels=1):
    """Normalize audio to WAV 16kHz mono using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", in_path,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-map_metadata", "-1",
        "-fflags", "+genpts",
        out_path
    ]
    subprocess.run(cmd, check=True)

def format_timestamp(seconds_float):
    """Return hh:mm:ss.mmm"""
    td = timedelta(seconds=float(seconds_float))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    frac = seconds_float - int(seconds_float)
    milliseconds = int(frac * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# -------------------------------------------------------------------
# SRT / VTT EXPORT (EN TRÊN – VI DƯỚI)
# -------------------------------------------------------------------
def segments_to_srt(segments, out_file):
    """SRT song ngữ: dòng 1 tiếng Anh, dòng 2 tiếng Việt."""
    with open(out_file, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            start = format_timestamp(seg["start"]).replace('.', ',')
            end = format_timestamp(seg["end"]).replace('.', ',')

            speaker = seg.get("speaker", "UNKNOWN")

            en = seg.get("corrected_text", "").strip()
            vi = seg.get("translated_text", en).strip()  # fallback EN nếu chưa dịch

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
            vi = seg.get("translated_text", en).strip()

            f.write(f"{start} --> {end}\n")
            f.write(f"[{speaker}] {en}\n")
            f.write(f"[{speaker}] {vi}\n\n")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------------------
# Gemini batch correction
# -------------------------
def correct_spelling_with_gemini_batch(segments, api_key, batch_size=5, model_name="gemini-2.5-flash"):
    if not segments:
        return segments

    if genai is None:
        print("⚠️ google.generativeai chưa cài → bỏ qua correction.")
        return [dict(s, corrected_text=s.get("text", "")) for s in segments]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print("⚠️ Lỗi Gemini:", e)
        return [dict(s, corrected_text=s.get("text", "")) for s in segments]

    corrected_segments = []
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_i in range(0, len(segments), batch_size):
        batch = segments[batch_i:batch_i + batch_size]
        print(f"  🔄 Batch {batch_i // batch_size + 1}/{total_batches}")

        batch_text = "\n".join(f"[{i}] {seg.get('text','').strip()}" for i, seg in enumerate(batch))
        prompt = f"""
Bạn là chuyên gia sửa lỗi chính tả. Nhiệm vụ: sửa từng dòng, giữ nguyên [N], không đổi ý nghĩa.
Trả về kết quả theo định dạng "[N] <dòng đã sửa>" trên mỗi dòng.

{batch_text}
"""

        max_retries = 4
        backoff_times = [5, 10, 20, 40]
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                corrected_lines = response.text.strip().splitlines()
                for i, seg in enumerate(batch):
                    new_seg = dict(seg)
                    matches = [l for l in corrected_lines if l.strip().startswith(f"[{i}]")]
                    if matches:
                        new_seg["corrected_text"] = matches[0].split("]", 1)[-1].strip()
                    else:
                        new_seg["corrected_text"] = seg.get("text", "")
                    corrected_segments.append(new_seg)
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                    if attempt < max_retries - 1:
                        wait = backoff_times[attempt]
                        print(f"  ⚠️ Rate limit → chờ {wait}s …")
                        time.sleep(wait)
                    else:
                        print("  ❌ Batch failed (quota). Dùng nguyên văn.")
                        for seg in batch:
                            corrected_segments.append(dict(seg, corrected_text=seg.get("text", "")))
                else:
                    print("❌ Lỗi Gemini:", e)
                    for seg in batch:
                        corrected_segments.append(dict(seg, corrected_text=seg.get("text", "")))
                    break
        time.sleep(2)
    return corrected_segments

# -------------------------
# FIXED: Translate segment by segment
# -------------------------
def translate_segments_with_gemini(segments, api_key, target_lang, batch_size=10, model_name="gemini-2.5-flash"):
    """Dịch từng segment riêng lẻ thay vì dịch toàn bộ text."""
    if genai is None:
        print("⚠️ google.generativeai chưa cài → bỏ qua translation.")
        for seg in segments:
            seg["translated_text"] = seg.get("corrected_text", "")
        return segments
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print("⚠️ Lỗi Gemini:", e)
        for seg in segments:
            seg["translated_text"] = seg.get("corrected_text", "")
        return segments

    total_batches = (len(segments) + batch_size - 1) // batch_size
    
    for batch_i in range(0, len(segments), batch_size):
        batch = segments[batch_i:batch_i + batch_size]
        print(f"  🌐 Translating batch {batch_i // batch_size + 1}/{total_batches}")
        
        # Tạo prompt với từng câu được đánh số
        batch_text = "\n".join(f"[{i}] {seg.get('corrected_text','').strip()}" for i, seg in enumerate(batch))
        prompt = f"""
Hãy dịch từng câu sau sang tiếng {target_lang}. 
Giữ nguyên format [N] và trả về mỗi câu đã dịch trên một dòng riêng.

{batch_text}
"""

        max_retries = 4
        backoff_times = [5, 10, 20, 40]
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                translated_lines = response.text.strip().splitlines()
                
                for i, seg in enumerate(batch):
                    # Tìm dòng dịch tương ứng với [i]
                    matches = [l for l in translated_lines if l.strip().startswith(f"[{i}]")]
                    if matches:
                        seg["translated_text"] = matches[0].split("]", 1)[-1].strip()
                    else:
                        seg["translated_text"] = seg.get("corrected_text", "")
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                    if attempt < max_retries - 1:
                        wait = backoff_times[attempt]
                        print(f"  ⚠️ Rate limit → chờ {wait}s …")
                        time.sleep(wait)
                    else:
                        print("  ❌ Translation batch failed. Using original text.")
                        for seg in batch:
                            seg["translated_text"] = seg.get("corrected_text", "")
                else:
                    print("❌ Lỗi dịch:", e)
                    for seg in batch:
                        seg["translated_text"] = seg.get("corrected_text", "")
                    break
        time.sleep(2)
    
    return segments

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg không được tìm thấy trong PATH.")
        sys.exit(1)

    input_path = args.input
    if not os.path.exists(input_path):
        print("❌ File input không tồn tại:", input_path)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    tmp_wav = None

    try:
        # Convert to WAV normalized
        tmp_wav = os.path.join(tempfile.gettempdir(), f"{base}_norm.wav")
        print("🔄 Chuẩn hóa audio ->", tmp_wav)
        run_ffmpeg_to_wav(input_path, tmp_wav, sample_rate=16000, channels=1)

        # Load whisperx model
        device = args.device
        compute_type = args.compute_type
        print(f"📦 Loading WhisperX model '{args.model}' on {device} ({compute_type})...")
        model = whisperx.load_model(args.model, device, compute_type=compute_type)

        # Load audio and transcribe
        print("🎵 Loading audio...")
        audio = whisperx.load_audio(tmp_wav)

        w_lang = INPUT_LANG if INPUT_LANG else None
        print(f"🎤 Transcribing (language={w_lang})")
        result = model.transcribe(audio, batch_size=args.batch_size, language=w_lang)

        # Save raw text
        raw_text = " ".join([s.get("text", "").strip() for s in result["segments"]])
        raw_outpath = os.path.join(args.out_dir, f"{base}_raw.txt")
        with open(raw_outpath, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Alignment
        align_lang = INPUT_LANG if INPUT_LANG else result.get("language")
        print("🔗 Loading align model for language:", align_lang)
        align_model, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
        print("⚡ Running alignment...")
        result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

        # Diarization
        if args.diarize:
            if DiarizationPipeline is None or not args.hf_token:
                print("⚠️ Diarization bị vô hiệu hóa.")
            else:
                print("👥 Chạy Diarization...")
                diarize_model = DiarizationPipeline(use_auth_token=args.hf_token, device=device)
                diarize_segments = diarize_model(tmp_wav)
                result = whisperx.assign_word_speakers(diarize_segments, result)
        else:
            print("ℹ️ Diarization: bị vô hiệu hóa.")

        # Gemini batch correction
        if args.gemini_key:
            print(f"✨ Sửa chính tả với Gemini...")
            corrected_segments = correct_spelling_with_gemini_batch(result["segments"], args.gemini_key, batch_size=args.gemini_batch_size)
        else:
            corrected_segments = [dict(s, corrected_text=s.get("text","")) for s in result["segments"]]

        final_corrected = " ".join([s.get("corrected_text","") for s in corrected_segments])
        corrected_outpath = os.path.join(args.out_dir, f"{base}_corrected.txt")
        with open(corrected_outpath, "w", encoding="utf-8") as f:
            f.write(final_corrected)

        # Translation - FIXED: Dịch từng segment
        translation_performed = False
        if args.gemini_key and OUTPUT_LANG and OUTPUT_LANG != INPUT_LANG:
            print(f"🌐 Dịch từng segment sang '{OUTPUT_LANG}'...")
            corrected_segments = translate_segments_with_gemini(
                corrected_segments, 
                args.gemini_key, 
                OUTPUT_LANG,
                batch_size=args.translation_batch_size
            )
            translation_performed = True
            
            # Save full translated text
            translated_full = " ".join([s.get("translated_text","") for s in corrected_segments])
            trans_outpath = os.path.join(args.out_dir, f"{base}_translated_{OUTPUT_LANG}.txt")
            with open(trans_outpath, "w", encoding="utf-8") as f:
                f.write(translated_full)
        else:
            for seg in corrected_segments:
                seg["translated_text"] = seg.get("corrected_text","")

        # Save JSON
        save_payload = {"segments": corrected_segments, "language": INPUT_LANG}
        if translation_performed:
            save_payload["translated_full_text_language"] = OUTPUT_LANG
        save_json(save_payload, os.path.join(args.out_dir, f"{base}.json"))

        # SRT / VTT
        print("📝 Tạo SRT / VTT...")
        segments_to_srt(corrected_segments, os.path.join(args.out_dir, f"{base}.srt"))
        segments_to_vtt(corrected_segments, os.path.join(args.out_dir, f"{base}.vtt"))

        print("\n✅ Hoàn tất! Outputs trong:", os.path.abspath(args.out_dir))
        print("\n📊 Preview (first 5 segments):")
        for seg in corrected_segments[:5]:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            en_text = seg.get("corrected_text","").strip()
            vi_text = seg.get("translated_text","").strip()
            speaker = seg.get("speaker","UNKNOWN")
            print(f"  [{speaker}] {start} --> {end}")
            print(f"    EN: {en_text}")
            print(f"    VI: {vi_text}\n")

    except subprocess.CalledProcessError as e:
        print("❌ Lỗi ffmpeg:", e)
    except Exception as e:
        print("❌ Lỗi pipeline:", e)
        import traceback
        traceback.print_exc()
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try: os.remove(tmp_wav)
            except: pass

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="WhisperX + diarization + Gemini correction + translation")
    p.add_argument("--input", required=True, help="Path to input audio/video")
    p.add_argument("--out_dir", default="outputs", help="Directory for outputs")
    p.add_argument("--model", default="large-v3", help="WhisperX model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compute_type", default="float16" if torch.cuda.is_available() else "int8")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--hf_token", default=None, help="HF token for diarization")
    p.add_argument("--gemini_key", default=None, help="Gemini API key")
    p.add_argument("--gemini_batch_size", type=int, default=30, help="Batch size for spelling correction")
    p.add_argument("--translation_batch_size", type=int, default=10, help="Batch size for translation")
    args = p.parse_args()
    main(args)