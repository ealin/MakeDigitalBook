#!/usr/bin/env python3
import os
import sys
import re
import base64
import json
import urllib.request
import urllib.error
import socket
import argparse
import time

def extract_page_number(filename: str) -> int:
    """Extract trailing numeric page number for sorting.
    Example: '台灣AI大未來 - 100.jpg' -> 100"""
    base = os.path.splitext(filename)[0]
    match = re.findall(r'\d+', base)
    if match:
        return int(match[-1])
    return 999999

def clean_ocr_text(text: str) -> str:
    """Clean OCR output:
    - Remove markdown code block markers.
    - Strip typical chatbot preambles.
    - Remove e‑reader status bar noise (time, battery, UI tokens).
    - Remove stray single‑line noise like "A4".
    """
    # 1. Remove markdown code block fences
    text = re.sub(r'^```[a-zA-Z]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
    # 2. Remove common chatbot intro phrases
    intro_phrases = [
        r'^\s*Here is the extracted text.*:\s*$',
        r'^\s*Extracted text.*:\s*$',
        r'^\s*以下是圖片中的文字.*:\s*$',
        r'^\s*解析結果.*:\s*$'
    ]
    for phrase in intro_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.MULTILINE)
    # 3. Remove status‑bar noise (time, battery, UI tokens), book titles, and page numbers – only in top/bottom lines
    text = text.strip()
    lines = text.split('\n')
    cleaned_lines = []
    time_pattern = re.compile(r'^\s*(AM|PM)?\s*\d{1,2}:\d{2}\s*$', re.IGNORECASE)
    battery_pattern = re.compile(r'^\s*\d{1,3}%\s*$')
    ui_pattern = re.compile(r'^\s*(AA|[‹›<>|])\s*$', re.IGNORECASE)
    a4_pattern = re.compile(r'^\s*A4\s*$', re.IGNORECASE)  # single line "A4"
    book_title_pattern = re.compile(r'^\s*(十二大密意|藥師佛的12大願)\s*$')
    page_num_pattern = re.compile(r'^\s*\d+\s*$')

    for i, line in enumerate(lines):
        is_top_or_bottom = (i < 8) or (i >= len(lines) - 8)
        stripped = line.strip()
        if is_top_or_bottom:
            if (time_pattern.match(stripped) or battery_pattern.match(stripped) or
                ui_pattern.match(stripped) or a4_pattern.match(stripped) or
                book_title_pattern.match(stripped) or page_num_pattern.match(stripped)):
                continue  # drop this line
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines).strip()
    return result

def has_repetition_loop(text: str, max_detect_len: int = 30, min_repeats: int = 5) -> bool:
    """Detect if the end of the text has a repeating substring pattern."""
    for l in range(2, max_detect_len + 1):
        if len(text) < l * min_repeats:
            continue
        pattern = text[-l:]
        is_loop = True
        for r in range(1, min_repeats):
            start_idx = -l * (r + 1)
            end_idx = -l * r
            if text[start_idx:end_idx] != pattern:
                is_loop = False
                break
        if is_loop:
            return True
    return False

def is_garbage_text(text: str, is_english_book: bool = False) -> tuple[bool, str]:
    """Detect whether the OCR output is clearly not a real article.
    Returns (is_garbage, reason_string).
    Checks performed:
      1. Bounding-box / grounding tokens (e.g. <|ref|>...<|det|>[[...]])
      2. Indexed numbered repetitions (e.g. ま[1] ま[2] ま[3]...)
      3. Content is entirely non-Chinese (English hallucinations) for a Chinese book
      4. Extremely short content (< 10 meaningful characters after stripping)
    """
    if not text or not text.strip():
        return True, "辨識結果為空"

    stripped = text.strip()

    # 1. Bounding-box / grounding tokens produced by vision model
    if re.search(r'<\|ref\|>|<\|det\|>|\[\[\d+,\s*\d+', stripped):
        return True, "偵測到 Bounding-box / Grounding 定位標記（模型將任務誤解為物體偵測）"

    # 2. Indexed repetition pattern: any character/token repeated with ascending numbers [1] [2] [3]...
    #    Match sequences of 5+ occurrences of (token[N]) consecutively
    if re.search(r'(.)\[\d+\](?:\s*\1\[\d+\]){4,}', stripped):
        return True, "偵測到索引型重複序列（例如：ま[1] ま[2] ま[3]...），辨識結果明顯不是正常文章"

    # 3. Detect if text has far more English sentences than Chinese characters
    #    Chinese Unicode range: \u4e00-\u9fff, CJK Extension A/B etc.
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', stripped))
    total_alpha = len(re.findall(r'[a-zA-Z]', stripped))
    total_meaningful = len(re.findall(r'[^\s\n\r]', stripped))

    if not is_english_book:
        if total_meaningful > 20 and chinese_chars == 0 and total_alpha > 20:
            return True, "辨識結果為純英文（無任何中文字），疑似模型幻覺或圖片載入失敗"

        if total_meaningful > 30 and chinese_chars > 0 and total_alpha / max(chinese_chars, 1) > 5:
            return True, f"英文字母數量（{total_alpha}）遠超中文字數（{chinese_chars}），疑似模型描述圖片而非提取文字"

    # 4. Extremely short meaningful content
    if total_meaningful < 10:
        return True, f"辨識結果過短（僅 {total_meaningful} 個有效字符），可能為空白頁或辨識失敗"

    return False, ""

def perform_ocr_with_api(image_path: str, timeout: float = None) -> tuple[str, bool]:
    """Call Ollama HTTP API for deepseek-ocr.
    Returns a tuple of (raw OCR text, loop_detected flag)."""
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "model": "deepseek-ocr",
        "prompt": "Extract the text in the image.",
        "images": [img_data],
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024
        }
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    accumulated_text = ""
    loop_detected = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    res_text = chunk.get("response", "")
                    accumulated_text += res_text
                    
                    if has_repetition_loop(accumulated_text):
                        print("\n   [⚠️ Repetition loop detected! Breaking early to prevent timeout.]")
                        loop_detected = True
                        break
                    
                    if chunk.get("done", False):
                        break
        return accumulated_text, loop_detected
    except (urllib.error.URLError, socket.timeout) as e:
        raise TimeoutError(f"Ollama API request timed out or unreachable: {e}")

def run_unit_tests():
    """Simple unit tests for sorting and noise filtering."""
    print("🧪 Running unit tests…")
    # Sorting test
    test_files = ["台灣AI大未來 - 10.jpg", "台灣AI大未來 - 2.jpg",
                  "台灣AI大未來 - 100.jpg", "台灣AI大未來 - 1.jpg"]
    sorted_files = sorted(test_files, key=extract_page_number)
    expected = ["台灣AI大未來 - 1.jpg", "台灣AI大未來 - 2.jpg",
                "台灣AI大未來 - 10.jpg", "台灣AI大未來 - 100.jpg"]
    assert sorted_files == expected, f"Sorting failed: {sorted_files}"
    print("   [PASS] Filename sorting works.")
    # Noise filtering test – includes the new "A4" line and footer/page numbers
    raw = """```markdown
AM2:07
73%
AA
A4
技術視角，見證這波生成式AI如何覆舊有的AI思潮流。
十二大密意

1
```"""
    cleaned = clean_ocr_text(raw)
    expected_clean = "技術視角，見證這波生成式AI如何覆舊有的AI思潮流。"
    assert cleaned == expected_clean, f"Cleanup failed: {cleaned}"
    print("   [PASS] Noise filtering (including A4, book title, page numbers) works.")
    print("🎉 All unit tests passed!\n")

def merge_texts(book_id: str, out_txt_dir: str, sorted_filenames: list, final_out_path: str):
    """Combine per‑page .txt files into a single book file."""
    print(f"\nMerging texts into {final_out_path}…")
    merged = []
    for filename in sorted_filenames:
        base = os.path.splitext(filename)[0]
        page_path = os.path.join(out_txt_dir, f"{base}.txt")
        if os.path.exists(page_path):
            with open(page_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    merged.append(content)
        else:
            print(f"⚠️ Missing text file for page {filename}")
    with open(final_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(merged))
    print(f"✅ Merged successfully! Output saved at: {final_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Scanned Book OCR Processor (Ollama API)")
    parser.add_argument("book_id", type=str, help="Book identifier, e.g. B001")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Process only N pages (useful for testing)")
    parser.add_argument("-t", "--timeout", type=int, default=180,
                        help="Timeout in seconds for each OCR request (default 180)")
    parser.add_argument("--scan-dir", type=str, default="SCAN_PAGES",
                        help="Base directory containing scanned images")
    parser.add_argument("--out-dir", type=str, default="TXT",
                        help="Base directory for per‑page text output")
    parser.add_argument("--start", type=int, default=None,
                        help="Start processing from this page number (e.g., 30 will begin with the image whose extracted page number >= 30).")
    parser.add_argument("--test", action="store_true",
                        help="Run internal unit tests and exit")
    parser.add_argument("--rtl", action="store_true",
                        help="Process vertical Chinese text (read from right to left)")
    parser.add_argument("--english", action="store_true",
                        help="Process an English book (disables pure-English hallucination warnings)")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
        sys.exit(0)

    # Locate the correct book folder inside SCAN_PAGES
    if not os.path.isdir(args.scan_dir):
        print(f"❌ Scan directory '{args.scan_dir}' not found.")
        sys.exit(1)
    matching = [d for d in os.listdir(args.scan_dir)
                if d.startswith(args.book_id) and os.path.isdir(os.path.join(args.scan_dir, d))]
    if not matching:
        print(f"❌ No folder starting with '{args.book_id}' found in {args.scan_dir}.")
        sys.exit(1)
    book_dir = os.path.join(args.scan_dir, matching[0])
    print(f"📂 Found book folder: {book_dir}")

    # Gather image files (jpg/jpeg/png)
    valid_ext = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(book_dir) if f.lower().endswith(valid_ext)]
    if not images:
        print(f"❌ No image files in {book_dir}.")
        sys.exit(1)
    images.sort(key=extract_page_number)
    print(f"🔢 Detected {len(images)} image files.")

    # Apply start page if given
    if args.start is not None:
        images = [img for img in images if extract_page_number(img) >= args.start]
        if not images:
            print(f"❌ No images found with page number >= {args.start}.")
            sys.exit(1)
        print(f"🚦 Starting from page {args.start}. Remaining pages: {len(images)}.")

    # Prepare output folder
    out_folder = os.path.join(args.out_dir, args.book_id)
    os.makedirs(out_folder, exist_ok=True)

    # Apply limit if given
    if args.limit is not None:
        images = images[:args.limit]
        print(f"🚦 Limiting to first {args.limit} pages for this run.")

    processed = 0
    first_page_time = None  # Track the processing time of the first successfully completed OCR page in this run
    repetition_triggered_images = []  # Track images that triggered the repetition loop detection
    garbage_images = []  # Track images whose OCR result is clearly not a real article
    for idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(book_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        out_txt = os.path.join(out_folder, f"{base_name}.txt")
        # If the output .txt already exists, ask whether to regenerate
        if os.path.exists(out_txt):
            overwrite = False
            while True:
                resp = input(f"⚠️ Text file '{out_txt}' already exists. Overwrite? [y/n]: ").strip().lower()
                if resp in ('y', 'yes'):
                    overwrite = True
                    break
                elif resp in ('n', 'no'):
                    print(f"⏭️ Skipping page '{img_name}'.")
                    break
                else:
                    print("Please answer 'y' or 'n'.")
            if not overwrite:
                continue

        # Proceed with OCR processing for this page
        while True:
            try:
                if first_page_time is None:
                    print(f"📖 [{idx}/{len(images)}] Processing '{img_name}' (First page - No timeout limits)…")
                    current_timeout = None
                else:
                    # Use 2x first page time, with a minimum of 120s to prevent false-positives on fast cache hits
                    current_timeout = max(120.0, first_page_time * 2)
                    print(f"📖 [{idx}/{len(images)}] Processing '{img_name}' (Adaptive timeout: {current_timeout:.1f}s)…")

                start = time.time()
                raw_text, loop_detected = perform_ocr_with_api(img_path, current_timeout)
                elapsed = time.time() - start
                print(f"   ⏱️ OCR completed in {elapsed:.2f} seconds.")

                # Establish baseline from the first successfully processed page
                if first_page_time is None:
                    first_page_time = elapsed
                    print(f"   ℹ️ First page baseline time set to {first_page_time:.2f}s. Subsequent timeouts: {first_page_time * 2:.2f}s.")

                cleaned = clean_ocr_text(raw_text)
                if args.rtl:
                    # 對中文直排（從右到左）排版，將提取的行順序在 Python 端進行完全逆轉
                    lines = cleaned.split('\n')
                    lines.reverse()
                    cleaned = '\n'.join(lines)

                # --- Garbage text detection ---
                is_garbage, garbage_reason = is_garbage_text(cleaned, args.english)
                if is_garbage:
                    print(f"   🚨 [垃圾偵測警示] 此頁辨識結果疑似無效！原因：{garbage_reason}")
                    print(f"      📄 對應文字檔：{out_txt}")
                    garbage_images.append((img_name, out_txt, garbage_reason))

                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                processed += 1

                if loop_detected:
                    repetition_triggered_images.append(img_name)

                break  # success, move to next page
            except (TimeoutError, Exception) as e:
                print("\n🚨 OCR failed or timed out!", e)
                print("Please ensure Ollama is running, then press [Enter] to retry this page…")
                input()
                print("🔁 Retrying…")

    # Unified pre-merge warning block: consolidate all issues needing manual review
    needs_review = []

    for img in repetition_triggered_images:
        base = os.path.splitext(img)[0]
        txt_file = os.path.join(out_folder, f"{base}.txt")
        needs_review.append((img, txt_file, "死循環截斷（內容可能不完整）"))

    for img, txt_file, reason in garbage_images:
        # Avoid double-listing if an image was also caught by repetition detection
        if not any(img == r[0] for r in needs_review):
            needs_review.append((img, txt_file, reason))

    if needs_review:
        print("\n" + "="*60)
        print("🚨  以下圖檔的 OCR 結果需要您手動確認或修正後才能合併：")
        print()
        for img, txt_file, reason in needs_review:
            print(f"   🖼️  {img}")
            print(f"       原因：{reason}")
            print(f"       對應文字檔：{txt_file}")
            print()
        print("💡  建議您現在手動開啟上述的文字檔，參考原圖補齊或修正其內容。")
        print("="*60 + "\n")

        while True:
            resp = input("👉 請在手動修正完成後，輸入 'OK' (不區分大小寫) 進行最終合併：").strip()
            if resp.upper() == "OK":
                print("   [OK 收到，即將開始進行最終文字合併。]\n")
                break
            else:
                print("❌ 輸入內容不正確。請輸入 'OK' 以確認繼續。")

    # Merge all pages (including those already existed)
    final_path = os.path.join(".", f"{args.book_id}.txt")
    merge_texts(args.book_id, out_folder, images, final_path)
    print(f"\n🎉 Finished! Processed {processed} new pages. Combined book: {final_path}")

if __name__ == "__main__":
    main()
