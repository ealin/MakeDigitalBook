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
    # 3. Remove status‑bar noise (time, battery, UI tokens) – only in top/bottom lines
    lines = text.split('\n')
    cleaned_lines = []
    time_pattern = re.compile(r'^\s*(AM|PM)?\s*\d{1,2}:\d{2}\s*$', re.IGNORECASE)
    battery_pattern = re.compile(r'^\s*\d{1,3}%\s*$')
    ui_pattern = re.compile(r'^\s*(AA|[‹›<>|])\s*$', re.IGNORECASE)
    a4_pattern = re.compile(r'^\s*A4\s*$', re.IGNORECASE)  # NEW: single line "A4"
    for i, line in enumerate(lines):
        is_top_or_bottom = (i < 5) or (i >= len(lines) - 5)
        stripped = line.strip()
        if is_top_or_bottom:
            if (time_pattern.match(stripped) or battery_pattern.match(stripped) or
                ui_pattern.match(stripped) or a4_pattern.match(stripped)):
                continue  # drop this line
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines).strip()
    return result

def perform_ocr_with_api(image_path: str, timeout: float = None) -> str:
    """Call Ollama HTTP API for deepseek-ocr.
    Returns the raw OCR text (may contain markdown etc.)."""
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "model": "deepseek-ocr",
        "prompt": "Extract the text in the image.",
        "images": [img_data],
        "stream": False
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res = json.loads(response.read().decode('utf-8'))
            return res.get("response", "")
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
    # Noise filtering test – includes the new "A4" line
    raw = """```markdown\nAM2:07\n73%\nAA\nA4\n技術視角，見證這波生成式AI如何覆舊有的AI思潮流。\n```"""
    cleaned = clean_ocr_text(raw)
    expected_clean = "技術視角，見證這波生成式AI如何覆舊有的AI思潮流。"
    assert cleaned == expected_clean, f"Cleanup failed: {cleaned}"
    print("   [PASS] Noise filtering (including A4) works.")
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
    parser.add_argument("--test", action="store_true",
                        help="Run internal unit tests and exit")
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

    # Prepare output folder
    out_folder = os.path.join(args.out_dir, args.book_id)
    os.makedirs(out_folder, exist_ok=True)

    # Apply limit if given
    if args.limit is not None:
        images = images[:args.limit]
        print(f"🚦 Limiting to first {args.limit} pages for this run.")

    processed = 0
    first_page_time = None  # Track the processing time of the first successfully completed OCR page in this run
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
                raw_text = perform_ocr_with_api(img_path, current_timeout)
                elapsed = time.time() - start
                print(f"   ⏱️ OCR completed in {elapsed:.2f} seconds.")

                # Establish baseline from the first successfully processed page
                if first_page_time is None:
                    first_page_time = elapsed
                    print(f"   ℹ️ First page baseline time set to {first_page_time:.2f}s. Subsequent timeouts: {first_page_time * 2:.2f}s.")

                cleaned = clean_ocr_text(raw_text)
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                processed += 1
                break  # success, move to next page
            except (TimeoutError, Exception) as e:
                print("\n🚨 OCR failed or timed out!", e)
                print("Please ensure Ollama is running, then press [Enter] to retry this page…")
                input()
                print("🔁 Retrying…")

    # Merge all pages (including those already existed)
    final_path = os.path.join(".", f"{args.book_id}.txt")
    merge_texts(args.book_id, out_folder, images, final_path)
    print(f"\n🎉 Finished! Processed {processed} new pages. Combined book: {final_path}")

if __name__ == "__main__":
    main()
