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
    """
    Extracts the trailing numeric page number from a filename for sorting.
    Example: '台灣AI大未來 - 100.jpg' -> 100
    """
    base = os.path.splitext(filename)[0]
    match = re.findall(r'\d+', base)
    if match:
        return int(match[-1])
    return 999999

def clean_ocr_text(text: str) -> str:
    """
    Cleans up OCR markdown markers, preambles, and e-reader top/bottom status bars.
    """
    # 1. Clean markdown code blocks
    text = re.sub(r'^```[a-zA-Z]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
    
    # 2. Clean chatbot introductory phrases
    intro_phrases = [
        r'^\s*Here is the extracted text.*:\s*$',
        r'^\s*Extracted text.*:\s*$',
        r'^\s*以下是圖片中的文字.*:\s*$',
        r'^\s*解析結果.*:\s*$'
    ]
    for phrase in intro_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 3. Clean status bars (time, battery, UI elements)
    lines = text.split('\n')
    cleaned_lines = []
    
    time_pattern = re.compile(r'^\s*(AM|PM)?\s*\d{1,2}:\d{2}\s*$', re.IGNORECASE)
    battery_pattern = re.compile(r'^\s*\d{1,3}%\s*$')
    ui_pattern = re.compile(r'^\s*(AA|[‹›<>|])\s*$', re.IGNORECASE)
    
    for i, line in enumerate(lines):
        # We only check for status bar noise at the very top (first 5 lines) or very bottom (last 5 lines)
        is_top_or_bottom = (i < 5) or (i >= len(lines) - 5)
        stripped = line.strip()
        
        if is_top_or_bottom:
            if time_pattern.match(stripped) or battery_pattern.match(stripped) or ui_pattern.match(stripped):
                continue # Skip this line
        
        cleaned_lines.append(line)
        
    result = '\n'.join(cleaned_lines)
    return result.strip()

def perform_ocr_with_api(image_path: str, timeout: int) -> str:
    """
    Directly calls the local Ollama HTTP API endpoint to perform OCR.
    """
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
        raise TimeoutError(f"Ollama API request timed out or was unreachable: {e}")

def run_unit_tests():
    """
    Runs self-contained unit tests to verify sorting and filtering logic.
    """
    print("🧪 Running unit tests...")
    
    # Test Sorting
    print("  - Testing page number extraction and sorting...")
    test_files = ["台灣AI大未來 - 10.jpg", "台灣AI大未來 - 2.jpg", "台灣AI大未來 - 100.jpg", "台灣AI大未來 - 1.jpg"]
    sorted_files = sorted(test_files, key=extract_page_number)
    expected_sorted = ["台灣AI大未來 - 1.jpg", "台灣AI大未來 - 2.jpg", "台灣AI大未來 - 10.jpg", "台灣AI大未來 - 100.jpg"]
    assert sorted_files == expected_sorted, f"Sorting failed. Got: {sorted_files}"
    print("    [PASS] Filename sorting works correctly!")
    
    # Test Cleanup
    print("  - Testing text cleanup and filtering...")
    raw_ocr_output = """```markdown
AM2:07
73%
AA
技術視角，見證這波生成式AI如何覆舊有的AI思潮流。

那時，我們還在懷疑AI的對話能力。
```"""
    cleaned = clean_ocr_text(raw_ocr_output)
    expected_cleaned = "技術視角，見證這波生成式AI如何覆舊有的AI思潮流。\n\n那時，我們還在懷疑AI的對話能力。"
    assert cleaned == expected_cleaned, f"Cleanup failed. Got:\n{cleaned}"
    print("    [PASS] Text cleanup and filtering works correctly!")
    
    print("🎉 All unit tests passed successfully!\n")

def merge_texts(book_id: str, out_txt_dir: str, sorted_filenames: list, final_out_path: str):
    """
    Combines all single page text files in numeric order into a single final document.
    """
    print(f"\nMerging all page texts into {final_out_path}...")
    merged_content = []
    
    for filename in sorted_filenames:
        base_name = os.path.splitext(filename)[0]
        page_txt_path = os.path.join(out_txt_dir, f"{base_name}.txt")
        
        if os.path.exists(page_txt_path):
            with open(page_txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    merged_content.append(content)
        else:
            print(f"Warning: Missing text file for page {filename}")
            
    combined = "\n\n".join(merged_content)
    
    with open(final_out_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✨ Merged successfully! Combined file saved at: {final_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Scanned Book OCR Processor & Integrator using Ollama API")
    parser.add_argument("book_id", type=str, help="The ID of the book (e.g., B001, B002)")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of pages to process in this run")
    parser.add_argument("-t", "--timeout", type=int, default=180, help="Timeout in seconds for OCR request (default: 180)")
    parser.add_argument("--scan-dir", type=str, default="SCAN_PAGES", help="Base directory for scanned images")
    parser.add_argument("--out-dir", type=str, default="TXT", help="Base directory for output text files")
    parser.add_argument("--test", action="store_true", help="Run self-contained unit tests and exit")
    
    args = parser.parse_args()
    
    if args.test:
        run_unit_tests()
        sys.exit(0)
        
    book_id = args.book_id
    scan_base = args.scan_dir
    out_base = args.out_dir
    timeout = args.timeout
    
    if not os.path.exists(scan_base):
        print(f"Error: Base directory '{scan_base}' does not exist.")
        sys.exit(1)
        
    # Find matching directory for the book ID (e.g. B001_台灣AI大未來)
    matching_dirs = [d for d in os.listdir(scan_base) if d.startswith(book_id) and os.path.isdir(os.path.join(scan_base, d))]
    if not matching_dirs:
        print(f"Error: No directory found starting with '{book_id}' in '{scan_base}'.")
        sys.exit(1)
        
    book_dir_name = matching_dirs[0]
    book_dir_path = os.path.join(scan_base, book_dir_name)
    print(f"📂 Found book directory: {book_dir_path}")
    
    # Gather image files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [f for f in os.listdir(book_dir_path) if f.lower().endswith(valid_extensions)]
    
    if not all_images:
        print(f"Error: No images found in directory '{book_dir_path}'.")
        sys.exit(1)
        
    # Sort files numerically based on the trailing page number
    all_images.sort(key=extract_page_number)
    print(f"🔢 Detected {len(all_images)} scanned page images.")
    
    # Establish output paths
    out_txt_dir = os.path.join(out_base, book_id)
    os.makedirs(out_txt_dir, exist_ok=True)
    
    # Filter/Slice by limit if set
    processed_count = 0
    pages_to_process = []
    
    for img in all_images:
        base_name = os.path.splitext(img)[0]
        out_txt_path = os.path.join(out_txt_dir, f"{base_name}.txt")
        
        # If already processed, we skip it
        if os.path.exists(out_txt_path):
            continue
            
        pages_to_process.append(img)
        if args.limit and len(pages_to_process) >= args.limit:
            break
            
    print(f"⚡ {len(all_images) - len(pages_to_process)} pages already processed. {len(pages_to_process)} pages remaining to process.")
    
    total_to_process = len(pages_to_process)
    
    for idx, filename in enumerate(pages_to_process):
        image_path = os.path.join(book_dir_path, filename)
        base_name = os.path.splitext(filename)[0]
        out_txt_path = os.path.join(out_txt_dir, f"{base_name}.txt")
        
        # Keep retrying until success (in case of Ollama hangs)
        while True:
            try:
                print(f"📖 [{idx+1}/{total_to_process}] Processing '{filename}'...")
                start_time = time.time()
                
                # Perform OCR
                raw_text = perform_ocr_with_api(image_path, timeout)
                
                elapsed = time.time() - start_time
                print(f"    ✅ Success! OCR finished in {elapsed:.2f} seconds.")
                
                # Clean and filter text
                cleaned_text = clean_ocr_text(raw_text)
                
                # Save individual page text
                with open(out_txt_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                
                processed_count += 1
                break # Break inner retry loop on success
                
            except (TimeoutError, Exception) as e:
                print("\n" + "="*65)
                print(f"🚨 [WARNING] OCR processing failed or timed out for '{filename}'!")
                print(f"Error details: {e}")
                print("\nThis might indicate that your local Ollama service is hung or offline.")
                print("Please follow these steps to recover:")
                print("  1. Check if Ollama is running.")
                print("  2. Restart the Ollama application / server if necessary.")
                print("  3. Make sure the 'deepseek-ocr' model is available.")
                print("="*65)
                print("\a", end="") # Sound terminal bell
                input("\n👉 Once Ollama is running correctly, press [Enter] to retry this page...")
                print("🔄 Retrying page OCR...")
                
    # Finally, merge all text files of the book in numeric order
    final_out_path = os.path.join(".", f"{book_id}.txt")
    merge_texts(book_id, out_txt_dir, all_images, final_out_path)
    print(f"\n🎉 Process finished! {processed_count} new pages processed.")

if __name__ == "__main__":
    main()
