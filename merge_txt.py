#!/usr/bin/env python3
import os
import sys
import re
import argparse

def extract_page_number(filename: str) -> int:
    """Extract trailing numeric page number from filename for correct sorting.
    Example: '台灣AI大未來 - 100.txt' -> 100"""
    base = os.path.splitext(filename)[0]
    match = re.findall(r'\d+', base)
    if match:
        return int(match[-1])
    return 999999

def merge_txt_files(input_dir: str, output_file: str):
    """Sort and merge all txt files in input_dir into output_file."""
    if not os.path.isdir(input_dir):
        print(f"❌ 找不到目錄: '{input_dir}'")
        sys.exit(1)

    # 取得目錄下所有 txt 檔案
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    if not txt_files:
        print(f"⚠️  在 '{input_dir}' 中找不到任何 .txt 檔案。")
        sys.exit(1)

    # 依照檔名尾部的數字進行自然排序
    txt_files.sort(key=extract_page_number)
    print(f"📂 找到 {len(txt_files)} 個文字檔，即將開始進行順序合併...")

    merged_content = []
    for file_name in txt_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    merged_content.append(content)
                    print(f"   ➕ 已讀取: '{file_name}'")
        except Exception as e:
            print(f"❌ 讀取 '{file_name}' 時發生錯誤: {e}")

    # 合併寫出
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(merged_content))
        print(f"✅ 合併成功！合併後的檔案已儲存至: '{output_file}' (共 {len(merged_content)} 頁)")
    except Exception as e:
        print(f"❌ 寫入合併檔案 '{output_file}' 時發生錯誤: {e}")

def main():
    parser = argparse.ArgumentParser(description="合併指定目錄下的所有 txt 檔案（依數字順序自然排序）")
    parser.add_argument("input_dir", type=str, help="輸入的 txt 目錄，例如 TXT/B001")
    parser.add_argument("output_file", type=str, help="合併後的輸出檔名，例如 B001.txt")
    args = parser.parse_args()

    merge_txt_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
