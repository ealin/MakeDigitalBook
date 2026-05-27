#!/usr/bin/env python3
import os
import sys
import time

# 匯入現有的 ocr_processor 中的函式
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ocr_processor import perform_ocr_with_api, clean_ocr_text, extract_page_number
except ImportError as e:
    print(f"❌ 無法匯入 ocr_processor.py: {e}")
    sys.exit(1)

def main():
    scan_dir = "SCAN_PAGES/B002_量子喜樂"
    if not os.path.isdir(scan_dir):
        print(f"❌ 找不到目錄: '{scan_dir}'")
        sys.exit(1)

    print("🚀 [量子喜樂 B002] 開始進行 OCR 前五頁測試...")

    # 取得圖片列表
    valid_ext = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(scan_dir) if f.lower().endswith(valid_ext)]
    if not images:
        print("❌ 目錄中沒有圖片檔案。")
        sys.exit(1)

    # 依頁碼排序
    images.sort(key=extract_page_number)
    
    # 取前五張
    test_images = images[:5]
    print(f"🔢 排序後的前五個測試檔案分別為:")
    for idx, img in enumerate(test_images, 1):
        print(f"   {idx}. {img} (頁碼: {extract_page_number(img)})")
    print("-" * 50)

    # 依序呼叫 OCR
    for idx, img_name in enumerate(test_images, 1):
        img_path = os.path.join(scan_dir, img_name)
        print(f"\n📖 [{idx}/5] 正在處理: '{img_name}'...")
        
        start_time = time.time()
        try:
            # 測試調用，第一頁與後續頁面均以無超時限制(None)發送以確保首次加載成功
            raw_text, loop_detected = perform_ocr_with_api(img_path, timeout=None)
            elapsed = time.time() - start_time
            print(f"   ⏱️  OCR 完成，耗時 {elapsed:.2f} 秒 (死循環偵測: {loop_detected})")
            
            cleaned_text = clean_ocr_text(raw_text)
            print("\n📝 --- 辨識文字內容 ---")
            if cleaned_text.strip():
                print(cleaned_text)
            else:
                print("[無文字或全為噪音被過濾]")
            print("-" * 50)
            
        except Exception as e:
            print(f"🚨 處理失敗: {e}")
            print("請確認 Ollama 服務是否正在運行！")
            sys.exit(1)

    print("\n🎉 前五頁測試已完成！請校對上述印出的文字內容。")

if __name__ == "__main__":
    main()
