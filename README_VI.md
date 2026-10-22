# XLM-R large — BTC only

Giải nén vào thư mục btc_only dưới dự án hiện tại. Mở thư mục đó trong VS Code,
mở xlmr_btc_only.ipynb, chọn môi trường .venv (Python3.11.9) hiện có trong ảnh của bạn.
Chạy cell kiểm tra GPU trước. Môi trường không cần có tên .venv-xlmr.

Nếu thiếu thư viện, mở terminal tại nơi chứa .venv và cài bằng interpreter đó:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r btc_only\requirements.txt
```

Restart kernel sau cài. Đường dẫn requirements điều chỉnh nếu bạn giải nén khác thư mục.
Ba CSV là nguyên bản BTC đã được người dùng cung cấp. Chỉ train bằng train.csv BTC;
không có code tải dataset ngoài hoặc nạp external adapter. Backbone gốc whitelist vẫn cần tải.
Thứ tự: GPU → config → split → backbone → tokenize → smoke → train → evaluate → predict.
LoRA, không full fine-tuning; chưa benchmark RTX5060. Không tự nộp Kaggle.

Dùng CLAUDE_BTC_ONLY.md để thay yêu cầu giao Claude trước đó.
