# PTIT 1b — Spam/Ham

Mở **[PTIT_spam_ham.ipynb](PTIT_spam_ham.ipynb)** để xem bài tập lớn hoàn chỉnh: nguồn dữ liệu, quy tắc gán nhãn, tiền xử lý tiếng Việt, BoW/TF-IDF, Naive Bayes/SVM, Accuracy/Precision/Recall/F1, confusion matrix và kết luận. Notebook đã chạy và lưu sẵn kết quả.

Dữ liệu: `comments.csv` là dữ liệu thô tăng thêm khi crawler chạy; `labeled_comments.csv` là bộ nhãn thủ công ban đầu (nếu còn file); `auto_labeled_comments.csv` là đầu ra gán nhãn hàng loạt. Nhãn chỉ là bản sơ bộ, cần nhóm rà soát với ngữ cảnh bài viết gốc. Hiện cả bốn mô hình đều bỏ sót spam trong tập test; xem phần cuối notebook trước khi diễn giải Accuracy.

Để chạy lại: `pip install -r requirements.txt`, mở notebook và chọn **Run All**. Nếu cần thu thập thêm từ ba group, xem `crawl_comments.py`; không đăng nhập: `python crawl_comments.py --no-login`.

Crawl mục tiêu 1.000 dòng: `python crawl_comments.py --target 1000`. Đăng nhập trực tiếp trong cửa sổ mở ra, rồi nhấn Enter ở terminal. Script dùng lại phiên trong `.fb-profile/` và chỉ ghi cột `content` vào `comments.csv`. Không đưa mật khẩu vào mã nguồn.

## Gán nhãn tự động bằng LLM

`label_comments.py` gửi nhiều bình luận trong mỗi request tới API Chat Completions tương thích OpenAI; model chỉ trả về chỉ số của spam, các dòng khác là ham. Script dùng lại nhãn trong `labeled_comments.csv` nếu file đó còn tồn tại, lưu sau mỗi lô và chạy tiếp được khi gián đoạn. Với 4.161 dòng hiện tại, cần khoảng **28 lô gọi** khi phản hồi hợp lệ; phản hồi sai sẽ được chia lô và thử lại.

Điền `API_KEY`, `BASE_URL`, `MODEL` ở đầu `label_comments.py`, rồi chạy:

```bash
python label_comments.py
```

Có thể đặt `DRY_RUN = True` để chỉ xem số lô dự kiến; `BATCH_SIZE`, `MAX_CHARS` và `MAX_SECONDS` điều chỉnh giới hạn mỗi lô. Lô nhỏ hơn giúp model miễn phí phản hồi nhanh hơn nhưng cần nhiều lượt gọi hơn.

Kết quả nằm trong `auto_labeled_comments.csv` (`content,label`). Nếu một bình luận đơn lẻ vẫn nhận phản hồi sai, nhãn `review` đánh dấu dòng cần kiểm tra thủ công và script tiếp tục. Nếu model có context nhỏ hoặc trả lỗi, đổi `BATCH_SIZE = 200`, `MAX_CHARS = 12000` rồi chạy lại; các lô đã xong không bị gọi lại. Đây là nhãn máy từ văn bản đơn lẻ, cần kiểm tra thủ công trước khi dùng làm dữ liệu đánh giá.
