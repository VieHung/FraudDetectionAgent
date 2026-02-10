# FraudDetectionAgent 🕵️‍♂️💸

**FraudDetectionAgent** là hệ thống Agentic AI phát hiện gian lận tài chính thông minh. Hệ thống kết hợp giữa **Quy tắc cứng (Hard Rules)** để xử lý nhanh và **AI Generative (LLM)** để suy luận các trường hợp phức tạp, tối ưu hóa giữa chi phí và độ chính xác.

## 🏗 Kiến trúc hệ thống (Hybrid Architecture)

Dự án được thiết kế theo luồng xử lý 4 lớp, bổ sung cơ chế **Circuit Breaker** (Layer 2.5):

1. **Layer 1 - Ingestion:** Tiếp nhận và chuẩn hóa dữ liệu giao dịch.
2. **⚡ Layer 2.5 - Circuit Breaker (Fast Rules):**
* **Mục tiêu:** Lọc ngay lập tức 20-30% giao dịch rõ ràng (VD: IP Blacklist, whitelist nội bộ) mà **KHÔNG** gọi tới AI.
* **Lợi ích:** Giảm độ trễ (Latency) và tiết kiệm chi phí Token cho LLM.


3. **Layer 2 - Analysis Support:** Các mô hình bổ trợ (NLP, Behavioral Scoring) cung cấp thông tin cho AI.
4. **Layer 3 - The Brain (AI Agent):** Chỉ được kích hoạt với các giao dịch "vùng xám" (nghi ngờ). AI sẽ tổng hợp dữ liệu để đưa ra phán quyết cuối cùng.
5. **Layer 4 - Actions:** Thực thi quyết định (Block, OTP, Alert).

---

## 📂 Cấu trúc dự án & Bản đồ trách nhiệm

Dưới đây là mapping giữa file code và các tính năng. **Thành viên nhóm vui lòng check kỹ trước khi code.**

```plaintext
src/react_agent
├── ingestion/               # [Layer 1] TEAM DATA
│   ├── schemas.py           # 👉 Định nghĩa Input (Transaction Model)
│   └── loader.py            # 👉 Logic load data
│
├── analytics/               # [Layer 2 & 2.5] TEAM DATA SCIENCE
│   ├── rules.py             # ⚡ [Layer 2.5] Chứa logic Circuit Breaker (Hàm fast_check trả về BLOCK/PASS ngay)
│   ├── behavioral.py        # 👉 [Layer 2] Logic tính điểm hành vi (cho AI tham khảo)
│   └── nlp.py               # 👉 [Layer 2] Logic phân tích nội dung message
│
├── actions/                 # [Layer 4] TEAM BACKEND
│   ├── notifications.py     # 👉 Code gửi Email/SMS/OTP
│   └── account_ops.py       # 👉 Code Lock tài khoản/Update DB
│
├── graph.py                 # [Layer 3] TEAM AGENT ENGINEER (QUAN TRỌNG)
│                            # 👉 Định nghĩa luồng đi: Check Rule -> (Nếu Pass/Block) -> End.
│                            #                          (Nếu Nghi ngờ) -> Gọi AI Agent.
│
├── prompts.py               # 🧠 TEAM PROMPT ENGINEERING
│                            # 👉 System Prompt cho AI Agent xử lý các ca khó.
│
├── tools.py                 # 🌉 CẦU NỐI (BRIDGE)
│                            # 👉 Đăng ký các hàm từ analytics/ và actions/ thành @tool.
│
└── utils/                   # TIỆN ÍCH CHUNG

```

---

## 🚀 Hướng dẫn phát triển (Developer Guide)

### 1. Phát triển Layer 2.5 (Fast Rules / Circuit Breaker)

* **Mục tiêu:** Thêm các luật chặn cứng/cho qua cứng.
* **File cần sửa:** `src/react_agent/analytics/rules.py`
* **Cách làm:** Viết hàm trả về trạng thái dứt khoát.
```python
def check_global_blacklist(ip):
    if ip in BLACKLIST: return "BLOCK"
    return "UNKNOWN" # Để đẩy sang cho AI xử lý

```


* **Lưu ý:** Logic này được gọi trực tiếp trong `graph.py` trước khi khởi động Agent.

### 2. Phát triển Layer 3 (AI Agent Reasoning)

* **Mục tiêu:** Giúp AI thông minh hơn trong việc xử lý các ca nghi ngờ.
* **File cần sửa:**
* `src/react_agent/analytics/*.py`: Viết thêm các hàm phân tích (VD: soi lịch sử 3 tháng).
* `src/react_agent/tools.py`: Đăng ký hàm đó thành Tool.
* `src/react_agent/prompts.py`: Dạy AI cách dùng Tool đó.



### 3. Phát triển Layer 4 (Actions)

* **Mục tiêu:** Tương tác với hệ thống bên ngoài.
* **File cần sửa:** `src/react_agent/actions/`.

---

## 🛠 Cài đặt & Chạy dự án (Setup)

### 1. Yêu cầu môi trường

* Python 3.11+
* [UV](https://github.com/astral-sh/uv) (khuyến nghị) hoặc Pip.

### 2. Cài đặt dependencies

```bash
# Clone repo
git clone <your-repo-url>
cd FraudDetectionAgent

# Cài đặt môi trường ảo
uv sync --frozen

```

### 3. Cấu hình biến môi trường

Copy file mẫu và điền API Key:

```bash
cp .env.example .env

```

```ini
ANTHROPIC_API_KEY=sk-ant-...  # Dùng cho AI Agent (Layer 3)
TAVILY_API_KEY=tvly-...       # Dùng cho Search Tool (nếu cần)

```

### 4. Chạy thử (Demo)

Script demo sẽ chạy giả lập 1 giao dịch để test luồng đi (Rule -> Agent -> Action):

```bash
python scripts/demo_graph.py

```

---

## 🔄 Workflow đóng góp code

1. **Check Issue:** Xem task thuộc Layer nào (2.5, 3, hay 4).
2. **Branching:** Tạo nhánh theo format `feat/[layer]-tên-tính-năng`.
* VD: `feat/L2.5-ip-blacklist` hoặc `feat/L3-sentiment-analysis`.


3. **Testing:**
* Nếu sửa Layer 2.5: Đảm bảo các case rõ ràng bị chặn ngay lập tức (Check log không thấy gọi LLM).
* Nếu sửa Layer 3: Đảm bảo AI suy luận có lý do (Reasoning trace).


4. **Pull Request:** Review chéo trước khi merge vào `main`.

Happy Coding! 🚀