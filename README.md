
---

```markdown
# 🧠 Persian RAG Assistant | دستیار تحقیقاتی با زبان فارسی

یک برنامه‌ی قدرتمند تحت وب با استفاده از Streamlit و LangChain برای استخراج و پاسخ‌دهی به سؤالات از روی فایل‌های PDF به زبان فارسی یا هر زبان دیگر 🌍

## ✨ امکانات

- پشتیبانی از زبان فارسی (با تشخیص خودکار زبان متن)
- آپلود فایل PDF و پردازش محتوای آن
- استفاده از مدل‌های زبانی (OpenAI یا جایگزین‌ها) برای پاسخ‌دهی به سؤالات
- نمایش پاسخ به صورت تعاملی در رابط کاربری زیبا و ساده

## 🧰 تکنولوژی‌های استفاده‌شده

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI GPT](https://platform.openai.com/)
- [HuggingFace Embeddings](https://huggingface.co/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [langdetect](https://pypi.org/project/langdetect/)

## 🚀 اجرا به صورت محلی

### 1. کلون کردن پروژه

```bash
git clone https://github.com/your-username/persian-rag-assistant.git
cd persian-rag-assistant
```

### 2. ساخت محیط مجازی و نصب وابستگی‌ها

```bash
python -m venv venv
source venv/bin/activate  # برای ویندوز: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔍 بررسی و توضیح `app.py`

### 1. **ایمپورت کتابخانه‌ها**
```python
import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langdetect import detect
```
- برای رابط کاربری از `streamlit` استفاده شده.
- `os` و `tempfile` برای مدیریت فایل موقتی.
- `PdfReader` برای استخراج متن از فایل‌های PDF.
- `detect` برای تشخیص زبان متن PDF.

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
```
- ماژول‌های `langchain` برای پردازش متن و اتصال به مدل زبانی و بردارها استفاده می‌شن:
  - `OpenAI`: مدل زبانی.
  - `RetrievalQA`: زنجیره‌ای برای پرسش‌وپاسخ.
  - `HuggingFaceEmbeddings`: تبدیل متن به بردار.
  - `Chroma`: دیتابیس برداری برای جستجو.
  - `TextSplitter`: تقسیم متن به قطعات کوچک‌تر.

---

### 2. **دریافت کلید API**
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
```
- کلید API از متغیر محیطی گرفته می‌شه. اگر نباشه، از مقدار پیش‌فرض استفاده می‌کنه (که باید جایگزین بشه).

---

### 3. **رابط کاربری با Streamlit**
```python
st.set_page_config(...)
st.title(...)
uploaded_file = st.file_uploader(...)
user_query = st.text_input(...)
```
- تنظیم عنوان، اپلود فایل و دریافت سوال کاربر.

---

### 4. **توابع کمکی**
#### استخراج متن از PDF:
```python
def extract_text_from_pdf(file):
    ...
```

#### تشخیص زبان:
```python
def detect_language(text):
    ...
```
- از `langdetect` برای تشخیص زبان متن استفاده می‌شه.

---

### 5. **منطق اصلی برنامه**
```python
if uploaded_file and user_query:
```
- فقط وقتی فایل PDF و سوال وارد شده باشه، پردازش انجام می‌شه.

#### ✅ مراحل داخل شرط:
1. **ذخیره موقت فایل PDF**
   ```python
   with tempfile.NamedTemporaryFile(...) as tmp_file:
       ...
   ```

2. **استخراج متن و تشخیص زبان**
   ```python
   text = extract_text_from_pdf(...)
   lang = detect_language(text)
   ```

3. **تقسیم متن**
   ```python
   splitter = CharacterTextSplitter(...)
   texts = splitter.split_text(text)
   ```

4. **تبدیل متن به بردار و ساخت پایگاه داده برداری**
   ```python
   embeddings = HuggingFaceEmbeddings(...)
   vectorstore = Chroma.from_texts(...)
   ```

5. **ساخت پرامپت مناسب با زبان**
   ```python
   prompt = ...  # فارسی یا زبان دیگر
   ```

6. **اتصال به مدل زبانی و ایجاد RetrievalQA**
   ```python
   llm = OpenAI(...)
   qa = RetrievalQA.from_chain_type(...)
   ```

7. **اجرای پرسش و نمایش پاسخ**
   ```python
   response = qa.run(prompt)
   st.write(response)
   ```

---

## 📌 خلاصه‌ی عملکرد
1. کاربر فایل PDF آپلود می‌کنه و سوال می‌پرسه.
2. متن استخراج می‌شه و زبان تشخیص داده می‌شه.
3. متن به قطعات کوچکتر تقسیم و به بردار تبدیل می‌شه.
4. سیستم با استفاده از مدل زبانی (مثل GPT) و اطلاعات استخراج‌شده، پاسخ رو تولید می‌کنه.
5. پاسخ به‌صورت متنی نمایش داده می‌شه.

---



## 📜 مجوز

این پروژه تحت مجوز [MIT](LICENSE) ارائه شده است.

---

🧑‍💻 ساخته‌شده با عشق برای فارسی‌زبان‌ها ♥  
```

---

