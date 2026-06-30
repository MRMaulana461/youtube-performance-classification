# YouTube Video Performance Predictor

Flask web app untuk inference model deep learning multimodal (judul video + thumbnail)
yang memprediksi performa video: **rendah**, **sedang**, atau **tinggi**.

## Struktur Project

```
project/
│
├── app.py
├── final_multimodal_model.h5   <-- LETAKKAN model Anda di sini
├── tokenizer.pkl                <-- LETAKKAN tokenizer Anda di sini
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── uploads/                     <-- otomatis dibuat saat app dijalankan
```

## Cara Menjalankan

### 1. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 2. Install dependency

```bash
pip install -r requirements.txt
```

### 3. Letakkan file model Anda

Copy file berikut ke folder root project (sejajar dengan `app.py`):

- `final_multimodal_model.h5`
- `tokenizer.pkl`

### 4. Download NLTK stopwords (sekali saja)

App akan otomatis mengunduh resource ini saat pertama kali dijalankan kalau belum ada.
Tapi jika ingin manual:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 5. Jalankan app

```bash
python app.py
```

Buka browser ke: **http://127.0.0.1:5000**

## Catatan Penting

- Pipeline preprocessing text & image **sudah disesuaikan** dengan training Anda
  (`MAX_SEQ_LENGTH=15`, `IMG_SIZE=(224,224)`, stopwords bahasa Indonesia dari NLTK).
- Nama input ke model tetap `image_input` dan `text_input` — **jangan diubah** kalau
  arsitektur model Anda mengharapkan nama tersebut.
- Urutan kelas mengikuti `CLASS_ORDER = ['rendah', 'sedang', 'tinggi']`. Jika urutan
  output model Anda berbeda, sesuaikan konstanta ini di `app.py`.
- File yang diupload disimpan di folder `uploads/` menggunakan `secure_filename`
  untuk keamanan dasar.
- Untuk demo publik (bukan hanya localhost), matikan `debug=True` di `app.py`
  sebelum deploy.
