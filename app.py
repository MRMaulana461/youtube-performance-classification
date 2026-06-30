"""
YouTube Video Performance Predictor
------------------------------------
Flask web app untuk melakukan inference model multimodal (text + image)
yang memprediksi performa video YouTube: rendah / sedang / tinggi.

Jalankan dengan:
    python app.py
"""

import os
import re
import pickle

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import nltk
from nltk.corpus import stopwords


# =========================================================
# 1. KONSTANTA & SETUP NLTK
# =========================================================

MAX_SEQ_LENGTH = 15
IMG_SIZE = (224, 224)
CLASS_ORDER = ['rendah', 'sedang', 'tinggi']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_PATH = 'final_multimodal_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'

# Pastikan resource NLTK stopwords tersedia.
# Kalau belum pernah didownload, nltk akan otomatis mengunduhnya sekali saja.
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    print("[INFO] NLTK stopwords belum tersedia, mengunduh resource 'stopwords'...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))


# =========================================================
# 2. INISIALISASI FLASK APP
# =========================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # batas upload 16MB
app.secret_key = 'ganti-dengan-secret-key-anda-sendiri'  # untuk flash message

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# =========================================================
# 3. LOAD MODEL & TOKENIZER (sekali saat startup)
# =========================================================

print("[INFO] Memuat model multimodal...")
model = load_model(MODEL_PATH)
print("[INFO] Model berhasil dimuat.")

print("[INFO] Memuat tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("[INFO] Tokenizer berhasil dimuat.")


# =========================================================
# 4. PREPROCESSING HELPER
# =========================================================

def allowed_file(filename):
    """Cek apakah ekstensi file termasuk yang diizinkan."""
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def clean_text(text):
    """
    Membersihkan teks judul video:
    - lowercase
    - hapus karakter selain huruf dan spasi
    - hapus stopwords bahasa Indonesia
    - hapus kata dengan panjang <= 1
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words and len(w) > 1])
    return text


def preprocess_text(title):
    """
    Pipeline preprocessing judul video menjadi array siap pakai model:
    1. clean_text
    2. tokenizer.texts_to_sequences
    3. pad_sequences sampai MAX_SEQ_LENGTH
    """
    cleaned_title = clean_text(title)
    sequence = tokenizer.texts_to_sequences([cleaned_title])
    padded = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH, padding='post')
    return padded


def preprocess_image(image_path):
    """
    Pipeline preprocessing thumbnail menjadi array siap pakai model:
    1. load image & resize ke IMG_SIZE
    2. convert ke array
    3. normalisasi / 255.0
    4. tambahkan batch dimension
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# =========================================================
# 5. ROUTE UTAMA
# =========================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    # ---------- POST: proses form ----------
    title = request.form.get('title', '').strip()
    image_file = request.files.get('thumbnail')

    # --- Validasi input ---
    if not title:
        flash('Judul video tidak boleh kosong.', 'error')
        return redirect(url_for('index'))

    if image_file is None or image_file.filename == '':
        flash('File thumbnail harus diupload.', 'error')
        return redirect(url_for('index'))

    if not allowed_file(image_file.filename):
        flash('Ekstensi file tidak valid. Gunakan PNG, JPG, atau JPEG.', 'error')
        return redirect(url_for('index'))

    try:
        # --- Simpan file dengan aman ---
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # --- Preprocessing ---
        text_array = preprocess_text(title)
        image_array = preprocess_image(image_path)

        # --- Prediksi ---
        predictions = model.predict({
            'image_input': image_array,
            'text_input': text_array
        })

        probs = predictions[0]  # array probabilitas per kelas, urutan sesuai CLASS_ORDER
        predicted_index = int(np.argmax(probs))
        predicted_label = CLASS_ORDER[predicted_index]
        confidence = float(probs[predicted_index]) * 100

        prob_rendah = float(probs[CLASS_ORDER.index('rendah')]) * 100
        prob_sedang = float(probs[CLASS_ORDER.index('sedang')]) * 100
        prob_tinggi = float(probs[CLASS_ORDER.index('tinggi')]) * 100

        return render_template(
            'index.html',
            predicted_label=predicted_label,
            confidence=confidence,
            prob_rendah=prob_rendah,
            prob_sedang=prob_sedang,
            prob_tinggi=prob_tinggi,
            uploaded_image=image_path.replace('\\', '/'),
            input_title=title
        )

    except Exception as e:
        # Error handling umum agar user tidak melihat stack trace mentah
        print(f"[ERROR] Prediksi gagal: {e}")
        flash('Terjadi kesalahan saat memproses prediksi. Silakan coba lagi.', 'error')
        return redirect(url_for('index'))


# =========================================================
# 6. ROUTE UNTUK MENAMPILKAN GAMBAR YANG DIUPLOAD
# =========================================================

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# =========================================================
# 7. RUN APP
# =========================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)