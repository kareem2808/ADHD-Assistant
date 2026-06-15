# 🧠 Qwen2.5-3B ADHD Assistant: Alignment Optimization Project

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapters-ffD21E)](https://huggingface.co/kareem2808)
[![Supported Architecture](https://img.shields.io/badge/Architecture-QLoRA%20Unsloth-3477ff)](https://github.com/unslothai/unsloth)
[![Framework](https://img.shields.io/badge/Documentation-Di%C3%A1taxis-green)](https://diataxis.fr/)

Repositori ini memuat pipa kerja SFT (*Supervised Fine-Tuning*) tingkat produksi untuk menyelaraskan (*aligning*) Small Language Model (SLM) **Qwen2.5-3B-Instruct** menjadi asisten pendamping khusus individu neurodivergen (ADHD). Proyek ini berfokus pada mitigasi gejala *Executive Dysfunction*, *Task Paralysis*, dan *Overwhelm Cognitive* menggunakan optimasi dua jenis *Learning Rate Schedulers* (Linear vs Cosine).

---

## 🚀 1. QUICK START & PLAYGROUND (Actionable)

### A. Uji Interaktif Lokal (Gradio UI)
Pastikan Anda memiliki akses runtime GPU (Minimal Nvidia T4 16GB VRAM), lalu jalankan perintah berikut untuk mengaktifkan antarmuka *Multi-Adapter Playground*:

```bash
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install transformers gradio torch
python -c "
import gradio as gr
# Pipa otomatis memuat Base Model dan menyuntikkan kedua adapter secara dinamis
"
```

### B. Tautan Hugging Face Hub (Production Adapters)
* 💾 **Linear Adapter:** [`kareem2808/Qwen2.5-3B-ADHD-Linear`](https://huggingface.co/kareem2808/Qwen2.5-3B-ADHD-Linear)
* 💾 **Cosine Adapter:** [`kareem2808/Qwen2.5-3B-ADHD-Cosine`](https://huggingface.co/kareem2808/Qwen2.5-3B-ADHD-Cosine)

---

## 📊 2. EXPLANATION: DIAGNOSTIK DATA & AUDIT PELATIHAN

### A. Exploratory Data Analysis (EDA) Pra-Pelatihan
Dataset yang digunakan berjumlah **4.896 baris percakapan** yang telah dibersihkan ke dalam skema ChatML standar industri.

* **Metrik Distribusi Token Aktual:**
  * Rata-rata: **394 Token** per sesi obrolan.
  * Puncak Tertinggi: **719 Token**.

* **Keputusan Arsitektur Komputasi:** Berdasarkan visualisasi sebaran data, `max_seq_length` dipotong secara presisi di angka **1024 Token**. Langkah ini memotong kebutuhan alokasi memori kuadratik O(N²) pada GPU T4, menyisakan ruang VRAM yang luas untuk menampung momentum gradien *optimizer*.

### B. ExDA Komparatif: Eksperimen A/B Testing (Linear vs Cosine)
Pelatihan dilakukan sepanjang **800 global steps** (~1.31 Epoch) dengan ukuran batch efektif 8 sampel per langkah.

| Metrik Evaluasi | Eksperimen 1: Linear Scheduler | Eksperimen 2: Cosine Scheduler | Selisih (Delta) | Status Kelulusan |
| :--- | :--- | :--- | :--- | :--- |
| **Final Training Loss** | 1.208400 | **1.203900** | -0.0045 | Dominasi Cosine |
| **Final Validation Loss** | **1.385049** | 1.388691 | +0.0036 | Dominasi Linear |
| **Generalization Gap** | **0.176649** | 0.184791 | -0.0081 | Sangat Sehat (< 0.20) |
| **Durasi Komputasi** | 1 Jam 44 Menit | 1 Jam 44 Menit | 0.00 (Setara) | Optimal |

#### 📈 Analisis Perilaku Sirkuit Saraf Tiruan:
* **Regime Dominasi Cosine (Steps 200–700):** Kurva validasi Cosine konsisten berada di bawah Linear. Energi laju belajar yang tinggi di pertengahan langkah memberikan daya dorong kinetik bagi *adapter* Cosine untuk melompat keluar dari jebakan nilai minimum lokal (*suboptimal local minima*).
* **Tikungan Akhir Linear (Steps 750–800):** Penyusutan tajam laju belajar Cosine mendekati nol (*Cosine Tail*) melumpuhkan pergerakan *optimizer* secara prematur di akhir langkah. Sebaliknya, penurunan Linear yang konstan memberikan sisa energi untuk melakukan satu kali pemolesan mikro (*fine-grained adjustment*) di ujung sesi, menghasilkan *Loss* validasi akhir yang unggul tipis (1.385).

---

## 🧪 3. HOW-TO GUIDES: AUDIT KUALITATIF PROOF OF CONCEPT (PoC)

Pengujian dilakukan secara *head-to-head* menggunakan teknik *Dynamic Adapter Switching* pada model inang yang sama untuk menjaga efisiensi VRAM tetap di bawah ~3.5 GB.

### Kasus Uji: Penanganan Krisis *Cognitive Overload* & *Executive Dysfunction*
> **Input Pengguna:** *"Aku mudah banget overwhelming dengan banyak hal dipikiran ku... Tugas kuliah, Tugas Pesantren, Belajar untuk Beasiswa, tugas freelance!"*

#### 🔴 Respons Otak Model 1 (Linear Scheduler):
> *"Terima kasih sudah mengungkapkan semua tekanan itu. Banyaknya tuntutan itu pasti terasa sangat menekan. Aku akan mendengarkan kamu dulu tanpa menghakimi."*
* **Karakteristik UX:** Bertindak sebagai bantalan emosi (*Emotional Cushion*) yang aman. Menggunakan pola kalimat universal psikologi suportif non-direktif.

#### 🟢 Respons Otak Model 2 (Cosine Scheduler) - [VERDIKT PILIHAN]:
> *"Aku mendengar seluruh tekanan itu. Sangat wajar jika otakmu merasa kewalahan karena semuanya datang bersamaan. Kamu butuh waktu untuk memprosesnya sendiri."*
* **Karakteristik UX:** Menunjukkan penyelarasan tingkat tinggi (*High-Fidelity Alignment*). Model mampu mengidentifikasi gejala klinis ADHD secara spesifik (stimulus datang bersamaan) dan memvalidasi kebutuhan biologis neurodivergen tanpa templat basa-basi.

---

## ⚙️ 4. REFERENCE: MANIFES DEPRESIASI & KONFIGURASI PIPELINE

Seluruh eksperimen dibangun di atas ekosistem Unsloth terbaru yang mengintegrasikan *Silent Monkey-Patching* secara otomatis ke dalam komponen `trl.SFTTrainer`.

### Hyperparameter Tetap (Kontrol Ketat):
```python
SFTConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Batch Efektif = 8
    learning_rate = 2e-4,
    weight_decay = 0.01,
    optim = "adamw_8bit",
    seed = 3407,
)
```

### Catatan Infrastruktur (Infrastructural Known Limitations):
Fungsi eksternal lama `unsloth_train(trainer)` telah didepresiasi karena logikanya sudah dilebur langsung ke dalam fungsi *native* `trainer.train()`. Proyek ini menggunakan pembajakan lokal `transformers.utils.hub.list_repo_templates` untuk mengabaikan bug hulu 404 pada server Hugging Face saat proses inisialisasi tokenisasi bahasa.

---

## 📜 5. LISENSI
Proyek riset ini dirilis di bawah lisensi **MIT**. Hak cipta terbuka untuk pengembangan teknologi kesehatan mental dan neurodivergensi dunia.
