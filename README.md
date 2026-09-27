# ReviewLens

Fondasi analisis review marketplace. Tahap aktif saat ini adalah **Data Audit & Preparation**.
Seluruh project berada langsung di root workspace/repository `ImpactHack2026-AI-Training`.
Notebook 01 membaca lima dataset raw, memeriksa schema dan label, menormalisasi teks secara
konservatif, melakukan deduplikasi sesuai dataset, lalu menulis serta memvalidasi CSV.

## Current Progress

- [x] Project structure
- [x] Dataset audit
- [x] Dataset cleaning
- [ ] Sentiment model
- [ ] CG detector
- [ ] BGE-M3 similarity
- [ ] Marketplace integration
- [ ] OpenRouter explanation
- [ ] FastAPI integration

## Struktur project

```text
ImpactHack2026-AI-Training/
├── data/
│   ├── raw/
│   │   ├── simple.json
│   │   ├── challange.json
│   │   ├── fake_reviews.csv
│   │   ├── labeledReview.datasetFix.json
│   │   └── tokopedia-product-reviews-2019.csv
│   └── processed/
│       ├── simple_clean.csv
│       ├── challenge_clean.csv
│       ├── labeled_clean.csv
│       ├── fake_clean.csv
│       └── tokopedia_clean.csv
├── notebooks/
│   ├── 01_data_audit/01_data_audit.ipynb
│   ├── 02_sentiment/02_sentiment_model.ipynb       # placeholder kosong
│   ├── 03_cg_detector/03_cg_detector.ipynb        # placeholder kosong
│   └── 04_similarity/04_similarity_bge_m3.ipynb  # placeholder kosong
├── src/reviewlens/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── sentiment.py                             # placeholder
│   ├── cg_detector.py                           # placeholder
│   └── similarity.py                            # placeholder
├── tests/test_data_utils.py
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

`config.py` menyimpan portable paths, mapping kolom sumber, dan kunci deduplikasi.
`data_utils.py` menyediakan loading JSON/CSV, `normalize_text()`, `audit_dataframe()`,
distribusi label, analisis konflik, cleaning, penulisan CSV, dan validasi readback.
Notebook mengatur urutan pipeline serta menampilkan hasil. `tests/` memeriksa kasus yang
bisa menghilangkan informasi seperti category berbeda, label bertentangan, Unicode, dan ID.

Dataset raw dan processed tersedia secara lokal tetapi diabaikan oleh Git melalui `.gitignore`.
Pada clone baru, salin kelima file sumber ke `data/raw/` terlebih dahulu. Jangan mengedit raw.
Nama input `challange.json` mengikuti file sumber; nama output menggunakan `challenge_clean.csv`.

## Menjalankan notebook pertama

Gunakan Python 3.11 atau lebih baru. Dari root repository, jalankan di PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/01_data_audit/01_data_audit.ipynb
```

Pilih kernel dari `.venv`, lalu **Restart Kernel and Run All**. Path ditemukan otomatis dari
root proyek atau direktori notebook. Dalam VS Code, pilih interpreter `.venv/Scripts/python.exe`.
Tidak perlu mengaktifkan virtual environment atau mengubah execution policy PowerShell.

Untuk menjalankan seluruh notebook tanpa browser dan menyimpan output pada notebook yang sama:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks/01_data_audit/01_data_audit.ipynb
```

Jalankan regression checks dengan:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Kebijakan preparation

| Dataset | Teks sumber | Kunci deduplikasi |
| --- | --- | --- |
| simple | `comment` | `clean_text + sentiment` |
| challenge | `comment` | `clean_text + sentiment + category` |
| labeled | `review` | `clean_text + sentimen` |
| fake | `text_` | `clean_text + label` |
| tokopedia | `text` | Seluruh baris dipertahankan |

Kunci challenge mencakup category sesuai keputusan pengguna setelah audit menemukan variasi
category pada text/sentiment yang sama. Nilai category tidak diubah. Untuk kunci berulang,
simpan baris pertama dalam urutan sumber; rating dan metadata non-kunci mengikuti baris itu.
Tidak ada penyaringan berdasarkan rating. Baris dengan label bertentangan tetap tersedia.
Baris teks kosong, jika ditemukan pada input berikutnya, dipertahankan untuk pemeriksaan.

Normalisasi hanya menangani null, konversi string, HTML entity, lowercase, dan whitespace.
Emoji, punctuation, slang, dan negation dipertahankan. URL replacement tersedia sebagai opsi
`normalize_text(text, replace_urls=True)` tetapi nonaktif dalam pipeline. Tidak ada stemming,
stopword removal, atau translation. `labeled.review` tetap Bahasa Indonesia dan `translate`
hanya metadata. Label binary 0/1 tidak dipetakan ke kelas sentimen tanpa konfirmasi sumber.
`CG` berarti computer-generated dan `OR` berarti original review; keduanya tidak diganti
menjadi fake/genuine.

CSV ditulis dengan UTF-8 BOM (`utf-8-sig`) dan tanpa index. ID produk/toko serta nilai `sold`
dibaca sebagai string. Hanya field CSV kosong dianggap missing; literal `NA` tetap teks.
CSV tidak menyimpan dtype atau membedakan null dari string kosong. Gunakan `load_dataset()`
untuk membaca ulang dengan kebijakan yang sama; schema/missing asli tersedia di laporan audit.

## Output dan validasi

Kelima CSV di `data/processed/` dibaca ulang dan dibandingkan dengan dataframe hasil cleaning:
jumlah baris, urutan kolom, keberadaan `clean_text`, dataset nonempty, dan seluruh nilai
serialized. Hash SHA-256 raw dibandingkan sebelum dan sesudah eksekusi. Menjalankan ulang
notebook memperbarui output dengan kebijakan yang sama.

Laporan di `outputs/reports/`:

- `audit_summary.csv`, `schema_audit.csv`: dimensi, dtype, missing values, exact duplicates.
- `duplicate_summary.csv`: duplicate normalized text dan jumlah text kosong.
- `label_distribution.csv`: distribusi sebelum/sesudah cleaning.
- `cleaning_summary.csv`: jumlah sebelum/sesudah, removal, dan persentasenya.
- `*_label_conflicts.csv`: baris asli yang memiliki text sama dengan label berbeda.
- `challenge_category_variants.csv`: kategori berbeda per text/sentiment.
- `simple_challenge_overlap.csv`: normalized text yang muncul pada kedua dataset.
- `raw_integrity.csv`: hash dan hasil pemeriksaan raw.
- `environment.json`: versi Python, pandas, dan NumPy saat eksekusi.

## Hasil eksekusi pertama (27 September 2026)

Notebook selesai dengan 13 code cell dieksekusi berurutan tanpa error (27 cell termasuk
markdown). Kelima processed CSV lolos validasi seluruh nilai, hash raw tetap sama dengan
file sumber yang disalin, dan 9 regression check lulus. Pemeriksaan dependency `pip check`
juga lulus. Environment: Python 3.14.5, pandas 3.0.6, NumPy 2.5.3.

| Dataset | Rows before | Rows after | Removed | Removal percent |
| --- | ---: | ---: | ---: | ---: |
| simple | 17000 | 2193 | 14807 | 87.10% |
| challenge | 4840 | 4795 | 45 | 0.93% |
| labeled | 11606 | 11589 | 17 | 0.15% |
| fake | 40432 | 40405 | 27 | 0.07% |
| tokopedia | 40607 | 40607 | 0 | 0.00% |

Angka di atas adalah snapshot eksekusi pertama. `cleaning_summary.csv` selalu dihitung
ulang dari dataframe pada setiap run; tidak ada row count yang di-hardcode dalam pipeline.

Temuan yang perlu diperhatikan sebelum tahap model:

- `simple` memiliki 14040 full-row exact duplicates. Deduplikasi text/sentiment juga
  menggabungkan baris dengan rating berbeda, sehingga total removal menjadi 14807.
  Proporsi label setelah cleaning berubah: positive 1034, negative 1004, neutral 155.
  Data tidak di-rebalance pada tahap ini.
- `challenge` memiliki 44 kelompok text/sentiment dengan category berbeda. Semuanya
  tetap terwakili dengan kunci tiga kolom. Label `oke banget` bertentangan (positive/neutral).
- `labeled` memiliki satu teks berlabel 0 dan 1: `barang bagus pengiriman agak lama`.
  Kedua observasi dipertahankan. Tidak ada pemetaan binary label berdasarkan asumsi.
- Ada 11 normalized text yang sama pada `simple` dan `challenge`. Periksa overlap ini
  saat merancang split agar hard evaluation set tidak bocor ke training.
- Tokopedia memiliki 3741 kemunculan tambahan normalized text yang sama secara global.
  Seluruhnya tetap ada. `sold` berisi 14 nilai kosong dan format lokal seperti `3,2rb`,
  sehingga tidak dikonversi otomatis menjadi angka.
- Tidak ditemukan `clean_text` kosong pada kelima dataset. Schema sumber sesuai kolom
  yang diberikan; challenge memiliki 12 category aktual. Semua nilai CG/OR dipertahankan.

Sentiment, CG detector, similarity/clustering, ingestion, OpenRouter, dan FastAPI belum
diimplementasikan. Tahap kedua menunggu instruksi lanjutan.
