# Perubahan S-YOLOv11 dan Rencana Penelitian GhostNetV3

Dokumen ini menjelaskan perubahan kode, batas klaim, rancangan ablation, dan
langkah penelitian lanjutan untuk integrasi GhostNetV3 dan DWConv pada
S-YOLOv11.

## 1. Tujuan

Penelitian menguji dua hipotesis yang dipisahkan:

1. `GhostModuleV3` meningkatkan atau mempertahankan kualitas representasi fitur.
2. `DWConv` mengurangi parameter dan komputasi downsampling backbone.

Kombinasi keduanya diharapkan memberi kompromi akurasi dan efisiensi. Hipotesis
akurasi belum dianggap terbukti sebelum eksperimen VisDrone selesai.

## 2. Batas klaim

Source Ghost-YOLO tidak tersedia. Implementasi ini menggunakan:

- diagram arsitektur dan tabel pada paper Ghost-YOLO;
- PDF GhostNet v1, v2, dan v3;
- source resmi Huawei `Efficient-AI-Backbones` commit
  `f90e129b645c3b1684fe07cd361cd557d0ad71f7`;
- source Ultralytics 8.3.0 pada repository.

Nama yang tepat adalah:

> Adaptasi standalone GhostModuleV3 pada S-YOLOv11 berdasarkan source resmi
> Huawei dan interpretasi diagram Ghost-YOLO.

Implementasi tidak diklaim sebagai reproduksi identik Ghost-YOLO.

## 3. Perubahan terhadap S-YOLOv11

### 3.1 Baseline

Baseline `s-yolov11.yaml` tidak diubah:

- backbone YOLOv11;
- EMAFPN empat level;
- WFF;
- EUCB;
- `Detect_ESDCDH`;
- NMIoU.

### 3.2 Standalone GhostModuleV3

Pada varian Ghost-only, node P4 backbone nomor 6 berubah:

```text
Sebelum: C3k2(256 -> 256)
Sesudah: GhostModuleV3(256 -> 256)
```

`C3k2` lain tetap asli. Internal `C3k2` tidak dimodifikasi.

`GhostModuleV3` memiliki:

- primary operation dengan tiga cabang Conv-BN;
- cheap operation dengan tiga cabang depthwise Conv-BN;
- scale branch `1x1`;
- identity-BN jika dimensi memenuhi syarat;
- ReLU;
- DFC gate berupa `1x1`, depthwise `1x5`, dan depthwise `5x1`;
- fungsi train-to-deploy fusion.

### 3.3 DWConv

Pada varian DWConv, lima downsampling backbone di node `0, 1, 3, 5, 7`
menggunakan `DWConv` bawaan Ultralytics. Neck tidak diubah.

Penggantian tersebut menurunkan tepat `1.926.720` parameter. Nilai ini sama
dengan selisih parameter varian Base dan DWConv pada tabel Ghost-YOLO:

```text
9.442.498 - 7.515.778 = 1.926.720
```

Kesamaan ini merupakan bukti kuat bahwa interpretasi lokasi dan implementasi
DWConv sesuai dengan konfigurasi yang dilaporkan paper, walaupun YAML aslinya
tidak tersedia.

## 4. Matriks ablation

| ID | Konfigurasi | GhostModuleV3 | DWConv backbone |
|---|---|---:|---:|
| A | `s-yolov11.yaml` | Tidak | Tidak |
| B | `s-yolov11-ghostmodulev3.yaml` | Ya | Tidak |
| C | `s-yolov11-dwconv.yaml` | Tidak | Ya |
| D | `s-yolov11-ghostmodulev3-dwconv.yaml` | Ya | Ya |

Hasil statis pada input 640:

| Varian | Parameter training | GFLOPs |
|---|---:|---:|
| A — baseline | 11.197.427 | 40,0083 |
| B — GhostModuleV3 | 11.024.883 | 39,2903 |
| C — DWConv | 9.270.707 | 34,4013 |
| D — GhostModuleV3 + DWConv | 9.098.163 | 33,6833 |

Setelah fusion, parameter varian B turun menjadi 10.955.123 karena cabang
training GhostNetV3 dilebur menjadi dua convolution deployment.

## 5. Kontrol eksperimen

Agar perbandingan sah, seluruh varian harus menggunakan:

- split dataset yang sama;
- seed yang sama;
- ukuran citra 640;
- batch size yang sama;
- jumlah epoch yang sama;
- optimizer dan scheduler yang sama;
- augmentasi yang sama;
- patience dan `close_mosaic` yang sama;
- hardware dan versi dependency yang sama;
- metode pemilihan checkpoint yang sama.

Jangan mengubah hyperparameter hanya pada satu varian. Jika tuning diperlukan,
lakukan tuning terpisah dan laporkan sebagai eksperimen kedua.

## 6. Tahap penelitian

### Tahap 0 — verifikasi lingkungan

Catat:

```text
Python
PyTorch
CUDA
cuDNN
Ultralytics
GPU
commit repository
```

Jalankan:

```bash
python replikasi/apply_patch.py
python replikasi/proof_static.py
python replikasi/proof_ghostmodulev3.py
python replikasi/proof_ablation_models.py
```

Semua pemeriksaan harus lulus sebelum training.

### Tahap 1 — pilot run

Jalankan 1--5 epoch untuk seluruh varian:

```bash
python replikasi/train_ablation.py --variant all --epochs 5 --batch 8 --device 0
```

Tujuannya bukan membandingkan mAP, melainkan memastikan:

- loss finite;
- tidak ada out-of-memory;
- validation berjalan;
- checkpoint dapat dimuat;
- waktu per epoch tercatat.

### Tahap 2 — eksperimen utama

Jalankan setiap varian dengan minimal tiga seed:

```bash
python replikasi/train_ablation.py --variant baseline --seed 0 --device 0
python replikasi/train_ablation.py --variant ghostv3 --seed 0 --device 0
python replikasi/train_ablation.py --variant dwconv --seed 0 --device 0
python replikasi/train_ablation.py --variant ghostv3_dwconv --seed 0 --device 0
```

Ulangi dengan seed, misalnya, `1` dan `2`. Urutan eksekusi sebaiknya dirotasi
untuk mengurangi bias akibat kondisi termal GPU.

### Tahap 3 — evaluasi akurasi

Laporkan per seed dan agregat mean ± standard deviation:

- Precision;
- Recall;
- mAP50;
- mAP50--95;
- AP small, medium, dan large jika evaluator mendukung;
- per-class AP;
- confusion matrix;
- false positive dan false negative pada objek kecil.

Untuk VisDrone, perhatian utama adalah objek tiny/small pada head P2 dan P3.

### Tahap 4 — evaluasi efisiensi

Ukur model training dan deployment secara terpisah:

- parameter;
- GFLOPs;
- ukuran checkpoint;
- peak VRAM training;
- waktu training per epoch;
- latency batch 1;
- throughput pada batch yang relevan;
- warm-up sebelum pengukuran;
- mean, median, P95 latency;
- konsumsi daya jika alat tersedia.

Jangan menyimpulkan kecepatan hanya dari GFLOPs. Depthwise/grouped convolution
dapat memiliki efisiensi hardware yang berbeda pada GPU, CPU, TensorRT, dan
perangkat edge.

### Tahap 5 — re-parameterization

Untuk varian Ghost:

1. Muat checkpoint terbaik.
2. Set model ke `eval()`.
3. Buat salinan model.
4. Panggil `switch_to_deploy()` pada seluruh `GhostModuleV3`.
5. Bandingkan output sebelum dan sesudah fusion.
6. Tolak hasil jika selisih melebihi toleransi yang ditetapkan.
7. Simpan checkpoint training dan deployment secara terpisah.

Jangan melakukan fusion pada checkpoint satu-satunya karena struktur cabang
training akan dihapus.

### Tahap 6 — ekspor

Uji minimal:

- PyTorch;
- ONNX;
- TensorRT atau OpenVINO sesuai target;
- FP32;
- FP16 jika didukung.

Validasi kesetaraan prediction setelah ekspor, bukan hanya keberhasilan proses
export.

### Tahap 7 — eksperimen tambahan

Setelah matriks utama selesai, eksperimen lanjutan dapat mencakup:

- posisi GhostModuleV3 di P2, P3, P4, atau P5;
- lebih dari satu GhostModuleV3;
- Ghost bottleneck/stage dengan expansion ratio resmi;
- knowledge distillation;
- pretrained GhostNetV3;
- rasio Ghost selain 2;
- jumlah re-parameterization branches;
- DFC gate aktif versus nonaktif;
- DWConv backbone versus DWConv neck.

Eksperimen tersebut tidak boleh dicampurkan ke matriks utama karena akan
mengubah lebih dari satu variabel sekaligus.

## 7. Analisis statistik

Untuk setiap metrik:

1. Simpan nilai seluruh seed.
2. Hitung mean dan standard deviation.
3. Laporkan selisih absolut dan relatif terhadap baseline.
4. Gunakan interval kepercayaan atau uji statistik yang sesuai bila jumlah run
   mencukupi.
5. Hindari memilih seed terbaik sebagai satu-satunya hasil.

Kenaikan kecil yang berada di dalam variasi antarseed tidak boleh dinyatakan
sebagai peningkatan arsitektural.

## 8. Kriteria keputusan

Contoh aturan keputusan yang ditetapkan sebelum melihat hasil:

- Ghost diterima bila AP small atau mAP50--95 meningkat secara konsisten tanpa
  kenaikan latency yang tidak dapat diterima.
- DWConv diterima bila penghematan latency/parameter sebanding dengan penurunan
  akurasi.
- Kombinasi dipilih bila memiliki Pareto trade-off terbaik.

Nilai ambang konkret harus disesuaikan dengan target deployment penelitian.

## 9. Artefak yang harus disimpan

Untuk setiap run simpan:

```text
model YAML
training arguments
seed
dependency versions
commit hash
best.pt dan last.pt
results.csv
curves dan confusion matrix
validation output
latency raw samples
exported model
log fusion equivalence
```

Penamaan run pada `train_ablation.py` berbentuk `<variant>-seed<seed>` agar
hasil tidak tercampur.

## 10. Risiko dan keterbatasan

- Posisi GhostModuleV3 merupakan interpretasi diagram Ghost-YOLO.
- Source paper Ghost-YOLO tidak tersedia.
- Parameter paper menunjukkan kenaikan yang tidak dapat dijelaskan persis dari
  diagram.
- Recipe KD GhostNetV3 klasifikasi belum dipindahkan ke deteksi.
- Hasil TT100K tidak dapat langsung digeneralisasi ke VisDrone.
- Pengurangan GFLOPs tidak menjamin penurunan latency aktual.
- Re-parameterization mengubah struktur state dictionary.

Seluruh keterbatasan ini harus ditulis pada laporan akhir.
