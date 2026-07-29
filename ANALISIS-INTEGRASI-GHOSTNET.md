# Analisis Integrasi GhostNet dan GhostNetV3 pada Replikasi S-YOLOv11

Tanggal analisis: 28 Juli 2026

## Ringkasan

Konfigurasi S-YOLOv11, S-YOLOv11 + GhostNet v1, dan S-YOLOv11 +
GhostNet-v3 yang tersedia di repository dapat dibangun dan menjalankan forward
pass. Bentuk feature map P2--P5 juga kompatibel dengan `Detect_ESDCDH`.

Walaupun kompatibel secara bentuk tensor, implementasi saat ini masih memiliki
beberapa persoalan arsitektural dan deployment. Persoalan yang paling awal
perlu diselesaikan adalah bahwa `C3k2_Ghost` dan `C3k2_GhostV3` tidak
mempertahankan topologi asli `C3k2` milik YOLO11.

## Hasil verifikasi awal

Pengujian dilakukan menggunakan Ultralytics lokal versi 8.3.0.

| Model | Parameter | GFLOPs | Forward pass | Output P2--P5 |
|---|---:|---:|---|---|
| S-YOLOv11 | 11.197.427 | 40,0 | Berhasil | Sesuai |
| S-YOLOv11 + GhostNet v1 | 9.577.035 | 34,7 | Berhasil | Sesuai |
| S-YOLOv11 + GhostNet-v3 saat ini | 9.588.915 | 34,7 | Berhasil | Sesuai |
| GhostNet-v3 setelah fusion | 9.576.243 | Sekitar setara v1 | Berhasil | Sesuai |

Dibandingkan S-YOLOv11, varian GhostNet v1 mengurangi parameter sekitar 14,5%
dan GFLOPs sekitar 13,3%.

Pengujian fusion `RepDWConv` menghasilkan selisih numerik maksimum sekitar
`1.43e-5` antara keluaran sebelum dan sesudah re-parameterization.

## 1. Masalah kesetaraan C3k2

### Implementasi C3k2 asli

`C3k2` YOLO11 merupakan turunan `C2f`. Alur dasarnya adalah:

1. Satu convolution menghasilkan `2 * hidden_channels`.
2. Tensor dibagi menjadi dua.
3. Cabang kedua diteruskan secara berantai melalui `n` bottleneck.
4. Dua tensor awal dan seluruh keluaran bottleneck digabungkan.
5. Convolution akhir menerima `(2 + n) * hidden_channels`.

Dengan demikian, setiap keluaran bottleneck dipertahankan sebagai bagian dari
feature aggregation.

### Implementasi C3k2_Ghost saat ini

Implementasi saat ini menggunakan pola:

```text
Conv(x) -> rangkaian GhostBottleneck --+
                                       +-> concat -> Conv
Conv(x) -------------------------------+
```

Convolution akhir selalu menerima `2 * hidden_channels`. Keluaran antara dari
setiap bottleneck tidak ikut digabungkan. Pola ini secara struktur lebih dekat
ke `C3Ghost` daripada `C3k2/C2f`.

### Dampak

- Model tetap valid secara bentuk tensor.
- Pengurangan parameter bukan hanya akibat penggantian convolution dengan
  Ghost operation, tetapi juga akibat perubahan topologi feature aggregation.
- Ablation terhadap `C3k2` asli menjadi kurang adil.
- Nama `C3k2_Ghost` dapat menimbulkan klaim implementasi yang tidak tepat.
- Perbedaan dapat memengaruhi retensi fitur objek kecil pada P2 dan P3.

### Rekomendasi

Buat implementasi Ghost yang mempertahankan semantik `C2f/C3k2`:

```text
Conv -> split(a, b)
              |
              b -> GhostBottleneck -> GhostBottleneck -> ...

concat(a, b, output_1, ..., output_n) -> Conv
```

Pilihan desain yang direkomendasikan:

- gunakan wrapper berbasis `C2f`;
- ganti hanya elemen di dalam `self.m` dengan `GhostBottleneck`;
- pertahankan `cv1`, `cv2`, proses split, dan ukuran input convolution akhir
  seperti `C3k2` asli;
- sediakan implementasi terpisah untuk Ghost v1 dan Ghost-v3/RepGhost;
- uji kesetaraan jumlah cabang, shape, gradient, dan parser YAML.

Topik ini menjadi prioritas implementasi pertama.

## 2. Status implementasi GhostNet-v3

Implementasi `C3k2_GhostV3` saat ini belum merepresentasikan GhostNetV3 penuh.
Komponen yang tersedia baru berupa cheap operation depthwise:

- depthwise 3x3;
- depthwise 1x1;
- identity BatchNorm;
- fusion menjadi satu depthwise 3x3.

Implementasi GhostNetV3 referensi juga mencakup:

- multi-branch re-parameterization pada primary convolution;
- multi-branch pada cheap operation;
- re-parameterization pada depthwise downsampling;
- long-kernel atau DFC-style gating pada bottleneck tertentu;
- perilaku yang bergantung pada stage.

Implementasi sekarang lebih tepat dinamai `RepGhost` atau `C3k2_RepGhost`,
kecuali komponen GhostNetV3 tersebut ikut diadaptasi.

## 3. Parameter mati pada RepDWConv

`rbr_reparam` dibuat sejak konstruktor, tetapi tidak dipakai selama training.
Pada model yang diuji terdapat:

- 24 blok `RepDWConv`;
- 7.920 parameter trainable tanpa gradient.

Parameter tersebut masuk optimizer meskipun tidak digunakan, dan dapat
bermasalah pada DDP ketika `find_unused_parameters=False`.

Solusi yang direkomendasikan adalah membuat `rbr_reparam` hanya ketika
`switch_to_deploy()` dipanggil.

## 4. Pipeline deployment

`switch_to_deploy()` belum terintegrasi dengan lifecycle model Ultralytics.
Belum ada fungsi tingkat model untuk:

1. memanggil fusion pada seluruh `RepDWConv`;
2. memverifikasi keluaran sebelum dan sesudah fusion;
3. menyimpan checkpoint deployment terpisah;
4. melakukan fusion sebelum ekspor ONNX atau TensorRT.

Checkpoint training dan deployment sebaiknya dipisahkan, misalnya:

```text
best_train.pt
best_deploy.pt
```

Fusion sebaiknya dilakukan pada salinan model supaya training masih dapat
dilanjutkan dari checkpoint aslinya.

## 5. Kompatibilitas pretrained weights

Susunan internal blok Ghost berbeda dari `C3k2`, sehingga pretrained weights
YOLO11 tidak dapat dipindahkan penuh. Layer yang tidak berubah masih dapat
menerima partial weight transfer, tetapi blok Ghost umumnya akan diinisialisasi
secara acak.

Checkpoint GhostNet klasifikasi juga tidak langsung kompatibel karena wrapper,
channel, dan susunan stage berbeda.

Perbandingan eksperimen harus menetapkan salah satu prosedur yang konsisten:

- semua model dilatih dari nol; atau
- semua model menggunakan partial transfer dengan aturan yang sama dan
  dilaporkan secara eksplisit.

## 6. Patch dan registry Ultralytics

Patch saat ini memiliki beberapa persoalan pemeliharaan:

- ditemukan import `modules_sy` ganda pada `tasks.py`;
- nama custom `GhostConv` dan `GhostBottleneck` dapat ditimpa oleh import modul
  bawaan Ultralytics;
- pemeriksaan idempotensi berbasis penggantian string gabungan tidak sepenuhnya
  aman untuk migrasi patch lama.

YAML yang tersedia masih dapat dibangun, tetapi registry sebaiknya dirapikan
sebelum eksperimen final.

## Strategi implementasi yang direkomendasikan

Urutan pekerjaan:

1. Perbaiki kesetaraan struktur `C3k2_Ghost` terhadap `C3k2/C2f`.
2. Buat ablation Ghost pada backbone saja.
3. Buat ablation Ghost pada backbone dan neck.
4. Pisahkan `RepGhost` dari implementasi GhostNetV3 yang lebih faithful.
5. Hilangkan parameter mati dan tambahkan pipeline deployment.
6. Validasi training, DDP, checkpoint, ONNX, dan TensorRT.

Metrik evaluasi yang diperlukan:

- mAP50;
- mAP50--95;
- AP small object;
- parameter dan GFLOPs;
- latency batch 1 setelah warm-up;
- throughput;
- peak VRAM selama training;
- ukuran model hasil ekspor;
- latency sebelum dan sesudah re-parameterization.

## Keputusan yang masih diperlukan

1. Apakah varian lanjutan akan menggunakan GhostNetV3 yang lebih faithful, atau
   mempertahankan desain ringan saat ini dengan nama `RepGhost`?
2. Apakah penggantian Ghost akan diuji pada backbone saja dan backbone + neck
   sebagai dua ablation terpisah?

Keputusan tersebut tidak harus diambil sebelum perbaikan masalah `C3k2`,
karena kesetaraan struktur `C3k2/C2f` merupakan fondasi bagi kedua opsi.

## Implementasi lanjutan: standalone GhostModuleV3

Pembaruan 29 Juli 2026:

- Ditambahkan `GhostModuleV3` sebagai blok feature extraction mandiri.
- Seluruh implementasi `C3k2` asli tidak dimodifikasi.
- Ditambahkan konfigurasi ablation `s-yolov11-ghostmodulev3.yaml`.
- Satu blok P4 backbone diganti dengan `GhostModuleV3`, mengikuti pola hybrid
  yang ditunjukkan diagram Ghost-YOLO.
- Downsampling tetap menggunakan `Conv`, bukan `DWConv`, agar pengaruh
  `GhostModuleV3` dapat diukur secara terisolasi.
- Tiga `C3k2` lain pada backbone dan delapan `C3k2` pada EMAFPN tetap asli.
- `rbr_reparam` tidak lagi dibuat selama training sehingga tidak ada parameter
  deploy yang masuk optimizer tanpa gradient.

Hasil verifikasi implementasi:

| Pemeriksaan | Hasil |
|---|---|
| Build melalui parser YAML | Berhasil |
| Jumlah standalone `GhostModuleV3` | 1 |
| Jumlah `C3k2` asli | 11 |
| Parameter | 11.024.883 |
| GFLOPs pada 640 | 39,3 |
| Forward skala P2--P5 | Sesuai |
| Backward dan gradient Ghost | Berhasil, tidak ada parameter tanpa gradient |
| Selisih maksimum setelah fusion | Sekitar `1.14e-5` |

Varian ini merupakan baseline **Ghost-only**. Integrasi `DWConv` sebaiknya
dibuat sebagai konfigurasi ablation berikutnya, bukan dicampurkan ke baseline
ini.

Audit terhadap source resmi Huawei kemudian menunjukkan bahwa versi awal
standalone module masih terlalu sederhana. Implementasi telah diperbarui agar
mengikuti grafik training GhostNetV3:

- tiga cabang Conv-BN pada primary operation;
- tiga cabang depthwise Conv-BN pada cheap operation;
- scale branch `1x1` dan identity-BN saat memenuhi syarat;
- aktivasi ReLU;
- DFC spatial gate;
- fusion primary dan cheap paths menjadi masing-masing satu convolution saat
  deployment.

Angka pada tabel di atas adalah angka setelah koreksi tersebut.

## Material referensi GhostNet tersedia

Pembaruan 29 Juli 2026:

Material primer GhostNet telah ditempatkan pada folder `ghostnet` dan akan
menjadi sumber validasi pada analisis berikutnya:

- `GhostNet-V1.pdf`;
- `GhostNet-V2.pdf`;
- `GhostNet-V3.pdf`;
- source GhostNet v1 pada `Efficient-AI-Backbones/ghostnet_pytorch`;
- source GhostNet v2 pada `Efficient-AI-Backbones/ghostnetv2_pytorch`;
- source GhostNet v3 pada `Efficient-AI-Backbones/ghostnetv3_pytorch`;
- metadata Git repository `Efficient-AI-Backbones`, sehingga asal commit dapat
  diperiksa.

Status pekerjaan saat catatan ini dibuat:

- material sudah terdeteksi dan tersimpan;
- source terverifikasi berasal dari
  `https://github.com/huawei-noah/Efficient-AI-Backbones.git`, commit
  `f90e129b645c3b1684fe07cd361cd557d0ad71f7`;
- audit ulang standalone `GhostModuleV3` terhadap source telah dilakukan;
- implementasi YOLOv11--GhostNet tambahan menunggu material integrasi yang
  sedang dicari;
- source Ghost-YOLO tidak ditemukan, sehingga integrasi tidak diklaim sebagai
  reproduksi identik paper tersebut.

### Batas validasi tanpa source Ghost-YOLO

Hal berikut dapat divalidasi:

- definisi operasi GhostNet v1;
- DFC attention GhostNet v2;
- cabang re-parameterization GhostNet v3;
- proses fusion train-to-deploy;
- posisi standalone module yang ditunjukkan diagram Ghost-YOLO;
- parameter, FLOPs, shape, gradient, dan equivalence fusion model lokal.

Hal berikut tidak dapat ditentukan secara pasti dari paper Ghost-YOLO:

- apakah kotak `GhostModule V3` berarti satu module atau satu bottleneck/stage;
- expansion ratio dan channel internal yang digunakan;
- apakah module mengganti atau ditambahkan setelah node sebelumnya;
- definisi persis `DWConv`;
- YAML, versi Ultralytics, pretrained weights, dan training recipe;
- penyebab persis kenaikan parameter `9.442.498 -> 10.410.434`.

Karena itu konfigurasi lokal dinamai sebagai ablation standalone
`GhostModuleV3`, bukan replika resmi Ghost-YOLO.
