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

## 6. Persiapan VPS Ubuntu

Bagian ini mengikuti tata letak dan virtual environment pada
`TRAINING-UBUNTU.md`. Contoh mengasumsikan repository berada di:

```text
~/yolo/s-yolov11-replication
```

Jika repository Anda berada di lokasi lain, misalnya
`/root/Object-Detection/s-yolov11-replication`, ganti path pada perintah dengan
path tersebut. Jangan mencampurkan dua tata letak dalam satu sesi.

### 6.1 Ambil branch penelitian

```bash
mkdir -p ~/yolo
cd ~/yolo
git clone https://github.com/ibanana0/s-yolov11-replication.git
cd s-yolov11-replication
git fetch origin
git switch feat/ghostnetv3-dwconv-ablation
git pull --ff-only
```

Jika repository sudah tersedia:

```bash
cd ~/yolo/s-yolov11-replication
git fetch origin
git switch feat/ghostnetv3-dwconv-ablation
git pull --ff-only
```

Catat commit yang digunakan:

```bash
git rev-parse HEAD
```

### 6.2 Siapkan Python dan Ultralytics

Ultralytics harus terpasang pada interpreter Python yang sama dengan yang
digunakan untuk menjalankan patch dan training. Tata letak yang direkomendasikan:

```text
~/yolo/
├── .venv/
├── ultralytics/
└── s-yolov11-replication/
```

Instalasi dari awal:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv git tmux

cd ~/yolo
python3.11 -m venv .venv
source ~/yolo/.venv/bin/activate
python -m pip install --upgrade pip

git clone --depth 1 --branch v8.3.0 \
  https://github.com/ultralytics/ultralytics.git

python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ~/yolo/ultralytics
```

Gunakan `cu121` sebagai pengganti `cu118` bila environment CUDA Anda memang
memerlukannya. Verifikasi environment:

```bash
which python
python --version
python -m pip --version
python -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Versi Ultralytics harus `8.3.0` dan `CUDA` harus bernilai `True`. Error
`ModuleNotFoundError: No module named 'ultralytics'` berarti virtual environment
belum aktif atau Ultralytics dipasang pada interpreter yang berbeda. Perbaiki
dengan:

```bash
source ~/yolo/.venv/bin/activate
python -m pip install -e ~/yolo/ultralytics
```

Gunakan selalu `python -m pip`, bukan `pip` dari environment yang belum
dipastikan.

### 6.3 Patch dan pemeriksaan awal

Semua perintah berikut dijalankan dari root repository replikasi, sehingga
tidak memakai awalan `replikasi/`:

```bash
cd ~/yolo/s-yolov11-replication
source ~/yolo/.venv/bin/activate

python apply_patch.py
python proof_static.py
python proof_ghostmodulev3.py
python proof_ablation_models.py
```

Hasil yang diharapkan dari pemeriksaan ablation:

```text
baseline         params=11,197,427 GFLOPs=40.0083
ghostv3          params=11,024,883 GFLOPs=39.2903
dwconv           params=9,270,707  GFLOPs=34.4013
ghostv3_dwconv   params=9,098,163  GFLOPs=33.6833
```

Jangan memulai training penuh jika salah satu pemeriksaan gagal.

## 7. Menjalankan training dengan tmux

`tmux` menjaga proses tetap berjalan di VPS ketika koneksi SSH terputus atau
window terminal ditutup. Menutup window SSH aman setelah sesi tmux dilepas
(*detach*), selama VPS tidak dimatikan atau direstart.

### 7.1 Buat sesi pilot

Pasang `tmux` jika belum tersedia:

```bash
sudo apt update
sudo apt install -y tmux
```

Buat sesi bernama `ghost-pilot`:

```bash
tmux new -s ghost-pilot
```

Setelah masuk ke tampilan tmux, jalankan:

```bash
source ~/yolo/.venv/bin/activate
cd ~/yolo/s-yolov11-replication
mkdir -p logs

python -u train_ablation.py \
  --variant all \
  --epochs 5 \
  --batch 8 \
  --device 0 \
  --project runs/s-yolov11-ablation \
  2>&1 | tee logs/pilot-all.log
```

Opsi `-u` membuat output Python langsung ditulis ke log. `2>&1 | tee` menyimpan
stdout dan error sekaligus sambil tetap menampilkannya.

### 7.2 Detach dan tutup SSH

Untuk meninggalkan tmux tanpa mematikan training:

1. Tekan `Ctrl+b`.
2. Lepaskan kedua tombol.
3. Tekan `d`.

Setelah muncul pesan `[detached ...]`, window SSH boleh ditutup. Jangan menekan
`Ctrl+C`, karena itu menghentikan training.

### 7.3 Sambungkan kembali

Setelah login ulang ke VPS:

```bash
tmux ls
tmux attach -t ghost-pilot
```

Jika tmux menampilkan sesi masih attached di terminal lain:

```bash
tmux attach -d -t ghost-pilot
```

Perintah tersebut melepaskan client lama lalu menyambungkan terminal baru.

### 7.4 Memantau tanpa attach

Dari sesi SSH lain:

```bash
nvidia-smi
tail -f ~/yolo/s-yolov11-replication/logs/pilot-all.log
```

Tekan `Ctrl+C` untuk keluar dari `tail -f`; ini hanya menghentikan pemantauan,
bukan training di tmux.

Hasil eksperimen tersimpan di:

```text
~/yolo/s-yolov11-replication/runs/s-yolov11-ablation/
```

Periksa folder run dan checkpoint:

```bash
find ~/yolo/s-yolov11-replication/runs/s-yolov11-ablation \
  -maxdepth 4 -type f \
  \( -name 'results.csv' -o -name 'best.pt' -o -name 'last.pt' \)
```

### 7.5 Scroll dan menyalin output tmux

Masuk mode scroll dengan `Ctrl+b`, lalu tekan `[`. Gunakan tombol panah,
`PageUp`, atau `PageDown`. Tekan `q` untuk keluar dari mode scroll.

### 7.6 Menghentikan training

Hentikan secara normal:

```bash
tmux attach -t ghost-pilot
```

Kemudian tekan `Ctrl+C` satu kali dan tunggu proses kembali ke shell. Keluar
dari shell dengan:

```bash
exit
```

Jika benar-benar perlu mematikan seluruh sesi:

```bash
tmux kill-session -t ghost-pilot
```

`kill-session` menghentikan semua proses dalam sesi dan sebaiknya hanya dipakai
setelah memastikan target sesi benar.

### 7.7 Training utama di tmux

Satu GPU sebaiknya menjalankan satu training pada satu waktu. Jangan membuat
empat sesi training paralel pada GPU yang sama. Gunakan queue runner agar setiap
model mempunyai log terpisah dan model berikutnya dimulai otomatis hanya jika
model sebelumnya selesai dengan sukses.

```bash
tmux new -s ablation-s0
```

Di dalam tmux:

```bash
source ~/yolo/.venv/bin/activate
cd ~/yolo/s-yolov11-replication
mkdir -p logs

python -u run_ablation_queue.py \
  --variants baseline ghostv3 dwconv ghostv3_dwconv \
  --epochs 300 \
  --seed 0 \
  --batch 8 \
  --device 0 \
  --project runs/s-yolov11-ablation \
  --log-root logs/ablation
```

Detach dengan `Ctrl+b`, lalu `d`. Untuk melihat kembali:

```bash
tmux attach -t ablation-s0
```

Log disimpan terpisah:

```text
logs/ablation/seed0/queue.log
logs/ablation/seed0/baseline.log
logs/ablation/seed0/ghostv3.log
logs/ablation/seed0/dwconv.log
logs/ablation/seed0/ghostv3_dwconv.log
```

`queue.log` mencatat waktu mulai, selesai, gagal, dan pergantian model. Jika
satu training keluar dengan status error, queue berhenti dan tidak menjalankan
model berikutnya. Pantau status tanpa masuk tmux:

```bash
tail -f logs/ablation/seed0/queue.log
```

Pantau output model yang sedang berjalan, misalnya:

```bash
tail -f logs/ablation/seed0/ghostv3.log
```

Jika GPU mempunyai VRAM kurang dari 24 GB, mulai dengan `--batch 4` atau
`--batch 2`. Gunakan pilot run untuk menentukan batch sebelum eksperimen utama.

### 7.8 Jika training sudah terlanjur berjalan di luar tmux

Proses yang sudah berjalan langsung di terminal SSH tidak otomatis berpindah
ke tmux. Pilihan paling aman:

1. Tunggu sampai setidaknya satu epoch selesai dan `last.pt` terbentuk.
2. Hentikan training dengan `Ctrl+C`.
3. Cari checkpoint terakhir:

   ```bash
   cd ~/yolo/s-yolov11-replication
   find runs -type f -path '*/weights/last.pt' -printf '%T@ %p\n' \
     | sort -nr | head
   ```

4. Buat sesi tmux:

   ```bash
   tmux new -s yolo-resume
   ```

5. Di dalam tmux, aktifkan environment dan lanjutkan checkpoint. Ganti path
   `last.pt` dengan hasil langkah sebelumnya:

   ```bash
   source ~/yolo/.venv/bin/activate
   cd ~/yolo/s-yolov11-replication
   mkdir -p logs

   python -u -c "from ultralytics import YOLO; YOLO('runs/s-yolov11-ablation/NAMA-RUN/weights/last.pt').train(resume=True)" \
     2>&1 | tee logs/resume.log
   ```

6. Detach dengan `Ctrl+b`, lalu `d`.

Jika `last.pt` belum ada, penghentian berarti run harus dimulai lagi. Jangan
menutup window SSH yang menjalankan training langsung sebelum proses dipindah
melalui prosedur stop-and-resume tersebut.

## 8. Tahap penelitian

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

Jalankan dari root repository:

```bash
source ~/yolo/.venv/bin/activate
cd ~/yolo/s-yolov11-replication
python apply_patch.py
python proof_static.py
python proof_ghostmodulev3.py
python proof_ablation_models.py
```

Semua pemeriksaan harus lulus sebelum training.

### Tahap 1 — pilot run

Jalankan 1--5 epoch untuk seluruh varian:

```bash
python train_ablation.py --variant all --epochs 5 --batch 8 --device 0
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
python train_ablation.py --variant baseline --seed 0 --device 0
python train_ablation.py --variant ghostv3 --seed 0 --device 0
python train_ablation.py --variant dwconv --seed 0 --device 0
python train_ablation.py --variant ghostv3_dwconv --seed 0 --device 0
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

## 9. Analisis statistik

Untuk setiap metrik:

1. Simpan nilai seluruh seed.
2. Hitung mean dan standard deviation.
3. Laporkan selisih absolut dan relatif terhadap baseline.
4. Gunakan interval kepercayaan atau uji statistik yang sesuai bila jumlah run
   mencukupi.
5. Hindari memilih seed terbaik sebagai satu-satunya hasil.

Kenaikan kecil yang berada di dalam variasi antarseed tidak boleh dinyatakan
sebagai peningkatan arsitektural.

## 10. Kriteria keputusan

Contoh aturan keputusan yang ditetapkan sebelum melihat hasil:

- Ghost diterima bila AP small atau mAP50--95 meningkat secara konsisten tanpa
  kenaikan latency yang tidak dapat diterima.
- DWConv diterima bila penghematan latency/parameter sebanding dengan penurunan
  akurasi.
- Kombinasi dipilih bila memiliki Pareto trade-off terbaik.

Nilai ambang konkret harus disesuaikan dengan target deployment penelitian.

## 11. Artefak yang harus disimpan

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

## 12. Risiko dan keterbatasan

- Posisi GhostModuleV3 merupakan interpretasi diagram Ghost-YOLO.
- Source paper Ghost-YOLO tidak tersedia.
- Parameter paper menunjukkan kenaikan yang tidak dapat dijelaskan persis dari
  diagram.
- Recipe KD GhostNetV3 klasifikasi belum dipindahkan ke deteksi.
- Hasil TT100K tidak dapat langsung digeneralisasi ke VisDrone.
- Pengurangan GFLOPs tidak menjamin penurunan latency aktual.
- Re-parameterization mengubah struktur state dictionary.

Seluruh keterbatasan ini harus ditulis pada laporan akhir.
