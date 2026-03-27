#!/bin/bash
# TSQ 학습 데이터셋 다운로드
# 무료 + 등록 불필요한 것부터 자동 다운로드
#
# Usage:
#   bash scripts/download_datasets.sh           # 전부
#   bash scripts/download_datasets.sh mit        # MIT DriveDB만
#   bash scripts/download_datasets.sh wesad      # WESAD만
#   bash scripts/download_datasets.sh bdd        # BDD100K (수동 필요)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ═══ 1. MIT DriveDB (스트레스 + bio) ═══
download_mit() {
    log "=== MIT DriveDB (stress + ECG/EDA) ==="
    DST="$DATA_DIR/mit_drivedb"
    if [ -d "$DST" ] && [ "$(ls $DST/*.dat 2>/dev/null | wc -l)" -gt 5 ]; then
        log "Already exists, skipping"
        return
    fi
    mkdir -p "$DST"
    cd "$DST"

    # PhysioNet에서 wget으로 다운로드 (credentialed access 불필요한 부분)
    log "Downloading from PhysioNet..."
    for subj in drive01 drive02 drive03 drive04 drive05 drive06 drive07 drive08 drive09 drive10; do
        wget -q -nc "https://physionet.org/files/drivedb/1.0.0/${subj}.dat" 2>/dev/null || true
        wget -q -nc "https://physionet.org/files/drivedb/1.0.0/${subj}.hea" 2>/dev/null || true
    done
    log "MIT DriveDB: $(ls *.dat 2>/dev/null | wc -l) files"
    cd "$SCRIPT_DIR"
}

# ═══ 2. WESAD (스트레스 bio baseline) ═══
download_wesad() {
    log "=== WESAD (stress + PPG/EDA/ECG) ==="
    DST="$DATA_DIR/wesad"
    if [ -d "$DST" ] && [ "$(ls -d $DST/S* 2>/dev/null | wc -l)" -gt 5 ]; then
        log "Already exists, skipping"
        return
    fi
    mkdir -p "$DST"
    cd "$DST"

    log "Downloading from UCI ML Repository..."
    wget -q -nc "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download" -O wesad.zip 2>/dev/null || true
    if [ -f wesad.zip ]; then
        unzip -q -o wesad.zip 2>/dev/null || true
        rm -f wesad.zip
    fi
    log "WESAD: $(ls -d S* 2>/dev/null | wc -l) subjects"
    cd "$SCRIPT_DIR"
}

# ═══ 3. NTHU-DDD (졸음 얼굴 영상) ═══
download_nthu() {
    log "=== NTHU-DDD (drowsiness face video) ==="
    log "Manual download required:"
    log "  URL: http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/"
    log "  Save to: $DATA_DIR/nthu_ddd/"
}

# ═══ 4. BDD100K (장면 + 객체 + lane) ═══
download_bdd() {
    log "=== BDD100K (scenes + objects + lanes) ==="
    log "Manual download required (1.8TB full, or 100K images subset):"
    log "  URL: https://bdd-data.berkeley.edu/"
    log "  Register and download 'bdd100k_images_100k.zip' + 'bdd100k_labels_release.zip'"
    log "  Save to: $DATA_DIR/bdd100k/"
    log ""
    log "Quick subset (labels only, ~700MB):"
    DST="$DATA_DIR/bdd100k"
    mkdir -p "$DST"
    # Labels are small enough to download
    wget -q -nc "https://dl.cv.ethz.ch/bdd100k/bdd100k_labels_release.zip" -O "$DST/labels.zip" 2>/dev/null || true
    if [ -f "$DST/labels.zip" ]; then
        cd "$DST" && unzip -q -o labels.zip 2>/dev/null || true
        rm -f labels.zip
        cd "$SCRIPT_DIR"
        log "BDD100K labels downloaded"
    fi
}

# ═══ 5. ACDC (악천후 장면) ═══
download_acdc() {
    log "=== ACDC (adverse conditions: fog/night/rain/snow) ==="
    log "Manual download required:"
    log "  URL: https://acdc.vision.ee.ethz.ch/"
    log "  Register and download"
    log "  Save to: $DATA_DIR/acdc/"
}

# ═══ 6. DADA-2000 (사고 장면 + 시선) ═══
download_dada() {
    log "=== DADA-2000 (accident videos + gaze) ==="
    DST="$DATA_DIR/dada2000"
    mkdir -p "$DST"
    cd "$DST"
    log "Cloning annotation repo..."
    git clone --depth 1 https://github.com/JWFangit/LOTVS-DADA.git 2>/dev/null || true
    log "DADA-2000 annotations cloned. Videos need separate download."
    cd "$SCRIPT_DIR"
}

# ═══ 메인 ═══
case "${1:-all}" in
    mit)    download_mit ;;
    wesad)  download_wesad ;;
    nthu)   download_nthu ;;
    bdd)    download_bdd ;;
    acdc)   download_acdc ;;
    dada)   download_dada ;;
    all)
        download_mit
        download_wesad
        download_dada
        download_bdd
        download_nthu
        download_acdc
        ;;
    *)
        echo "Usage: $0 [all|mit|wesad|nthu|bdd|acdc|dada]"
        ;;
esac

log ""
log "=== Download Summary ==="
for d in "$DATA_DIR"/*/; do
    name=$(basename "$d")
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    log "  $name: $size"
done
