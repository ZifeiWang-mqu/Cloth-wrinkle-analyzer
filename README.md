# Illustration Wrinkle Inconsistency Detector

イラスト中のキャラクター衣服に描かれた「非合理的な皺」を検知する作画支援アプリです。
画像をアップロードし、服領域を選択すると、人体構造・重力・張力・立体・線密度・陰影の
6 観点から皺の不整合を検査します。

> **このリポジトリの現状: バックエンド（FastAPI）+ フロントエンド（Next.js）+
> 写真参照特徴スクリプト（`ml/`）の MVP が一通り揃っています。**
> 高精度な異常検知（DINOv2 / Anomalib）と SAM による服領域自動抽出は次フェーズです。

---

## 何ができるか（バックエンド MVP）

- 画像アップロード（jpg / png / webp）と安全なファイル保存
- 服領域の手動 bbox 指定（無ければ画像全体）
- 古典的 CV パイプライン（前処理 → Canny → HoughLinesP → 構造特徴量）
- 6 要件のルールベース・スコアリング（各 0.0〜1.0）
- 日本語の説明文付き issue 生成・overall_score 算出
- SQLite への検査履歴保存とユーザーフィードバック保存
- MediaPipe Pose は任意（未インストールでも動作、`pose_detected=false`）
- 学習済みワンクラス異常検知（マハラノビス距離）を任意で利用、未学習時はダミーに自動フォールバック
  （`ml/train_anomaly_model.py` で学習。将来 DINOv2 / PatchCore へ差し替え可能）

---

## ディレクトリ構成

```
.
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI 本体 (/health, ルータ, CORS, 静的配信)
│   │   ├── settings.py        # 設定 (環境変数 WRINKLE_* で上書き可)
│   │   ├── schemas.py         # Pydantic スキーマ (API 契約)
│   │   ├── routes/
│   │   │   ├── inspect.py     # POST /api/inspect-wrinkle
│   │   │   └── feedback.py    # POST /api/feedback
│   │   ├── services/
│   │   │   ├── image_io.py    # アップロード検証・デコード・保存
│   │   │   ├── segmentation.py# 服領域決定 (将来 SAM)
│   │   │   ├── pose.py        # MediaPipe Pose (任意)
│   │   │   ├── wrinkle_edges.py     # 皺候補線の抽出
│   │   │   ├── feature_extraction.py# 構造特徴量
│   │   │   ├── rule_engine.py # 6 要件スコア + 統合
│   │   │   ├── explanation.py # 日本語説明文生成
│   │   │   └── anomaly_model.py     # ダミー異常検知 (差し替え可)
│   │   └── db/
│   │       ├── database.py    # SQLAlchemy エンジン/セッション
│   │       └── models.py      # Inspection / IssueFeedback
│   ├── tests/                 # pytest (rule engine + API)
│   ├── requirements.txt
│   └── pyproject.toml         # ruff + pytest 設定
├── frontend/                  # Next.js 14 (App Router) + TypeScript
│   ├── app/                   # layout.tsx / page.tsx / globals.css
│   ├── components/            # Uploader / RegionSelector / Heatmap / Result / Feedback
│   └── lib/                   # api.ts / types.ts
├── ml/
│   ├── extract_photo_features.py  # 写真→構造的参照特徴 (parquet + reference_stats)
│   ├── configs/default.yaml
│   └── requirements.txt
├── data/                      # 画像・DB・特徴量の保存先 (実行時に自動生成)
├── reports/
├── docker-compose.yml
├── README.md
└── CLAUDE.md
```

---

## セットアップと起動

### ローカル (Python 3.10+)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate    
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- ヘルスチェック: <http://localhost:8000/health>
- OpenAPI ドキュメント: <http://localhost:8000/docs>

> MediaPipe を使ったポーズ推定を有効にしたい場合のみ `pip install mediapipe`
> を追加してください。未インストールでもアプリは動作します。

### フロントエンド (Next.js)

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev            # http://localhost:3000  (バックエンドを 8000 で起動しておく)
```

### 写真参照特徴の抽出 + 学習モデル (ml)

```bash
cd ml
pip install -r requirements.txt          # backend の依存も入ります
# data/photos_raw/<garment>/ に写真を配置してから:
python extract_photo_features.py         # 1) 構造的特徴を抽出
#   -> data/features/photo_reference_features.parquet
#   -> data/features/reference_stats.json
python train_anomaly_model.py            # 2) ワンクラス異常検知を学習
#   -> data/models/anomaly_model.json    （バックエンドが起動時に自動読み込み）
```

> 学習モデルはマハラノビス距離ベースの簡易な one-class モデルで、numpy だけで
> 動作します。`anomaly_model.json` が無い場合はヒューリスティックに自動フォールバック
> します。データセット配置の規定と複数データセットの扱いは `ml/README.md` を参照。

### Docker

```bash
docker compose up --build
# backend  -> http://localhost:8000/health
# frontend -> http://localhost:3000
```

---

## API

### `POST /api/inspect-wrinkle`  (multipart/form-data)

| field           | type            | 必須 | 説明 |
|-----------------|-----------------|:---:|------|
| `image`         | file            | ✓   | jpg / png / webp |
| `garment_type`  | string          |     | shirt / skirt / pants / dress / jacket / unknown |
| `selected_region` | JSON string   |     | `{"x":100,"y":200,"w":300,"h":400}` |

レスポンス（抜粋）:

```json
{
  "inspection_id": "…",
  "result": "ok | needs_review",
  "overall_score": 0.0,
  "issues": [
    {
      "id": "gravity_inconsistency-0",
      "type": "gravity_inconsistency",
      "label": "重力と皺の矛盾",
      "severity": "low | medium | high",
      "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
      "confidence": 0.0,
      "message": "日本語の説明文"
    }
  ],
  "debug": {
    "pose_detected": false,
    "garment_region_used": true,
    "num_wrinkle_candidates": 0,
    "processing_time_ms": 0,
    "scores": { "gravity_inconsistency": 0.0, "...": 0.0 },
    "notes": []
  }
}
```

`curl` 例:

```bash
curl -F "image=@your_illustration.png" \
     -F "garment_type=skirt" \
     -F 'selected_region={"x":40,"y":60,"w":200,"h":300}' \
     http://localhost:8000/api/inspect-wrinkle
```

### `POST /api/feedback`  (application/json)

```json
{
  "inspection_id": "…",
  "issue_id": "gravity_inconsistency-0",
  "feedback": "correct | false_positive | missed_issue | wrong_location | wrong_reason",
  "corrected_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
  "corrected_type": "optional",
  "comment": "optional"
}
```

→ `{"status": "saved", "feedback_id": "…"}`

フィードバックは将来のイラスト専用データセット作成のため必ず保存されます。

---

## テスト

```bash
cd backend
pytest          # rule engine の単体テスト + API テスト
ruff check .    # Lint
```

---

## 設計メモ

- **構造的特徴のみを使用**: 写真とイラストのドメイン差が大きいため、色や質感ではなく
  線方向・密度・収束・明暗勾配などの構造的特徴を抽出します（要件 2, 15）。
- **落ちない設計**: 画像処理が失敗しても 200 を返し、`debug.notes` に理由を記録します。
- **差し替え前提**: `anomaly_model.py` はインターフェース中心。`segmentation.py` は
  SAM、`pose.py` は MediaPipe を後から差し込めます。
- **しきい値は `settings.py`** に集約。写真 100 枚から作る `reference_stats` で
  `score_line_density` を将来チューニングできます。

## ロードマップ（未実装）

- `reference_stats.json` を `score_line_density` に接続して密度判定を写真基準に
- DINOv2 / Anomalib（PatchCore）による異常検知の本実装（`anomaly_model.py` を差し替え）
- SAM による服領域自動抽出（`segmentation.py` の `segment_with_sam`）
- MediaPipe を使った関節ルールの高度化
- `ml/`: `build_reference_dataset.py` / `train_anomaly_model.py` / `evaluate.py`
