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
- SQLite への検査履歴保存と**強化されたユーザーフィードバック**（image_id / issue_type /
  original・corrected bbox / confidence / severity / source などを保存。`missed_issue` も追加可能）
- MediaPipe Pose は任意（未インストールでも動作、`pose_detected=false`）
- 学習済みワンクラス異常検知（マハラノビス距離）を任意で利用、未学習時はダミーに自動フォールバック
- **issue_type / garment_type 別の判定閾値**（`backend/app/config/thresholds.json`）+ フロント表示閾値スライダー
- **作画添削UI**: 検出枠の色分け（issue_type 別）・クリック選択・種類/重大度フィルタ・並び替え・見逃し手動追加
- **外部ツール向けAPI**: `POST /api/inspect-base64`・`GET /api/model/status`・`GET /api/inspection/{id}`
- **評価スクリプト** `ml/evaluate_illustration_results.py`（IoU で P/R/F1 を算出 → `reports/`）

---

## ディレクトリ構成

```
.
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI 本体 (/health, ルータ, CORS, 静的配信)
│   │   ├── settings.py        # 設定 (環境変数 WRINKLE_* で上書き可)
│   │   ├── schemas.py         # Pydantic スキーマ (API 契約)
│   │   ├── config/thresholds.json   # issue_type/garment_type 別 判定閾値
│   │   ├── routes/
│   │   │   ├── inspect.py     # POST /api/inspect-wrinkle, /api/inspect-base64
│   │   │   ├── feedback.py    # POST /api/feedback (強化版)
│   │   │   └── status.py      # GET /api/model/status, /api/inspection/{id}
│   │   ├── services/
│   │   │   ├── image_io.py        # アップロード/Base64 検証・デコード・保存
│   │   │   ├── inspection_service.py # 検査パイプライン共通ロジック
│   │   │   ├── segmentation.py    # 服領域決定 (将来 SAM)
│   │   │   ├── pose.py            # MediaPipe Pose (任意)
│   │   │   ├── wrinkle_edges.py   # 皺候補線の抽出
│   │   │   ├── feature_extraction.py # 構造特徴量
│   │   │   ├── rule_engine.py     # 6 要件スコア + 統合 + 閾値適用
│   │   │   ├── thresholds.py      # thresholds.json ローダ
│   │   │   ├── reference_stats.py # reference_stats.json ローダ (密度判定)
│   │   │   ├── explanation.py     # 日本語説明文生成
│   │   │   └── anomaly_model.py   # 異常検知 (Mahalanobis / dummy)
│   │   └── db/                # database.py / models.py (Inspection / IssueFeedback)
│   ├── tests/                 # pytest (rule engine + API + thresholds + anomaly)
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/                  # Next.js 14 (App Router) + TypeScript
│   ├── app/                   # layout.tsx / page.tsx / globals.css
│   ├── components/            # Uploader / RegionSelector / Heatmap / Result / Feedback /
│   │                          #   IssueControls / MissedIssueForm
│   └── lib/                   # api.ts / types.ts
├── ml/
│   ├── extract_photo_features.py      # 写真→構造的参照特徴 (parquet + reference_stats)
│   ├── train_anomaly_model.py         # 参照特徴→ anomaly_model.json
│   ├── evaluate_illustration_results.py # IoU 評価 → reports/
│   ├── configs/default.yaml
│   └── requirements.txt
├── data/                      # 画像・DB・特徴量・モデル (Git 管理外。下記参照)
│   ├── illustrations_raw/     # ユーザー入力イラスト (アップロード保存先)
│   ├── photos_raw/<garment>/  # 参照用の実写服写真
│   ├── annotations/           # illustration_feedback.csv (評価アノテーション)
│   ├── features/ models/ outputs/
├── reports/                   # evaluation_report.md / evaluation_metrics.json
├── docker-compose.yml
├── README.md
└── CLAUDE.md
```

---

## セットアップと起動

> **依存のインストールは初回のみ**。`pip install -r requirements.txt` は
> 初回または `requirements.txt` 変更時のみ、`npm install` は初回または
> `package.json` 変更時のみ実行すれば十分です。普段は `uvicorn …` /
> `npm run dev` だけで起動できます。バックエンドとフロントエンドは別ターミナルで
> 同時に起動してください（例: タブを2つ）。

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
      "message": "日本語の説明文",
      "score": 0.0,
      "threshold": 0.72,
      "flagged": true
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

### `POST /api/inspect-base64`  (application/json) — 外部ツール / Photoshop 用

`/api/inspect-wrinkle` と**同じ結果形式**を返します（同じ inspection service を使用）。

```json
{
  "image_base64": "data:image/png;base64,iVBORw0K... または 生のbase64",
  "garment_type": "shirt | skirt | pants | dress | jacket | unknown",
  "selected_region": {"x": 0, "y": 0, "w": 100, "h": 100},
  "source": "web | photoshop | external"
}
```

### `GET /api/model/status`

```json
{
  "model_loaded": true,
  "model_type": "mahalanobis",
  "model_path": "…/data/models/anomaly_model.json",
  "reference_stats_loaded": true,
  "reference_stats_path": "…/data/features/reference_stats.json",
  "thresholds_loaded": true,
  "thresholds_path": "…/backend/app/config/thresholds.json",
  "available_garment_models": ["global", "shirt", "skirt", "pants", "dress", "jacket"],
  "version": "0.1.0"
}
```

### `GET /api/inspection/{id}`

過去の検査結果（issues / debug）と、その検査に紐づくフィードバック一覧を返します。

### `POST /api/feedback`  (application/json)

```json
{
  "inspection_id": "…",
  "issue_id": "gravity_inconsistency-0",
  "feedback": "correct | false_positive | missed_issue | wrong_location | wrong_reason",
  "image_id": "eval_001.png",
  "garment_type": "shirt",
  "issue_type": "joint_inconsistency",
  "original_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
  "corrected_bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
  "corrected_type": "optional",
  "confidence": 0.0,
  "severity": "low | medium | high",
  "source": "web | photoshop | external",
  "comment": "optional"
}
```

→ `{"status": "saved", "feedback_id": "…"}`

`missed_issue`（AIが見逃した不整合）は `issue_id` を省略し `corrected_bbox` と
`issue_type` を指定して送ります。フィードバックは将来のイラスト専用データセット
作成のため必ず保存されます。

> **CORS**: 既定はフロント（localhost:3000）のみ許可。Photoshop UXP など非ブラウザ/
> 不定オリジンから使う場合は環境変数 `WRINKLE_CORS_ALLOW_ALL=true` で全許可にできます。

---

## テスト

```bash
cd backend
pytest          # rule engine / 閾値 / 異常モデル / API (inspect・base64・status・feedback)
ruff check .    # Lint

cd ../frontend
npx tsc --noEmit   # 型チェック（任意）
```

---

## データ管理と Git 除外（重要）

ユーザー入力イラスト・データセット・学習済みモデル・特徴量・出力は **GitHub に載せません**。
`.gitignore` で以下を除外しています（`.gitkeep` でフォルダ骨格のみ保持）。

```
data/illustrations_raw/  data/uploads/  data/outputs/
data/photos_raw/  data/photos_masks/  data/features/  data/models/
data/*.json  data/*.csv  data/*.parquet
external/  .dataset_venv/  backend/.venv/  frontend/node_modules/  frontend/.next/
```

- **実ファイルは削除されません**。ローカルの画像データはそのまま残ります。
- **配置場所**:
  - 参照用の実写写真 → `data/photos_raw/<garment>/<dataset_name>/...`
  - ユーザー入力イラスト（評価対象・アップロード保存先）→ `data/illustrations_raw/`
  - 評価アノテーション → `data/annotations/illustration_feedback.csv`

### すでに Git に追加してしまった場合

`.gitignore` の追加だけでは**追跡済みファイルは外れません**。次で追跡からのみ外します
（`--cached` なのでローカルの実ファイルは消えません）。

```bash
git rm -r --cached data/illustrations_raw
git rm -r --cached data/photos_raw
git rm -r --cached data/photos_masks
git rm -r --cached data/features
git rm -r --cached data/models
git rm -r --cached data/uploads
git rm -r --cached data/outputs
git commit -m "chore: stop tracking local data/models/features"
```

`git status` に `data/illustrations_raw/` 以下の画像が出なくなれば成功です。

---

## 精度改善フロー

写真とイラストのドメイン差が大きいため、いきなり高精度は狙わず、計測 →
修正データ蓄積 → 閾値調整 → 専用データ蓄積の順で進めます。

1. **フィードバック保存**: UI で correct / false_positive / wrong_location /
   wrong_reason を送信。AI が見逃した不整合は「見逃しの追加」から `missed_issue`
   として bbox 付きで保存（将来のイラスト専用学習データ）。
2. **評価**: `data/annotations/illustration_feedback.csv` を用意して評価スクリプトを実行。
3. **閾値調整**: `backend/app/config/thresholds.json` で issue_type / garment_type 別の
   判定閾値を調整して誤検出を抑制。フロントの「表示スコア閾値」スライダーは**表示側**の
   フィルタで、バックエンド判定とは独立です。
4. **イラスト専用データ蓄積**: 蓄積した feedback を将来モデル学習に利用。

### 評価スクリプト

```bash
python ml/evaluate_illustration_results.py \
  --images-root data/illustrations_raw \
  --annotations data/annotations/illustration_feedback.csv \
  --output-md reports/evaluation_report.md \
  --output-json reports/evaluation_metrics.json \
  --iou-threshold 0.3
```

- 予測（flagged issue）と正解アノテーションを **IoU** でマッチングし、全体 / issue_type 別 /
  garment_type 別の **Precision / Recall / F1**、confidence 帯別の誤検出率、
  FP / FN / wrong_location / wrong_reason 一覧を出力します。
- `data/illustrations_raw` が空、または annotation CSV が無い場合は分かりやすく失敗し、
  テンプレート（`data/annotations/illustration_feedback.csv.example`）を案内します。

---

## 外部ツール連携 / Photoshop UXP（予定）

将来構成: Photoshop UXP Plugin → `POST /api/inspect-base64` → FastAPI →
検査エンジン → JSON → Photoshop パネルに bbox と説明を表示。

このフェーズで対応済み: base64 入力 API、`GET /api/model/status`、CORS の明示設定、
ファイル/base64 で同一の結果形式。Plugin 本体は次フェーズです。

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

- DINOv2 / Anomalib（PatchCore）による異常検知の本実装（`anomaly_model.py` を差し替え）
- SAM による服領域自動抽出（`segmentation.py` の `segment_with_sam`）
- MediaPipe を使った関節ルールの高度化（pose 検出時の圧縮側/伸び側の判定）
- Photoshop UXP Plugin 本体の実装
- 蓄積した feedback を使ったイラスト専用モデルの学習

> 注: `reference_stats.json` の密度判定への接続、issue_type/garment 別閾値、評価スクリプト、
> base64/status API、作画添削UI は本フェーズで実装済みです。
