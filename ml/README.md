# ml — 写真からの参照特徴抽出

リアルな服の写真（約100枚）から「自然な皺の構造的特徴」を抽出し、
イラスト検査時の基準（特に密度判定 `score_line_density`）に使うためのスクリプト群です。

> 写真だけで完成モデルを作るのが目的ではありません（要件 15）。
> ここでは **自然な皺パターンの参照統計** を作ります。色や質感ではなく、
> エッジ・線方向・密度・収束・明暗勾配などの **構造的特徴** のみを抽出します。

## セットアップ

```bash
cd ml
pip install -r requirements.txt   # backend の依存も一緒に入ります
```

## 写真の置き方

```
data/photos_raw/
├── shirt/   *.jpg ...
├── skirt/
├── pants/
├── dress/
└── jacket/
```

- garment_type は **フォルダ名から推定** します。
- 服領域マスクがあれば `data/photos_masks/` に同じ相対パス（拡張子は任意）で置くと、
  そのマスクの範囲・カバレッジでパッチを絞り込みます。無ければ画像全体を使います。

## 実行

```bash
# リポジトリ直下から
python ml/extract_photo_features.py
# 例: パッチサイズやマスク閾値を変更
python ml/extract_photo_features.py --patch-size 160 --min-patch-coverage 0.6
```

出力:

- `data/features/photo_reference_features.parquet`（pyarrow 不在時は `.csv`）
  - 列: `image_id, garment_type, split, patch_x/y/w/h,`
    `edge_density, dominant_line_angle, line_count, line_length_sum,`
    `gradient_orientation, local_contrast,` および
    `line_count_density, orientation_dispersion, convergence_strength, gradient_coherence`
- `data/features/reference_stats.json`
  - garment ごとの `line_count_density` / `edge_density` の mean・std（**train 分割のみ**で集計）

## 学習モデルの作成（anomaly model）

特徴量を作ったら、簡易な「学習モデル」を作れます。これは **自然な皺の特徴に
多変量ガウス分布を当てはめ**、新しい画像がどれだけ外れているかを
マハラノビス距離で 0〜1 に変換する **ワンクラス異常検知** です（numpy だけで動くため
バックエンドが追加依存なしで読み込めます）。

```bash
python ml/extract_photo_features.py     # 1) 特徴量を作る
python ml/train_anomaly_model.py        # 2) data/models/anomaly_model.json を作る
# 3) バックエンドを再起動 -> 自動でモデルを読み込みます
```

推論時の流れ: バックエンドはイラストの領域から **同じ特徴量** を計算し、
garment 別のガウス分布（無ければ global）でスコア化、`settings.anomaly_weight`
（既定 0.15）の重みで `overall_score` に混ぜ、デバッグ欄に
「異常スコア（学習モデル）」として表示します。

> モデルが無い場合はヒューリスティックな簡易モデルに自動フォールバックします。

---

## データセットの規定（重要）

写真データは以下の規定に従って配置してください。

1. **フォルダ＝種類**: `data/photos_raw/<garment>/...` の **最初の階層名** が
   garment_type になります。`<garment>` はバックエンドの種類
   （`shirt / skirt / pants / dress / jacket`）に合わせると検査時に種類別モデルが効きます。
2. **サブフォルダ可**: 再帰的に読み込むので
   `data/photos_raw/skirt/datasetA/img001.jpg` のようにデータセット名で分けてOK。
   この場合も garment は `skirt`、`image_id` は `skirt/datasetA/img001.jpg`。
3. **形式**: jpg / png / webp。ファイル名は重複させない（image_id が衝突します）。
4. **画質**: 皺がはっきり見える実写。さまざまなポーズ・照明があると良い。
   過度なフィルタ・文字入れは避け、できれば主役の服は1点。解像度は ~512px 以上推奨
   （パッチに十分な情報が入るように）。
5. **マスク（任意）**: `data/photos_masks/` に同じ相対パス（拡張子は任意、白=服）。
   あるとパッチをその範囲・カバレッジで絞り込みます。
6. **件数の目安**: garment ごとに **個別モデル** を作るには `--min-samples`
   （既定 30）以上のパッチが必要（数学的には特徴次元 6 + 1 が下限、安定には 50〜100
   パッチ以上が望ましい）。満たない garment は自動で global モデルに代替されます。
   写真100枚 × 1枚あたり数十パッチ = 通常は十分な数になります。
7. **分割はスクリプトに任せる**: train/test は **画像単位** で自動分割
   （`test_ratio` / `seed`）。同じ写真のパッチが train/test に跨らないようになっており、
   モデルは **train 分割のみ** で学習します。手動で混在させないでください。
8. **一貫性**: 1つのモデルに使う抽出は `patch_size` / `stride` を揃える。

## 複数の「100枚データセット」の扱い方

目的に応じて 3 通りあります。

- **A. 自然例を増やしたい（推奨・最も一般的）**:
  すべて同じ garment フォルダに入れます（トレーサビリティのため
  `…/<garment>/<dataset名>/` のサブフォルダ推奨）。`extract` → `train` を再実行するだけ。
  データが増えるほどガウス分布が安定し、判定が良くなります。
  ※ このモデルは「正常（自然）データ」だけで学習します。**異常イラストは入れないでください**
  （ワンクラス前提が崩れます）。
- **B. 出所やスタイルが違うが同列に使いたい**:
  これも A と同様に統合してOK。モデルは garment 単位・出所非依存です。
  どのデータ由来かは `image_id`（サブフォルダ名）で追跡できます。
- **C. 検証用に分けたい / バージョン管理したい**:
  片方を検証用に残す、もしくは別パスで学習して比較します。
  `--features` / `--output` を変えれば複数のモデル JSON を作れ、
  バックエンドは `settings.model_path`（既定 `data/models/anomaly_model.json`）を読みます。
  環境変数 `WRINKLE_ANOMALY_MODEL_FILENAME` 等でも切り替え可能です。

```bash
# 例: データセットを足して作り直す
#   data/photos_raw/skirt/2026-06-A/*.jpg
#   data/photos_raw/skirt/2026-06-B/*.jpg
python ml/extract_photo_features.py
python ml/train_anomaly_model.py
# 例: 検証用に別モデルを作る
python ml/train_anomaly_model.py \
  --features data/features/photo_reference_features.parquet \
  --output  data/models/anomaly_model_v2.json
```

---

## 設計メモ

- **バックエンドと同一の特徴量**: パッチ特徴は `backend/app/services` の
  `wrinkle_edges` + `feature_extraction` を再利用します。これにより写真の統計と
  イラスト検査時の値が直接比較可能になります。
- **リーク防止の分割**: train/test は **画像単位** で決定（同一写真のパッチが
  train/test に跨りません）。`test_ratio` / `seed` で制御。
- **差し替え可能**: `--extractor classic`（既定）/ `dino`（将来の DINOv2 用スタブ）。

## バックエンドへの反映（任意）

`reference_stats.json` を `score_line_density` に渡すと密度判定が写真基準に近づきます。
`backend/app/services/rule_engine.py` の `DEFAULT_DENSITY_STATS` を JSON の
`per_garment[*].line_count_density.{mean,std}` で置き換える / 読み込む実装を足すだけです。

## 今後（未実装）

- `build_reference_dataset.py` / `evaluate.py`（評価・PR曲線など）
- DINOv2 特徴 + Anomalib/PatchCore による異常検知（`anomaly_model.py` を差し替え）
- `reference_stats.json` を `score_line_density` に接続
