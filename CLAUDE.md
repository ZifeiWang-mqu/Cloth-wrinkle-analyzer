# CLAUDE.md

このファイルは Claude Code / 開発エージェントがこのリポジトリで作業する際の指針です。

## プロジェクト概要

イラストの衣服の「非合理的な皺」を検知する作画支援アプリ。MVP はルールベース +
構造特徴量を優先し、深層学習は後フェーズで差し替え可能な構成にしている。

現状: **バックエンド（FastAPI）+ フロントエンド（Next.js）+ `ml/` 参照特徴抽出の
MVP が一通り完成**。高精度な異常検知（DINOv2 / Anomalib）と SAM 自動抽出は次フェーズ。

## 技術スタック

- Backend: FastAPI / Pydantic v2 / SQLAlchemy 2.0 / Uvicorn
- Image: OpenCV (headless) / NumPy / scikit-image
- Pose: MediaPipe（任意・未導入でも動作）
- DB: SQLite
- Test/Lint: pytest / ruff

## よく使うコマンド

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000   # 起動
pytest                                       # テスト
ruff check . && ruff format .                # Lint / format
```

## アーキテクチャの要点

- API 契約は `app/schemas.py` が単一の真実。TypeScript 側もこれに合わせる。
- 検査の共通ロジックは `services/inspection_service.py` の `run_inspection`。
  `inspect-wrinkle`（ファイル）と `inspect-base64`（外部ツール）が共有する:
  `segmentation -> pose -> wrinkle_edges -> feature_extraction -> anomaly_model -> rule_engine`。
- 6 要件のスコア関数は `services/rule_engine.py`。各関数は `ScoreResult`
  （score / confidence / bbox / detail）を返す。新ルール追加時はここに足し、
  `_SCORERS` と `run_all_scores` に登録する。`integrate_scores` が閾値適用と統合を行う。
- 判定閾値は `app/config/thresholds.json`（`services/thresholds.py` で読込、
  issue_type/garment_type 別）。密度判定は `data/features/reference_stats.json`
  （`services/reference_stats.py`）を使用。**バックエンド判定閾値**と
  **フロント表示閾値**は区別する。
- issue は `score` / `threshold` / `flagged` を含み、`score >= min_report_score` で返却。
- 画像処理は**例外を投げない**方針。失敗時は空結果 + `debug.notes`。
- 設定・しきい値は `app/settings.py`（環境変数 `WRINKLE_*` で上書き）。
- 外部ツール用: `routes/status.py`（`/api/model/status`, `/api/inspection/{id}`）。
  CORS は既定でフロントのみ許可、`WRINKLE_CORS_ALLOW_ALL=true` で全許可。
- DB スキーマ変更は `db/database.py` の追加カラム自動マイグレーション（ALTER TABLE）。
- フィードバックは強化版（image_id / issue_type / original・corrected bbox / confidence /
  severity / source）。`missed_issue` も保存（将来の学習データ）。

## 作業ルール

1. 実装前に変更予定ファイルと手順を提示する。
2. 大きな変更は段階的に。最後に変更ファイル一覧を出す。
3. 型定義（Pydantic / 型ヒント）を使う。
4. アップロードファイル名は信頼しない（`services/image_io.py` で UUID 化）。
5. UI 表示・説明文は日本語。
6. AI モデルは高精度より差し替え可能性を優先。
7. フィードバック保存は必ず維持（将来の学習データ）。

## 拡張ポイント

- `services/anomaly_model.py`: `BaseAnomalyModel` を継承して DINOv2 / PatchCore 実装。
- `services/segmentation.py`: `segment_with_sam` を実装して服領域を自動抽出。
- `services/rule_engine.py`: `reference_stats` に写真由来の統計を渡して密度判定を改善。
