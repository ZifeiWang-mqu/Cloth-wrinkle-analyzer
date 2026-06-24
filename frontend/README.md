# Frontend — イラスト皺チェッカー (Next.js)

画像アップロード → 服領域のドラッグ選択 → 検査 → 赤枠オーバーレイ + 日本語の
issue 一覧 + フィードバック送信、までを行う作画支援 UI です。

## 構成

- Next.js 14 (App Router) + React 18 + TypeScript（プレーン CSS、ビルド依存少なめ）
- `lib/types.ts` … バックエンド `schemas.py` と対応する API 型
- `lib/api.ts` … `/api/inspect-wrinkle`・`/api/feedback` クライアント
- `components/`
  - `ImageUploader.tsx` … ドラッグ&ドロップ / クリックで画像選択
  - `RegionSelector.tsx` … 画像表示・服領域のドラッグ選択（自然座標へ変換）
  - `HeatmapOverlay.tsx` … 重大度別の検出枠を画像に重ねて描画
  - `ResultPanel.tsx` … 総合スコア・issue 一覧・スコアバー・デバッグ情報
  - `FeedbackButtons.tsx` … correct / false_positive / … の送信
- `app/page.tsx` … 全体のオーケストレーション

## セットアップ

```bash
cd frontend
cp .env.local.example .env.local   # 必要なら API のURLを変更
npm install
npm run dev                        # http://localhost:3000
```

バックエンド（FastAPI）を `http://localhost:8000` で起動しておいてください。
別ホストの場合は `.env.local` の `NEXT_PUBLIC_API_BASE` を変更します。

## 使い方

1. 画像（jpg / png / webp）を選択
2. 服の種類を選択（任意）
3. プレビュー上をドラッグして服領域を指定（任意。無ければ画像全体）
4. 「検査する」を押すと、問題箇所が画像上に色枠（赤=高 / 橙=中 / 黄=低）で表示
5. 各 issue にフィードバックを送信（将来の学習データとして保存されます）

> 座標はオリジナル画像のピクセル系で扱い、表示サイズに合わせて拡縮して描画するため、
> ウィンドウサイズが変わっても枠の位置はずれません。
