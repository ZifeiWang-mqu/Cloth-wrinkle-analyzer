"use client";

import { useRef, useState } from "react";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function ImageUploader({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handle(file: File | undefined | null) {
    if (!file) return;
    if (!ACCEPT.includes(file.type)) {
      setError("jpg / png / webp の画像を選択してください。");
      return;
    }
    setError(null);
    onFile(file);
  }

  return (
    <div>
      <div
        className={`dropzone${drag ? " drag" : ""}`}
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (disabled) return;
          handle(e.dataTransfer.files?.[0]);
        }}
        role="button"
        aria-disabled={disabled}
      >
        <div>画像をドラッグ＆ドロップ</div>
        <div className="hint">またはクリックして選択（jpg / png / webp）</div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT.join(",")}
        style={{ display: "none" }}
        onChange={(e) => handle(e.target.files?.[0])}
      />
      {error && <div className="error">{error}</div>}
    </div>
  );
}
