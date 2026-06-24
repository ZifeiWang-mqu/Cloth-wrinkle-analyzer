"use client";

import {
  ISSUE_COLORS,
  type IssueType,
  type Severity,
  TYPE_LABELS,
} from "@/lib/types";

export type SeverityFilter = "all" | Severity;
export type SortBy = "score" | "confidence";

interface Props {
  allTypes: IssueType[];
  visibleTypes: Set<IssueType>;
  onToggleType: (t: IssueType) => void;
  severityFilter: SeverityFilter;
  onSeverityChange: (s: SeverityFilter) => void;
  sortBy: SortBy;
  onSortChange: (s: SortBy) => void;
  displayThreshold: number;
  onThresholdChange: (n: number) => void;
}

export default function IssueControls({
  allTypes,
  visibleTypes,
  onToggleType,
  severityFilter,
  onSeverityChange,
  sortBy,
  onSortChange,
  displayThreshold,
  onThresholdChange,
}: Props) {
  return (
    <div className="controls">
      <div className="control-row">
        <span className="control-label">表示する種類</span>
        <div className="type-toggles">
          {allTypes.map((t) => {
            const on = visibleTypes.has(t);
            const color = ISSUE_COLORS[t];
            return (
              <button
                key={t}
                type="button"
                className={`type-chip${on ? " on" : ""}`}
                style={on ? { borderColor: color, background: `${color}22` } : undefined}
                onClick={() => onToggleType(t)}
              >
                <span className="dot" style={{ background: color }} />
                {TYPE_LABELS[t]}
              </button>
            );
          })}
        </div>
      </div>

      <div className="control-row">
        <span className="control-label">
          表示スコア閾値: {displayThreshold.toFixed(2)}
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={displayThreshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-row inline">
        <label className="control-label">
          重大度
          <select
            value={severityFilter}
            onChange={(e) => onSeverityChange(e.target.value as SeverityFilter)}
          >
            <option value="all">すべて</option>
            <option value="high">高</option>
            <option value="medium">中</option>
            <option value="low">低</option>
          </select>
        </label>
        <label className="control-label">
          並び替え
          <select value={sortBy} onChange={(e) => onSortChange(e.target.value as SortBy)}>
            <option value="score">スコア順</option>
            <option value="confidence">信頼度順</option>
          </select>
        </label>
      </div>
    </div>
  );
}
