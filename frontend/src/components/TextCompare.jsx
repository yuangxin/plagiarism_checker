import { useMemo } from 'react';
import { theme } from 'antd';
import './TextCompare.css';

/**
 * Side-by-side text comparison with similarity highlighting.
 * Each "hit" marks a sentence as high/medium/low similarity.
 */
export default function TextCompare({ detail, hits = [] }) {
  if (!detail) return null;

  const { token } = theme.useToken();
  const isDark = theme.useToken().theme?.algorithm === theme.darkAlgorithm;
  const { sentences = {}, pair = [] } = detail;

  // Build a map of sentence_id -> hit info for left and right
  const leftHits = useMemo(() => {
    const m = {};
    hits.forEach(h => {
      m[h.sent_id_i] = m[h.sent_id_i] || [];
      m[h.sent_id_i].push(h);
    });
    return m;
  }, [hits]);

  const rightHits = useMemo(() => {
    const m = {};
    hits.forEach(h => {
      m[h.sent_id_j] = m[h.sent_id_j] || [];
      m[h.sent_id_j].push(h);
    });
    return m;
  }, [hits]);

  const leftId = pair[0] || 'Target';
  const rightId = pair[1] || 'Reference';
  const leftSents = sentences[leftId] || {};
  const rightSents = sentences[rightId] || {};

  const leftEntries = Object.entries(leftSents).sort(([a], [b]) => Number(a) - Number(b));
  const rightEntries = Object.entries(rightSents).sort(([a], [b]) => Number(a) - Number(b));

  const maxLen = Math.max(leftEntries.length, rightEntries.length);

  function getHighlightClass(side, sentId) {
    const hitMap = side === 'left' ? leftHits : rightHits;
    const hitList = hitMap[sentId];
    if (!hitList || hitList.length === 0) return '';
    const bestSim = Math.max(...hitList.map(h => h.adjusted_sim ?? h.sim));
    if (bestSim >= 0.90) return 'highlight-high';
    if (bestSim >= 0.80) return 'highlight-medium';
    return 'highlight-low';
  }

  function getTooltip(side, sentId) {
    const hitMap = side === 'left' ? leftHits : rightHits;
    const hitList = hitMap[sentId];
    if (!hitList || hitList.length === 0) return '';
    const h = hitList[0];
    const sim = h.adjusted_sim ?? h.sim;
    let tip = `Similarity: ${(sim * 100).toFixed(1)}%`;
    if (h.citation_label) tip += ` | ${h.citation_label}`;
    return tip;
  }

  const rows = [];
  for (let i = 0; i < maxLen; i++) {
    rows.push({ left: leftEntries[i], right: rightEntries[i] });
  }

  return (
    <div>
      {/* Legend */}
      <div className="compare-legend">
        <span className="legend-item">
          <span className="legend-box" style={{ background: '#ff6b6b' }} />
          High (≥90%)
        </span>
        <span className="legend-item">
          <span className="legend-box" style={{ background: '#ffd93d' }} />
          Medium (80-90%)
        </span>
        <span className="legend-item">
          <span className="legend-box" style={{ background: '#a8e6cf' }} />
          Low (&lt;80%)
        </span>
        <span className="legend-item">
          <span className="legend-box" style={{ background: '#d4a5ff', border: '2px dashed #8b5cf6' }} />
          Citation
        </span>
      </div>

      <div className="compare-container">
        <div className="compare-pane" style={{ borderColor: '#1890ff' }}>
          <div className="compare-header" style={{ background: isDark ? '#111d2c' : '#e3f2fd' }}>{leftId}</div>
          <div className="compare-body">
            {leftEntries.map(([id, info]) => (
              <div
                key={id}
                className={`compare-sentence ${getHighlightClass('left', Number(id))}`}
                title={getTooltip('left', Number(id))}
              >
                {info.text}
              </div>
            ))}
          </div>
        </div>
        <div className="compare-pane" style={{ borderColor: '#fa8c16' }}>
          <div className="compare-header" style={{ background: isDark ? '#2b1d11' : '#fff3e0' }}>{rightId}</div>
          <div className="compare-body">
            {rightEntries.map(([id, info]) => (
              <div
                key={id}
                className={`compare-sentence ${getHighlightClass('right', Number(id))}`}
                title={getTooltip('right', Number(id))}
              >
                {info.text}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
