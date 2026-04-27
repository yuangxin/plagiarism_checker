import { useState, useMemo } from 'react';
import {
  Select, Card, Statistic, Row, Col, Tag, Tabs, Collapse,
  List, Typography, Divider, Progress, Space, Alert, theme,
} from 'antd';
import {
  SwapOutlined,
  WarningOutlined,
  SafetyOutlined,
} from '@ant-design/icons';
import TextCompare from './TextCompare';
import AgentReport from './AgentReport';

const { Text, Paragraph } = Typography;

function getRiskLevel(score) {
  if (score >= 0.7) return { label: 'High Risk', color: 'red', icon: <WarningOutlined /> };
  if (score >= 0.5) return { label: 'Medium Risk', color: 'orange', icon: <WarningOutlined /> };
  return { label: 'Low Risk', color: 'green', icon: <SafetyOutlined /> };
}

export default function PairDetail({ results }) {
  const { token } = theme.useToken();
  const bgBlue = token.colorInfoBg || '#e3f2fd';
  const bgOrange = token.colorWarningBg || '#fff3e0';
  const { sent_stats = [], sent_details = [], para_details = [], agent_reports = [] } = results;
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [viewLevel, setViewLevel] = useState('sentence');

  if (!sent_stats.length) return null;

  // Build pair options for selector
  const pairOptions = sent_stats.map((s, idx) => {
    const pair = s.pair;
    const risk = getRiskLevel(s.score);
    return {
      value: idx,
      label: (
        <Space>
          <span>{pair[0]} ↔ {pair[1]}</span>
          <Tag color={risk.color}>{risk.label}</Tag>
          <Text type="secondary">{s.score.toFixed(3)}</Text>
          {s.cross_lingual && <Tag color="purple">Cross-lingual</Tag>}
        </Space>
      ),
      search: `${pair[0]} ${pair[1]}`,
    };
  });

  const stat = sent_stats[selectedIdx];
  const detail = sent_details[selectedIdx];
  const paraDetail = para_details.find(
    d => d.pair[0] === stat.pair[0] && d.pair[1] === stat.pair[1]
  );

  // Find agent report for this pair
  const agentReport = agent_reports.find(
    r => r.pair && r.pair[0] === stat.pair[0] && r.pair[1] === stat.pair[1]
  );

  if (!stat) return null;

  const risk = getRiskLevel(stat.score);
  const hits = detail?.hits || [];

  // Citation summary
  const citationCounts = { explicit: 0, general: 0, none: 0 };
  hits.forEach(h => {
    const label = h.citation_label || '';
    if (label.includes('明确引用') || label.includes('规范引用')) citationCounts.explicit++;
    else if (label.includes('一般引用') || label.includes('LLM')) citationCounts.general++;
    else citationCounts.none++;
  });

  return (
    <div>
      {/* Pair selector */}
      <Select
        value={selectedIdx}
        onChange={setSelectedIdx}
        options={pairOptions}
        style={{ width: '100%', marginBottom: 16 }}
        showSearch
        optionFilterProp="search"
      />

      {/* Metrics row */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={4}>
          <Statistic title="Score" value={stat.score} precision={3} />
        </Col>
        <Col span={4}>
          <Statistic title="Risk" value={risk.label} prefix={risk.icon}
            valueStyle={{ color: risk.color === 'red' ? '#cf1322' : risk.color === 'orange' ? '#d48806' : '#3f8600', fontSize: 16 }}
          />
        </Col>
        <Col span={4}>
          <Statistic title="Matches" value={stat.count} />
        </Col>
        <Col span={4}>
          <Statistic title="Avg Similarity" value={stat.mean_sim} precision={3} />
        </Col>
        <Col span={4}>
          <Statistic title="Coverage" value={stat.coverage_min}
            precision={3} suffix={`(${stat.coverage_a?.toFixed?.(2) ?? ''} / ${stat.coverage_b?.toFixed?.(2) ?? ''})`}
          />
        </Col>
        <Col span={4}>
          <Statistic title="Max Similarity" value={stat.max_sim} precision={3} />
        </Col>
      </Row>

      {/* Cross-lingual tag */}
      {stat.cross_lingual && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message={`Cross-lingual match: ${stat.lang_a?.toUpperCase() || '?'} → ${stat.lang_b?.toUpperCase() || '?'}`}
        />
      )}

      {/* View level tabs */}
      <Tabs
        activeKey={viewLevel}
        onChange={setViewLevel}
        type="card"
        items={[
          { key: 'sentence', label: 'Sentences' },
          { key: 'paragraph', label: 'Paragraphs', disabled: !paraDetail },
        ]}
      />

      {viewLevel === 'sentence' ? (
        <>
          {/* Text comparison */}
          <TextCompare detail={detail} hits={hits} />

          {/* Citation summary */}
          {hits.length > 0 && (citationCounts.explicit + citationCounts.general > 0) && (
            <Card size="small" style={{ marginTop: 12 }} title="Citation Assessment">
              <Space>
                <Tag color="green">Explicit: {citationCounts.explicit}</Tag>
                <Tag color="blue">General: {citationCounts.general}</Tag>
                <Tag>No citation: {citationCounts.none}</Tag>
              </Space>
            </Card>
          )}

          {/* Detailed match list (collapsed) */}
          <Collapse
            style={{ marginTop: 16 }}
            items={[{
              key: 'matches',
              label: `Detailed Matches (${hits.length})`,
              children: (
                <List
                  size="small"
                  dataSource={hits.slice(0, 20)}
                  renderItem={(h, idx) => (
                    <List.Item style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                        <div style={{ background: bgBlue, padding: 8, borderRadius: 4 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>Target (sent {h.sent_id_i})</Text>
                          <Paragraph style={{ margin: 0, fontSize: 13 }}>{h.text_i}</Paragraph>
                        </div>
                        <div style={{ background: bgOrange, padding: 8, borderRadius: 4 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>Reference (sent {h.sent_id_j})</Text>
                          <Paragraph style={{ margin: 0, fontSize: 13 }}>{h.text_j}</Paragraph>
                        </div>
                      </div>
                      <Space style={{ marginTop: 4 }} size="small">
                        <Tag color={
                          (h.adjusted_sim ?? h.sim) >= 0.9 ? 'red' :
                          (h.adjusted_sim ?? h.sim) >= 0.8 ? 'orange' : 'green'
                        }>
                          {(h.adjusted_sim ?? h.sim).toFixed(3)}
                        </Tag>
                        {h.citation_label && <Tag>{h.citation_label}</Tag>}
                        {h.citation_explanation && (
                          <Text type="secondary" style={{ fontSize: 12 }}>{h.citation_explanation}</Text>
                        )}
                      </Space>
                    </List.Item>
                  )}
                />
              ),
            }]}
          />

          {/* AI Report (inline) */}
          <AgentReport report={agentReport?.report} />
        </>
      ) : (
        /* Paragraph view */
        paraDetail && (
          <div>
            <Row gutter={16} style={{ marginBottom: 12 }}>
              <Col><Statistic title="Para Score" value={paraDetail.score} precision={3} /></Col>
              <Col><Statistic title="Para Matches" value={paraDetail.count} /></Col>
              <Col><Statistic title="Para Coverage" value={paraDetail.coverage_min} precision={3} /></Col>
            </Row>
            <List
              size="small"
              dataSource={paraDetail.matches || []}
              renderItem={(m, idx) => (
                <List.Item style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div style={{ background: bgBlue, padding: 8, borderRadius: 4 }}>
                      <Text type="secondary" style={{ fontSize: 11 }}>Para {m.para_id_i}</Text>
                      <Paragraph style={{ margin: 0, fontSize: 13 }}>{m.text_i}</Paragraph>
                    </div>
                    <div style={{ background: bgOrange, padding: 8, borderRadius: 4 }}>
                      <Text type="secondary" style={{ fontSize: 11 }}>Para {m.para_id_j}</Text>
                      <Paragraph style={{ margin: 0, fontSize: 13 }}>{m.text_j}</Paragraph>
                    </div>
                  </div>
                  <Tag color={m.sim >= 0.8 ? 'red' : m.sim >= 0.6 ? 'orange' : 'green'} style={{ marginTop: 4 }}>
                    Similarity: {m.sim.toFixed(3)}
                  </Tag>
                </List.Item>
              )}
            />
            <AgentReport report={agentReport?.report} />
          </div>
        )
      )}
    </div>
  );
}
