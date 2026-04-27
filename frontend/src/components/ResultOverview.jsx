import { Row, Col, Card, Statistic, Tag, Alert, Result } from 'antd';
import {
  FileSearchOutlined,
  WarningOutlined,
  BarChartOutlined,
  RobotOutlined,
} from '@ant-design/icons';

export default function ResultOverview({ results }) {
  const { sent_stats = [], agent_reports = [], auto_adjusted, original_threshold, adjusted_threshold, auto_crosslingual } = results;

  // Empty results
  if (!sent_stats || sent_stats.length === 0) {
    return (
      <Result
        status="info"
        title="No similar content detected"
        subTitle="Try lowering the sensitivity level or ensure the uploaded files have overlapping subject matter."
      />
    );
  }

  const highRisk = sent_stats.filter(s => s.score >= 0.7).length;
  const avgScore = sent_stats.reduce((a, s) => a + (s.score || 0), 0) / sent_stats.length;
  const aiCount = agent_reports?.length || 0;

  return (
    <div>
      {auto_adjusted && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message={`No matches at threshold ${original_threshold?.toFixed(2)}. Auto-adjusted to ${adjusted_threshold?.toFixed(2)}.`}
        />
      )}
      {auto_crosslingual && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="Cross-lingual detection auto-enabled: detected files in different languages."
        />
      )}

      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Pairs Found"
              value={sent_stats.length}
              prefix={<FileSearchOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="High Risk"
              value={highRisk}
              prefix={<WarningOutlined />}
              valueStyle={{ color: highRisk > 0 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Avg Score"
              value={avgScore}
              precision={3}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="AI Reports"
              value={aiCount}
              suffix={aiCount > 0 ? `/ ${sent_stats.length}` : ''}
              prefix={<RobotOutlined />}
              valueStyle={{ color: aiCount > 0 ? '#1890ff' : undefined }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
}
