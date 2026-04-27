import { Card, Alert, Typography, List, Collapse, Tag } from 'antd';
import {
  CheckCircleOutlined,
  WarningOutlined,
  RobotOutlined,
} from '@ant-design/icons';

const { Paragraph, Title } = Typography;

export default function AgentReport({ report }) {
  if (!report) return null;

  // report is a string (markdown) or an object with structured data
  const reportText = typeof report === 'string' ? report : report.report || '';

  // Try to extract structured info from the markdown
  const isPlagiarism = reportText.includes('Potential plagiarism') || reportText.includes('⚠️');
  const confidenceMatch = reportText.match(/(?:Confidence|置信度)[：:]\s*([\d.]+%?)/i);
  const confidence = confidenceMatch ? confidenceMatch[1] : null;

  return (
    <Card
      style={{ borderLeft: '4px solid #1890ff', marginTop: 16 }}
      title={
        <span>
          <RobotOutlined style={{ marginRight: 8 }} />
          AI Analysis Report
        </span>
      }
    >
      {/* Verdict badge */}
      <Alert
        type={isPlagiarism ? 'error' : 'success'}
        showIcon
        icon={isPlagiarism ? <WarningOutlined /> : <CheckCircleOutlined />}
        message={isPlagiarism ? 'Potential plagiarism detected' : 'No obvious plagiarism detected'}
        description={confidence ? `Confidence: ${confidence}` : ''}
        style={{ marginBottom: 16 }}
      />

      {/* Report content */}
      <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>
        {reportText
          .replace(/##\s*/g, '')           // remove h2 markers
          .replace(/\*\*/g, '')            // remove bold markers
          .replace(/###\s*/g, '')          // remove h3 markers
        }
      </div>
    </Card>
  );
}
