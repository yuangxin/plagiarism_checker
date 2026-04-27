import { Button, Space, Typography, theme } from 'antd';
import {
  DownloadOutlined,
  FileExcelOutlined,
  FileTextOutlined,
  FileWordOutlined,
} from '@ant-design/icons';
import { getExportUrl } from '../api';

const { Text } = Typography;

export default function ExportPanel({ jobId }) {
  if (!jobId) return null;

  const { token } = theme.useToken();

  const formats = [
    { key: 'csv', label: 'CSV', icon: <FileExcelOutlined />, ext: 'csv' },
    { key: 'json', label: 'JSON', icon: <FileTextOutlined />, ext: 'json' },
    { key: 'docx', label: 'Word Detail', icon: <FileWordOutlined />, ext: 'docx' },
    { key: 'docx_summary', label: 'Word Summary', icon: <FileWordOutlined />, ext: 'docx' },
    { key: 'para_csv', label: 'Paragraph CSV', icon: <FileExcelOutlined />, ext: 'csv' },
    { key: 'para_docx', label: 'Paragraph Word', icon: <FileWordOutlined />, ext: 'docx' },
  ];

  return (
    <div style={{ marginTop: 16, padding: 16, background: token.colorBgContainer, borderRadius: 8, border: `1px solid ${token.colorBorderSecondary}` }}>
      <Text strong style={{ marginBottom: 8, display: 'block' }}>
        <DownloadOutlined /> Export Reports
      </Text>
      <Space wrap>
        {formats.map(f => (
          <Button
            key={f.key}
            icon={f.icon}
            href={getExportUrl(jobId, f.key)}
            target="_blank"
          >
            {f.label}
          </Button>
        ))}
      </Space>
    </div>
  );
}
