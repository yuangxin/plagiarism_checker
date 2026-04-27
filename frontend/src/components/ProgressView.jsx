import { Progress, Card, Typography } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const { Text } = Typography;

export default function ProgressView({ percent, stage }) {
  return (
    <Card style={{ textAlign: 'center', padding: 24 }}>
      <LoadingOutlined style={{ fontSize: 32, color: '#1890ff', marginBottom: 16 }} />
      <div style={{ marginBottom: 8 }}>
        <Text>{stage || 'Processing...'}</Text>
      </div>
      <Progress
        percent={percent}
        status="active"
        strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
      />
    </Card>
  );
}
