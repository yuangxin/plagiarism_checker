import { Radio, Switch, Select, Space, Typography, Divider } from 'antd';
import {
  GlobalOutlined,
  RobotOutlined,
  SettingOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

export default function ConfigPanel({
  config,
  onChange,
  systemConfig,
  disabled,
}) {
  const update = (key, val) => onChange({ ...config, [key]: val });

  return (
    <div style={{ padding: '16px 0' }}>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        {/* Sensitivity */}
        <div>
          <Text strong><SettingOutlined /> Sensitivity</Text>
          <Radio.Group
            value={config.sensitivity}
            onChange={e => update('sensitivity', e.target.value)}
            optionType="button"
            buttonStyle="solid"
            size="middle"
            disabled={disabled}
            style={{ marginTop: 8, display: 'block' }}
          >
            <Radio.Button value="low">Low</Radio.Button>
            <Radio.Button value="medium">Medium</Radio.Button>
            <Radio.Button value="high">High</Radio.Button>
          </Radio.Group>
        </div>

        {/* Cross-lingual */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <GlobalOutlined />
            <Text strong>Cross-lingual</Text>
          </Space>
          <Switch
            checked={config.enableCrosslingual}
            onChange={v => update('enableCrosslingual', v)}
            disabled={disabled}
          />
        </div>

        <Divider style={{ margin: '4px 0' }} />

        {/* AI Analysis — single toggle */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <RobotOutlined />
            <Text strong>AI Analysis</Text>
          </Space>
          <Switch
            checked={config.enableAi}
            onChange={v => update('enableAi', v)}
            disabled={disabled || !systemConfig?.api_available}
          />
        </div>
        {!systemConfig?.api_available && (
          <Text type="warning" style={{ fontSize: 12 }}>API unavailable — check api_config.json</Text>
        )}
        {config.enableAi && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Space>
              <ThunderboltOutlined />
              <Text>Depth</Text>
            </Space>
            <Select
              value={config.agentDepth}
              onChange={v => update('agentDepth', v)}
              disabled={disabled}
              size="small"
              style={{ width: 120 }}
              options={[
                { value: 'quick', label: 'Quick' },
                { value: 'thorough', label: 'Thorough' },
              ]}
            />
          </div>
        )}
      </Space>
    </div>
  );
}
