import { Upload, Radio, Space, Typography, List, Tag } from 'antd';
import { InboxOutlined, FileTextOutlined } from '@ant-design/icons';

const { Dragger } = Upload;
const { Text } = Typography;

export default function FileUpload({
  mode,
  onModeChange,
  targetFiles,
  referenceFiles,
  onTargetUpload,
  onReferenceUpload,
  onAllUpload,
  disabled,
}) {
  const uploadProps = (onChange) => ({
    multiple: true,
    accept: '.txt,.md',
    showUploadList: false,
    beforeUpload: (file, fileList) => {
      onChange(fileList);
      return false; // prevent auto upload
    },
    disabled,
  });

  return (
    <div>
      {/* Mode selector */}
      <Radio.Group
        value={mode}
        onChange={e => onModeChange(e.target.value)}
        optionType="button"
        buttonStyle="solid"
        style={{ marginBottom: 16 }}
      >
        <Radio.Button value="all">All Files Comparison</Radio.Button>
        <Radio.Button value="target">Target vs Reference</Radio.Button>
      </Radio.Group>

      {mode === 'all' ? (
        /* All files mode */
        <Dragger {...uploadProps(onAllUpload)} style={{ padding: '20px 0' }}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: 40, color: '#1890ff' }} />
          </p>
          <p className="ant-upload-text">Click or drag files here</p>
          <p className="ant-upload-hint">Support .txt and .md files (minimum 2 files)</p>
        </Dragger>
      ) : (
        /* Target vs Reference mode */
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div>
            <Tag color="blue" style={{ marginBottom: 8 }}>Target Files</Tag>
            <Dragger {...uploadProps(onTargetUpload)} style={{ padding: '12px 0' }}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined style={{ fontSize: 28, color: '#1890ff' }} />
              </p>
              <p className="ant-upload-text" style={{ fontSize: 13 }}>Target files</p>
            </Dragger>
            {targetFiles.length > 0 && (
              <List
                size="small"
                dataSource={targetFiles}
                renderItem={f => (
                  <List.Item>
                    <FileTextOutlined style={{ marginRight: 8 }} />
                    <Text>{f.name}</Text>
                  </List.Item>
                )}
              />
            )}
          </div>
          <div>
            <Tag color="orange" style={{ marginBottom: 8 }}>Reference Files</Tag>
            <Dragger {...uploadProps(onReferenceUpload)} style={{ padding: '12px 0' }}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined style={{ fontSize: 28, color: '#fa8c16' }} />
              </p>
              <p className="ant-upload-text" style={{ fontSize: 13 }}>Reference files</p>
            </Dragger>
            {referenceFiles.length > 0 && (
              <List
                size="small"
                dataSource={referenceFiles}
                renderItem={f => (
                  <List.Item>
                    <FileTextOutlined style={{ marginRight: 8 }} />
                    <Text>{f.name}</Text>
                  </List.Item>
                )}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
