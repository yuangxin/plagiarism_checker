import { useState, useCallback, useMemo } from 'react';
import { Layout, Typography, Button, message, Divider, ConfigProvider, Switch, Space, theme } from 'antd';
import { SearchOutlined, ClearOutlined, BulbOutlined, BulbFilled } from '@ant-design/icons';
import zhCN from 'antd/locale/zh_CN';

import ConfigPanel from './components/ConfigPanel';
import FileUpload from './components/FileUpload';
import ProgressView from './components/ProgressView';
import ResultOverview from './components/ResultOverview';
import PairDetail from './components/PairDetail';
import ExportPanel from './components/ExportPanel';

import { uploadFiles, runDetect, getConfig } from './api';

import './App.css';

const { Header, Sider, Content, Footer } = Layout;
const { Title, Text } = Typography;

function App() {
  // ── Dark mode ──────────────────────────────────────────
  const [isDark, setIsDark] = useState(false);

  // ── Config state ────────────────────────────────────────
  const [config, setConfig] = useState({
    sensitivity: 'medium',
    enableAi: false,
    agentDepth: 'quick',
    enableCrosslingual: false,
  });
  const [systemConfig, setSystemConfig] = useState(null);

  // ── File state ──────────────────────────────────────────
  const [mode, setMode] = useState('all');
  const [targetFiles, setTargetFiles] = useState([]);
  const [referenceFiles, setReferenceFiles] = useState([]);
  const [allFiles, setAllFiles] = useState([]);

  // ── Detection state ─────────────────────────────────────
  const [detecting, setDetecting] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, stage: '' });
  const [results, setResults] = useState(null);
  const [jobId, setJobId] = useState(null);

  // Load system config on mount
  useState(() => {
    getConfig().then(setSystemConfig).catch(() => {});
  });

  // ── Actions ─────────────────────────────────────────────
  const handleDetect = useCallback(async () => {
    // Validate files
    const filesToUpload = mode === 'all'
      ? allFiles
      : [...targetFiles, ...referenceFiles];

    if (filesToUpload.length < 2) {
      message.warning('Please upload at least 2 files');
      return;
    }

    if (mode === 'target' && (targetFiles.length === 0 || referenceFiles.length === 0)) {
      message.warning('Please upload both target and reference files');
      return;
    }

    setDetecting(true);
    setProgress({ percent: 10, stage: 'Uploading files...' });
    setResults(null);

    try {
      // Upload
      const targetNames = mode === 'target'
        ? targetFiles.map(f => f.name.replace(/\.[^.]+$/, ''))
        : [];
      const uploadRes = await uploadFiles(filesToUpload, {
        mode,
        targetNames,
      });

      setJobId(uploadRes.job_id);
      setProgress({ percent: 20, stage: 'Files uploaded. Starting detection...' });

      // Detect
      const detectRes = await runDetect({
        job_id: uploadRes.job_id,
        sensitivity: config.sensitivity,
        enable_ai: config.enableAi,
        agent_depth: config.agentDepth,
        enable_crosslingual: config.enableCrosslingual,
        detection_mode: mode,
        target_names: targetNames,
      });

      setProgress({ percent: 100, stage: 'Done!' });
      setResults(detectRes);
      message.success('Detection complete!');
    } catch (err) {
      message.error(`Detection failed: ${err.response?.data?.detail || err.message}`);
      console.error(err);
    } finally {
      setDetecting(false);
    }
  }, [mode, allFiles, targetFiles, referenceFiles, config]);

  const handleClear = () => {
    setResults(null);
    setJobId(null);
    setTargetFiles([]);
    setReferenceFiles([]);
    setAllFiles([]);
    setProgress({ percent: 0, stage: '' });
  };

  // ── Render ──────────────────────────────────────────────
  const bgMain = isDark ? '#141414' : '#f5f5f5';
  const bgCard = isDark ? '#1f1f1f' : '#fff';
  const borderColor = isDark ? '#303030' : '#f0f0f0';

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: isDark ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 8,
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        {/* Header */}
        <Header style={{
          background: bgCard,
          borderBottom: `1px solid ${borderColor}`,
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <SearchOutlined style={{ fontSize: 24, color: '#1890ff' }} />
            <Title level={3} style={{ margin: 0 }}>Plagiarism Detection</Title>
          </div>
          <Space>
            <Text type="secondary">Academic Integrity Analysis System</Text>
            <Switch
              checked={isDark}
              onChange={setIsDark}
              checkedChildren={<BulbFilled />}
              unCheckedChildren={<BulbOutlined />}
            />
          </Space>
        </Header>

        <Layout>
          {/* Sidebar */}
          <Sider
            width={280}
            style={{ background: bgCard, padding: '16px', borderRight: `1px solid ${borderColor}` }}
          >
            <ConfigPanel
              config={config}
              onChange={setConfig}
              systemConfig={systemConfig}
              disabled={detecting}
            />

            <Divider />

            <Button
              type="primary"
              icon={<SearchOutlined />}
              block
              size="large"
              loading={detecting}
              onClick={handleDetect}
              style={{ marginBottom: 8 }}
            >
              {detecting ? 'Detecting...' : 'Start Detection'}
            </Button>

            <Button
              icon={<ClearOutlined />}
              block
              onClick={handleClear}
              disabled={detecting}
            >
              Clear All
            </Button>
          </Sider>

          {/* Main content */}
          <Content style={{ padding: 24, background: bgMain }}>
            {!results && !detecting && (
              <div style={{ maxWidth: 800, margin: '0 auto' }}>
                <FileUpload
                  mode={mode}
                  onModeChange={setMode}
                  targetFiles={targetFiles}
                  referenceFiles={referenceFiles}
                  onTargetUpload={setTargetFiles}
                  onReferenceUpload={setReferenceFiles}
                  onAllUpload={setAllFiles}
                  disabled={detecting}
                />
              </div>
            )}

            {detecting && (
              <div style={{ maxWidth: 600, margin: '0 auto' }}>
                <ProgressView percent={progress.percent} stage={progress.stage} />
              </div>
            )}

            {results && (
              <div>
                <ResultOverview results={results} />

                {results.sent_stats?.length > 0 && (
                  <div style={{ marginTop: 24, background: bgCard, padding: 24, borderRadius: 8 }}>
                    <PairDetail results={results} />
                  </div>
                )}

                <ExportPanel jobId={jobId} />
              </div>
            )}
          </Content>
        </Layout>

        <Footer style={{ textAlign: 'center', background: bgCard }}>
          Plagiarism Detection System v2.0 | Semantic Similarity + AI Analysis
        </Footer>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
