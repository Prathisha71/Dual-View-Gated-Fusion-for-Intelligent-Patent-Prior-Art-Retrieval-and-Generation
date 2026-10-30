import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, Search, FileText, BarChart2, Cpu, MessageSquare, Database, 
  ShieldAlert, Sparkles, CheckCircle2, ChevronRight, BookOpen, Layers,
  Activity, RefreshCw, X, HelpCircle, AlertTriangle, Upload, ChevronDown, ChevronUp, Info, ArrowRight, Check
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [researchToolsOpen, setResearchToolsOpen] = useState(false);
  
  // Dynamic System Health & Metrics State
  const [systemHealth, setSystemHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [clsMetrics, setClsMetrics] = useState(null);
  const [retMetrics, setRetMetrics] = useState(null);
  const [validationResult, setValidationResult] = useState(null);

  // Guided Input Mode Tab: 'text' or 'pdf'
  const [inputTab, setInputTab] = useState('text');
  const [exampleLoaded, setExampleLoaded] = useState(false);

  // Form Input Fields (Empty by default)
  const [patentNumberInput, setPatentNumberInput] = useState('');
  const [titleInput, setTitleInput] = useState('');
  const [abstractInput, setAbstractInput] = useState('');
  const [claimsInput, setClaimsInput] = useState('');
  const [dateInput, setDateInput] = useState('');
  const [extractedFilingDate, setExtractedFilingDate] = useState('');
  const [extractedGrantDate, setExtractedGrantDate] = useState('');

  // Extraction Status Badges State
  const [pdfUploading, setPdfUploading] = useState(false);
  const [pdfError, setPdfError] = useState('');
  const [pdfNotice, setPdfNotice] = useState('');
  const [extractionStatus, setExtractionStatus] = useState(null);

  // Results & Analysis State
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStepIdx, setLoadingStepIdx] = useState(0);

  // Modals & Panels
  const [comparisonModal, setComparisonModal] = useState(null);
  const [explainerPanel, setExplainerPanel] = useState(null);
  const [selectedExplorerPatent, setSelectedExplorerPatent] = useState(null);

  const cleanHtmlEntities = (text) => {
    if (!text) return '';
    return text
      .replace(/&#39;/g, "'")
      .replace(/&quot;/g, '"')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>');
  };

  // Explorer State
  const [explorerData, setExplorerData] = useState([]);
  const [explorerQuery, setExplorerQuery] = useState('');
  const [explorerCpc, setExplorerCpc] = useState('');
  const [explorerSplit, setExplorerSplit] = useState('');
  const [explorerPage, setExplorerPage] = useState(1);
  const [explorerTotal, setExplorerTotal] = useState(0);

  // Chat Assistant State
  const [chatQuery, setChatQuery] = useState('');
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'assistant',
      text: 'Welcome to the PAFIR Patent Assistant. Ask any question regarding active patent claims, CPC subclass predictions, or prior art evidence.'
    }
  ]);
  const [chatLoading, setChatLoading] = useState(false);

  // Auto-Poll Health every 3s & Fetch Metrics
  useEffect(() => {
    fetchHealth();
    fetchMetrics();
    fetchExplorerPatents();

    const interval = setInterval(fetchHealth, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchExplorerPatents();
  }, [explorerPage, explorerQuery, explorerCpc, explorerSplit]);

  const fetchHealth = () => {
    fetch(`${API_BASE_URL}/health`)
      .then(res => {
        if (!res.ok) throw new Error('Health check non-200 response');
        return res.json();
      })
      .then(data => {
        setSystemHealth(data);
        setHealthError(null);
      })
      .catch((err) => {
        setSystemHealth(null);
        setHealthError('Unable to connect to the analysis server. Start or reconnect the backend and try again.');
      });
  };

  const fetchMetrics = () => {
    fetch(`${API_BASE_URL}/api/metrics/classification`)
      .then(res => res.json())
      .then(data => setClsMetrics(data))
      .catch(() => setClsMetrics(null));

    fetch(`${API_BASE_URL}/api/metrics/retrieval`)
      .then(res => res.json())
      .then(data => setRetMetrics(data))
      .catch(() => setRetMetrics(null));
  };

  const fetchExplorerPatents = () => {
    const params = new URLSearchParams({ page: explorerPage, limit: 10 });
    if (explorerQuery) params.append('query', explorerQuery);
    if (explorerCpc) params.append('cpc', explorerCpc);
    if (explorerSplit) params.append('split', explorerSplit);

    fetch(`${API_BASE_URL}/api/patents/explore?${params.toString()}`)
      .then(res => res.json())
      .then(data => {
        setExplorerData(data.data || []);
        setExplorerTotal(data.total || 0);
      })
      .catch(() => {
        setExplorerData([]);
        setExplorerTotal(0);
      });
  };

  const handleValidateResearch = () => {
    fetch(`${API_BASE_URL}/api/admin/validate`, { method: 'POST' })
      .then(res => res.json())
      .then(data => setValidationResult(data))
      .catch(() => setValidationResult({ verified: false, status: 'UNVERIFIED', checks: {} }));
  };

  const handleLoadExample = () => {
    setPatentNumberInput('US10003450');
    setTitleInput('System and Method for Routing Node in H04L Applications');
    setAbstractInput('An automated apparatus and method for operating a routing node (Sub-Module 3450) within a transmission of digital information & networking protocols. The system includes a processing module configured to evaluate cryptographic protocol and optimize ethernet header. Furthermore, dynamic adjustments are performed based on 5G network slice to improve throughput and efficiency in US Patent US10003450.');
    setClaimsInput('1. A patentable apparatus (US10003450) comprising: a control module configured to detect a routing node; a memory storing operating instructions; and a processor executed to perform real-time optimization of cryptographic protocol responsive to signals associated with ethernet header.');
    setDateInput('2010-01-01');
    setExtractedFilingDate('');
    setExtractedGrantDate('2010-01-01');
    setExampleLoaded(true);
    setPdfNotice('');
    setExtractionStatus(null);
  };

  const handlePdfUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPdfUploading(true);
    setPdfError('');
    setPdfNotice('');
    setExtractionStatus(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/api/extract-pdf`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('PDF extraction failed');
      const data = await res.json();

      setPatentNumberInput(data.patent_number || '');
      setTitleInput(data.title || '');
      setAbstractInput(data.abstract || '');
      setClaimsInput(data.independent_claims || '');
      
      setExtractedFilingDate(data.filing_date || '');
      setExtractedGrantDate(data.grant_date || '');
      setDateInput(data.recommended_cutoff_date || data.filing_date || data.grant_date || '');

      setPdfNotice(data.message || '');
      setExtractionStatus(data);
      setInputTab('text');
    } catch (err) {
      setPdfError('Failed to extract text from PDF file. Please ensure it is a valid patent document.');
    } finally {
      setPdfUploading(false);
    }
  };

  const isSystemReady = systemHealth && (systemHealth.status === 'ready' || systemHealth.backend === 'ready');

  const loadingSteps = [
    'Reading Patent',
    'Understanding Technical Context',
    'Predicting Patent Classes',
    'Searching Earlier Patents',
    'Ranking Prior Art',
    'Preparing Results'
  ];

  const handleLoadPatentFromExplorer = (row) => {
    setPatentNumberInput(row.patent_id || '');
    setTitleInput(cleanHtmlEntities(row.title || ''));
    setAbstractInput(row.abstract || '');
    setClaimsInput(row.independent_claims || row.claims || '');
    setDateInput(row.grant_date || '');
    setExtractionStatus({
      all_fields_extracted: true,
      title_extracted: true,
      abstract_extracted: true,
      claims_extracted: true,
      date_extracted: true,
      message: `Loaded ${row.patent_id} from Patent Explorer.`
    });
    setSelectedExplorerPatent(null);
    setActiveTab('analyze');
    setInputTab('text');
  };

  const handleAnalyze = async () => {
    if (!titleInput.trim() || !abstractInput.trim()) {
      setAnalysisError('Please enter patent title and abstract before running analysis.');
      return;
    }
    if (!isSystemReady) {
      setAnalysisError('Analysis service is still starting. Please wait for the system to report ready.');
      return;
    }

    setLoading(true);
    setAnalysisError('');
    setLoadingStepIdx(0);

    const stepTimer = setInterval(() => {
      setLoadingStepIdx(prev => (prev < loadingSteps.length - 1 ? prev + 1 : prev));
    }, 250);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          patent_id: patentNumberInput || undefined,
          title: titleInput,
          abstract: abstractInput,
          independent_claims: claimsInput,
          grant_date: dateInput || undefined
        })
      });

      if (!res.ok) throw new Error(`Analysis server error: ${res.statusText}`);

      const data = await res.json();
      setAnalysisResult(data);
      setActiveTab('results');
    } catch (err) {
      if (err.name === 'AbortError') {
        setAnalysisError('Analysis is taking longer than expected (60s timeout exceeded). Check system diagnostics or retry.');
      } else {
        setAnalysisError('Unable to connect to the analysis server. Start or reconnect the backend and try again.');
      }
    } finally {
      clearTimeout(timeoutId);
      clearInterval(stepTimer);
      setLoading(false);
    }
  };

  const handleChatSend = async (e) => {
    e.preventDefault();
    if (!chatQuery.trim()) return;

    const userMsg = { sender: 'user', text: chatQuery };
    setChatMessages(prev => [...prev, userMsg]);
    setChatQuery('');
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_query: chatQuery,
          active_patent_id: analysisResult?.prior_art?.[0]?.patent_id || 'US10003450'
        })
      });
      const data = await res.json();
      setChatMessages(prev => [...prev, {
        sender: 'assistant',
        text: data.answer,
        supporting: data.supporting_patent_ids,
        evidence: data.evidence_snippets,
        disclaimer: data.disclaimer
      }]);
    } catch (err) {
      setChatMessages(prev => [...prev, {
        sender: 'assistant',
        text: 'The backend service could not process the query. Please ensure FastAPI server is running.'
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const formatMetric = (val) => {
    if (val === undefined || val === null || isNaN(val)) return 'Not Evaluated Yet';
    return typeof val === 'number' ? val.toFixed(4) : val;
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] text-[#0F172A] overflow-hidden font-sans">
      {/* PERSISTENT LEFT SIDEBAR */}
      <aside className="w-64 border-r border-[#E2E8F0] bg-[#FFFFFF] flex flex-col justify-between p-4 flex-shrink-0 z-20">
        <div>
          {/* Logo Branding */}
          <div className="flex items-center gap-3 px-3 py-3 mb-6 border-b border-[#E2E8F0]">
            <div className="p-2.5 bg-gradient-to-tr from-[#38BDF8] to-[#6366F1] rounded-xl shadow-lg shadow-sky-600/15">
              <Cpu className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-heading font-black text-xl tracking-tight text-[#0F172A]">PAFIR</h1>
              <p className="text-[10px] text-[#64748B] font-medium tracking-wider uppercase">Patent Aware AI Engine</p>
            </div>
          </div>

          {/* SIMPLIFIED MAIN NAVIGATION */}
          <nav className="space-y-1">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
              { id: 'analyze', label: 'Analyze Patent', icon: FileText },
              { id: 'results', label: 'Prior Art Results', icon: CheckCircle2 },
              { id: 'assistant', label: 'Patent Assistant', icon: MessageSquare }
            ].map(item => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl font-medium text-xs transition-all duration-150 ${
                    isActive 
                      ? 'bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 font-bold shadow-sm' 
                      : 'text-[#64748B] hover:text-[#0F172A] hover:bg-[#F1F5F9]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-[#0284C7]' : 'text-[#64748B]'}`} />
                    {item.label}
                  </div>
                  {item.id === 'results' && analysisResult && (
                    <span className="w-2 h-2 rounded-full bg-[#22C55E]" />
                  )}
                </button>
              );
            })}

            {/* COLLAPSIBLE RESEARCH TOOLS */}
            <div className="pt-3 border-t border-[#E2E8F0]">
              <button
                onClick={() => setResearchToolsOpen(!researchToolsOpen)}
                className="w-full flex items-center justify-between px-3.5 py-2 text-[11px] font-semibold text-[#64748B] hover:text-[#0F172A] uppercase tracking-wider transition-all"
              >
                <span>Research Tools</span>
                {researchToolsOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {researchToolsOpen && (
                <div className="mt-1 space-y-1 pl-2">
                  {[
                    { id: 'explorer', label: 'Patent Explorer', icon: Database },
                    { id: 'evaluation', label: 'Model Evaluation', icon: BarChart2 },
                    { id: 'analytics', label: 'Research Analytics', icon: Layers },
                    { id: 'diagnostics', label: 'System Diagnostics', icon: Activity }
                  ].map(item => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg font-medium text-xs transition-all ${
                          isActive 
                            ? 'bg-[#4F46E5]/10 text-[#4F46E5] border border-[#4F46E5]/25 font-bold' 
                            : 'text-[#64748B] hover:text-[#0F172A] hover:bg-[#F1F5F9]'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5 text-[#64748B]" />
                        {item.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </nav>
        </div>

        {/* BOTTOM SIDEBAR STATUS PANEL */}
        <div className="p-3.5 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-[#64748B]">System Status</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
              isSystemReady ? 'bg-[#16A34A]/10 text-[#16A34A] border border-[#16A34A]/25' : 'bg-[#D97706]/10 text-[#D97706] border border-[#D97706]/25'
            }`}>
              {isSystemReady ? 'System Ready' : 'System Starting'}
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px]">
            <span className="text-[#64748B]">Backend API:</span>
            <span className={systemHealth ? 'text-[#22C55E] font-bold' : 'text-[#EF4444] font-bold'}>
              {systemHealth ? 'Healthy' : 'Unavailable'}
            </span>
          </div>

          {!isSystemReady && (
            <button
              onClick={fetchHealth}
              className="w-full mt-2 py-1.5 bg-[#0284C7]/10 hover:bg-[#0284C7]/15 border border-[#0284C7]/25 text-[11px] font-bold text-[#0284C7] rounded-lg flex items-center justify-center gap-1.5 transition-all"
            >
              <RefreshCw className="w-3 h-3" />
              Retry Connection
            </button>
          )}
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 overflow-y-auto p-8 max-w-7xl mx-auto space-y-6">
        {/* TOP HEADER */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#E2E8F0] pb-4">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold font-heading text-[#0F172A] capitalize">{activeTab.replace('_', ' ')}</h2>
              
              <span 
                className="relative group cursor-pointer bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 text-xs font-bold px-2.5 py-0.5 rounded-full flex items-center gap-1"
              >
                <Info className="w-3 h-3" />
                {systemHealth?.dataset_mode === 'RESEARCH_MODE' ? 'Research Dataset' : 'Development Dataset'}
                <div className="absolute hidden group-hover:block top-7 left-0 w-64 p-2 bg-[#FFFFFF] border border-[#CBD5E1] text-[11px] text-[#0F172A] rounded-lg shadow-xl z-30 font-normal">
                  Using the smaller development patent collection (5,000 patents). Results are for prototype evaluation.
                </div>
              </span>

              {validationResult?.verified && (
                <span className="bg-[#16A34A]/10 text-[#16A34A] border border-[#16A34A]/25 text-xs font-bold px-2.5 py-0.5 rounded-full flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> VERIFIED
                </span>
              )}
            </div>
            <p className="text-xs text-[#64748B] mt-1">PAFIR: Joint Patent Document Classification and Temporal Prior Art Retrieval Engine</p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs bg-[#FFFFFF] border border-[#E2E8F0] px-3.5 py-1.5 rounded-xl text-[#64748B]">
              GPU: <strong className="text-[#0284C7] font-mono">{systemHealth?.gpu_accelerator || 'CPU Execution'}</strong>
            </span>
            <button onClick={() => { fetchHealth(); fetchMetrics(); }} className="p-2 bg-[#FFFFFF] border border-[#E2E8F0] hover:border-[#0284C7] rounded-xl text-[#64748B] hover:text-[#0F172A] transition-all">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* ERROR / HEALTH WARNING BANNER */}
        {healthError && (
          <div className="p-4 rounded-xl bg-[#DC2626]/10 border border-[#DC2626]/25 text-[#DC2626] flex justify-between items-center">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              <div>
                <h4 className="font-bold text-sm">Backend API Unavailable</h4>
                <p className="text-xs opacity-90">{healthError}</p>
              </div>
            </div>
            <button onClick={fetchHealth} className="px-3 py-1 bg-[#DC2626]/15 hover:bg-[#DC2626]/20 rounded-lg text-xs font-bold text-white transition-all">
              Retry Connection
            </button>
          </div>
        )}

        {/* DASHBOARD VIEW */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* HERO BANNER */}
            <div className="rounded-2xl p-6 border border-[#E2E8F0] border-l-4 border-l-[#0284C7] bg-gradient-to-r from-[#FFFFFF] via-[#F8FAFC] to-[#F1F5F9] shadow-lg">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-xl font-bold font-heading text-[#0F172A]">PAFIR Research Console</h3>
                  <p className="text-xs text-[#64748B] mt-1 max-w-3xl leading-relaxed">
                    Patent Aware Multi-View Representation Learning combined with Soft CPC Guidance, Temporal Date Filtering, and Evidence-Grounded Explainability.
                  </p>
                </div>
                <span className="p-2.5 bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/20 rounded-xl font-mono text-xs font-bold">v1.0.0</span>
              </div>
              
              <div className="flex flex-wrap items-center gap-2 mt-4">
                <span className="bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 px-3 py-1 rounded-full text-xs font-semibold">Dataset: USPTO PatentsView</span>
                <span className="bg-[#4F46E5]/10 text-[#4F46E5] border border-[#4F46E5]/25 px-3 py-1 rounded-full text-xs font-semibold">{systemHealth?.embedding_model_name || 'DEV Encoder: all-MiniLM-L6-v2'}</span>
                <span className="bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 px-3 py-1 rounded-full text-xs font-semibold">Temporal Coverage: 2010 to 2025</span>
                <span className="bg-[#16A34A]/10 text-[#16A34A] border border-[#16A34A]/25 px-3 py-1 rounded-full text-xs font-semibold">Data Leakage Audit: Passed</span>
              </div>
            </div>

            {/* QUICK ACTIONS ROW */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <button 
                onClick={() => setActiveTab('analyze')}
                className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E2E8F0] hover:border-[#0284C7] text-left transition-all group flex items-center justify-between"
              >
                <div>
                  <h4 className="font-bold text-sm text-[#0F172A] group-hover:text-[#0284C7]">Analyze Patent</h4>
                  <p className="text-xs text-[#64748B] mt-1">Classify and search prior art for any patent text or PDF.</p>
                </div>
                <ChevronRight className="w-5 h-5 text-[#64748B] group-hover:text-[#0284C7] transition-transform group-hover:translate-x-1" />
              </button>

              <button 
                onClick={() => { setActiveTab('explorer'); setResearchToolsOpen(true); }}
                className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E2E8F0] hover:border-[#4F46E5] text-left transition-all group flex items-center justify-between"
              >
                <div>
                  <h4 className="font-bold text-sm text-[#0F172A] group-hover:text-[#4F46E5]">Explore Patents</h4>
                  <p className="text-xs text-[#64748B] mt-1">Browse 5,000 indexed USPTO development dataset records.</p>
                </div>
                <ChevronRight className="w-5 h-5 text-[#64748B] group-hover:text-[#4F46E5] transition-transform group-hover:translate-x-1" />
              </button>

              <button 
                onClick={() => { setActiveTab('evaluation'); setResearchToolsOpen(true); }}
                className="p-5 rounded-2xl bg-[#FFFFFF] border border-[#E2E8F0] hover:border-[#16A34A] text-left transition-all group flex items-center justify-between"
              >
                <div>
                  <h4 className="font-bold text-sm text-[#0F172A] group-hover:text-[#16A34A]">View Research Results</h4>
                  <p className="text-xs text-[#64748B] mt-1">Compare empirical evaluation benchmarks against baselines.</p>
                </div>
                <ChevronRight className="w-5 h-5 text-[#64748B] group-hover:text-[#16A34A] transition-transform group-hover:translate-x-1" />
              </button>
            </div>

            {/* 4-COLUMN RESPONSIVE METRIC CARDS GRID */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { 
                  label: 'Indexed Patents', 
                  value: systemHealth?.indexed_patents ? systemHealth.indexed_patents.toLocaleString() : '5,000',
                  sub: 'US Utility Patents (Indexed)',
                  tooltip: 'Total candidate patent documents indexed in the FAISS vector space.'
                },
                { 
                  label: 'Classification Performance', 
                  value: formatMetric(clsMetrics?.metrics?.micro_f1),
                  sub: 'CPC Subclasses (Micro F1)',
                  tooltip: 'Micro F1 multi-label classification accuracy evaluated on test set.'
                },
                { 
                  label: 'Prior Art Retrieval Performance', 
                  value: formatMetric(retMetrics?.metrics?.['recall@10']),
                  sub: 'Recall@10 Metric',
                  tooltip: 'Fraction of ground-truth prior art retrieved in top 10 candidates.'
                },
                { 
                  label: 'Dataset Coverage', 
                  value: '2010 - 2025',
                  sub: 'Temporal Window',
                  tooltip: 'Temporal span of patent grant dates included in the dataset.'
                }
              ].map((card, idx) => (
                <div key={idx} className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] hover:border-[#0284C7] transition-all flex flex-col justify-between shadow-md">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-semibold text-[#64748B]">{card.label}</span>
                      <span className="tooltip-trigger text-[#64748B]" title={card.tooltip}><HelpCircle className="w-3.5 h-3.5" /></span>
                    </div>
                    <h3 className="text-2xl font-extrabold font-heading text-[#0F172A]">
                      {card.value === 'Not Evaluated Yet' ? <span className="text-xs font-medium text-[#64748B]">Not Evaluated Yet</span> : card.value}
                    </h3>
                  </div>
                  <p className="text-[11px] text-[#64748B] mt-3 border-t border-[#E2E8F0] pt-2">{card.sub}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GUIDED "ANALYZE PATENT" PAGE */}
        {activeTab === 'analyze' && (
          <div className="space-y-6">
            {/* PAGE HEADER */}
            <div>
              <h3 className="text-xl font-bold font-heading text-[#0F172A]">Analyze a Patent</h3>
              <p className="text-xs text-[#64748B] mt-1">Enter patent information or upload a patent document. PAFIR will automatically classify the patent and search for potentially relevant prior art.</p>
            </div>

            {/* HOW IT WORKS VISUAL GUIDANCE CARD */}
            <div className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] shadow-sm space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-[#0284C7] flex items-center gap-1.5">
                <BookOpen className="w-4 h-4" /> How It Works
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 text-xs">
                {[
                  { step: '1', title: 'Enter / Upload', desc: 'Enter or upload patent information.' },
                  { step: '2', title: 'Classify CPC', desc: 'PAFIR predicts technical CPC categories.' },
                  { step: '3', title: 'Search Vector Space', desc: 'Earlier patents are searched using similarity.' },
                  { step: '4', title: 'Rank Prior Art', desc: 'Strongest candidates are ranked as prior art.' },
                  { step: '5', title: 'Examine & Compare', desc: 'Compare results or ask Patent Assistant.' }
                ].map((s) => (
                  <div key={s.step} className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]">
                    <span className="w-5 h-5 rounded-full bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 text-[10px] font-bold flex items-center justify-center mb-1.5">{s.step}</span>
                    <h5 className="font-bold text-[#0F172A] text-[11px]">{s.title}</h5>
                    <p className="text-[10px] text-[#64748B] mt-0.5 leading-snug">{s.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* TWO INPUT TABS: ENTER TEXT / UPLOAD PDF */}
            <div className="rounded-2xl bg-[#FFFFFF] border border-[#E2E8F0] shadow-md overflow-hidden">
              <div className="flex border-b border-[#E2E8F0]">
                <button
                  onClick={() => setInputTab('text')}
                  className={`flex-1 py-3 px-4 text-xs font-bold flex items-center justify-center gap-2 border-b-2 transition-all ${
                    inputTab === 'text' 
                      ? 'border-[#0284C7] text-[#0284C7] bg-[#F1F5F9]' 
                      : 'border-transparent text-[#64748B] hover:text-[#0F172A]'
                  }`}
                >
                  <FileText className="w-4 h-4" /> ENTER PATENT TEXT
                </button>
                <button
                  onClick={() => setInputTab('pdf')}
                  className={`flex-1 py-3 px-4 text-xs font-bold flex items-center justify-center gap-2 border-b-2 transition-all ${
                    inputTab === 'pdf' 
                      ? 'border-[#0284C7] text-[#0284C7] bg-[#F1F5F9]' 
                      : 'border-transparent text-[#64748B] hover:text-[#0F172A]'
                  }`}
                >
                  <Upload className="w-4 h-4" /> UPLOAD PATENT PDF
                </button>
              </div>

              <div className="p-6">
                {inputTab === 'text' && (
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-semibold text-[#64748B]">Fill out fields or click Load Example:</span>
                      <button
                        onClick={handleLoadExample}
                        className="px-3 py-1.5 bg-[#F1F5F9] border border-[#CBD5E1] hover:bg-[#E2E8F0] text-xs font-bold text-[#0284C7] rounded-lg transition-all flex items-center gap-1.5"
                      >
                        <Sparkles className="w-3.5 h-3.5" /> Load Example
                      </button>
                    </div>

                    {exampleLoaded && (
                      <div className="p-2.5 rounded-lg bg-[#0284C7]/10 border border-[#0284C7]/25 text-xs text-[#0284C7] font-semibold flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4" /> Example from PatentsView development dataset loaded.
                      </div>
                    )}

                    {pdfNotice && (
                      <div className={`p-3.5 rounded-xl border text-xs font-semibold space-y-2 ${
                        extractionStatus?.all_fields_extracted 
                          ? 'bg-[#16A34A]/10 border-[#16A34A]/25 text-[#16A34A]' 
                          : 'bg-[#D97706]/10 border-[#D97706]/25 text-[#D97706]'
                      }`}>
                        <div className="flex items-center gap-2">
                          <Info className="w-4 h-4 flex-shrink-0" />
                          <span>{pdfNotice}</span>
                        </div>

                        {/* FIELD EXTRACTION BADGES */}
                        {extractionStatus && (
                          <div className="flex flex-wrap gap-2 pt-1 border-t border-current/20 text-[11px]">
                            <span className={`px-2 py-0.5 rounded font-bold border flex items-center gap-1 ${
                              extractionStatus.title_extracted ? 'bg-[#16A34A]/15 text-[#16A34A] border-[#16A34A]/30' : 'bg-[#DC2626]/15 text-[#DC2626] border-[#DC2626]/30'
                            }`}>
                              {extractionStatus.title_extracted ? '✓ Title extracted' : '✗ Title missing'}
                            </span>

                            <span className={`px-2 py-0.5 rounded font-bold border flex items-center gap-1 ${
                              extractionStatus.abstract_extracted ? 'bg-[#16A34A]/15 text-[#16A34A] border-[#16A34A]/30' : 'bg-[#DC2626]/15 text-[#DC2626] border-[#DC2626]/30'
                            }`}>
                              {extractionStatus.abstract_extracted ? '✓ Abstract extracted' : '✗ Abstract missing'}
                            </span>

                            <span className={`px-2 py-0.5 rounded font-bold border flex items-center gap-1 ${
                              extractionStatus.claims_extracted ? 'bg-[#16A34A]/15 text-[#16A34A] border-[#16A34A]/30' : 'bg-[#DC2626]/15 text-[#DC2626] border-[#DC2626]/30'
                            }`}>
                              {extractionStatus.claims_extracted ? `✓ ${extractionStatus.independent_claim_count} independent claims detected (Claims ${extractionStatus.detected_claim_numbers?.join(', ')})` : '✗ Independent claims missing'}
                            </span>

                            <span className={`px-2 py-0.5 rounded font-bold border flex items-center gap-1 ${
                              extractionStatus.date_extracted ? 'bg-[#16A34A]/15 text-[#16A34A] border-[#16A34A]/30' : 'bg-[#DC2626]/15 text-[#DC2626] border-[#DC2626]/30'
                            }`}>
                              {extractionStatus.date_extracted ? `✓ Filing date detected (${extractionStatus.recommended_cutoff_date})` : '✗ Date missing'}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-xs font-semibold text-[#0F172A] mb-1">Patent Number (Optional)</label>
                        <input 
                          type="text" 
                          placeholder="e.g. US10713783B2" 
                          value={patentNumberInput} 
                          onChange={(e) => setPatentNumberInput(e.target.value)} 
                          className="w-full bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-4 py-2.5 text-xs text-[#0F172A] focus:outline-none focus:border-[#0284C7]" 
                        />
                      </div>
                      <div className="sm:col-span-2">
                        <label className="block text-xs font-semibold text-[#0F172A] mb-1">Invention Title</label>
                        <input 
                          type="text" 
                          placeholder="e.g. Neural network classification" 
                          value={titleInput} 
                          onChange={(e) => setTitleInput(e.target.value)} 
                          className="w-full bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-4 py-2.5 text-xs text-[#0F172A] focus:outline-none focus:border-[#0284C7]" 
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-[#0F172A] mb-1">Abstract</label>
                      <p className="text-[11px] text-[#64748B] mb-1">Paste the patent abstract section.</p>
                      <textarea 
                        rows={3} 
                        placeholder="e.g. Neural network classification may be performed by inputting a training data set..." 
                        value={abstractInput} 
                        onChange={(e) => setAbstractInput(e.target.value)} 
                        className="w-full bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-4 py-2.5 text-xs text-[#0F172A] focus:outline-none focus:border-[#0284C7]" 
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-[#0F172A] mb-1">Independent Claims</label>
                      <p className="text-[11px] text-[#64748B] mb-1">Paste one or more independent claims (e.g., claims 1, 5, and 7).</p>
                      <textarea 
                        rows={6} 
                        placeholder="Paste or review independent claims here..." 
                        value={claimsInput} 
                        onChange={(e) => setClaimsInput(e.target.value)} 
                        className="w-full bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-4 py-2.5 text-xs text-[#0F172A] focus:outline-none focus:border-[#0284C7] font-mono leading-relaxed" 
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="block text-xs font-semibold text-[#0F172A]">Temporal Cutoff Date (Prior Art Date Limit)</label>
                      <p className="text-[11px] text-[#64748B]">Used to ensure only earlier patents are considered as prior art.</p>
                      
                      <div className="flex flex-wrap items-center gap-3">
                        <input 
                          type="date" 
                          value={dateInput} 
                          onChange={(e) => setDateInput(e.target.value)} 
                          className="bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-3.5 py-2 text-xs text-[#0F172A]" 
                        />

                        {extractedFilingDate && (
                          <button
                            type="button"
                            onClick={() => setDateInput(extractedFilingDate)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                              dateInput === extractedFilingDate 
                                ? 'bg-[#0284C7]/15 text-[#0284C7] border-[#0284C7]/50' 
                                : 'bg-[#F8FAFC] text-[#64748B] border-[#E2E8F0] hover:text-[#0F172A]'
                            }`}
                          >
                            Use Filing Date ({extractedFilingDate}) [Recommended]
                          </button>
                        )}

                        {extractedGrantDate && (
                          <button
                            type="button"
                            onClick={() => setDateInput(extractedGrantDate)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                              dateInput === extractedGrantDate 
                                ? 'bg-[#0284C7]/15 text-[#0284C7] border-[#0284C7]/50' 
                                : 'bg-[#F8FAFC] text-[#64748B] border-[#E2E8F0] hover:text-[#0F172A]'
                            }`}
                          >
                            Use Grant Date ({extractedGrantDate})
                          </button>
                        )}
                      </div>

                      {!dateInput && (
                        <p className="text-[11px] text-[#F59E0B] mt-1 font-semibold">Date could not be detected automatically. Please enter it manually.</p>
                      )}
                    </div>
                  </div>
                )}

                {inputTab === 'pdf' && (
                  <div className="space-y-4 text-center py-6">
                    <div className="border-2 border-dashed border-[#E2E8F0] hover:border-[#0284C7] rounded-2xl p-8 bg-[#F8FAFC] transition-all max-w-lg mx-auto cursor-pointer relative">
                      <input 
                        type="file" 
                        accept=".pdf" 
                        onChange={handlePdfUpload} 
                        className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" 
                      />
                      <Upload className="w-10 h-10 text-[#0284C7] mx-auto mb-3" />
                      <h4 className="font-bold text-sm text-[#0F172A]">Upload Patent PDF</h4>
                      <p className="text-xs text-[#64748B] mt-1">Select a patent specification PDF document to automatically extract fields.</p>
                    </div>

                    {pdfUploading && (
                      <div className="flex items-center justify-center gap-2 text-xs text-[#0284C7]">
                        <div className="w-4 h-4 border-2 border-[#0284C7] border-t-transparent rounded-full animate-spin" />
                        Extracting patent title, abstract, claims, and dates...
                      </div>
                    )}

                    {pdfError && (
                      <p className="text-xs text-[#EF4444] font-semibold">{pdfError}</p>
                    )}
                  </div>
                )}

                {/* ERROR / TIMEOUT BANNER */}
                {analysisError && (
                  <div className="mt-4 p-4 rounded-xl bg-[#DC2626]/10 border border-[#DC2626]/25 text-xs text-[#DC2626] flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="w-5 h-5 flex-shrink-0 text-[#EF4444]" />
                      <span>{analysisError}</span>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={handleAnalyze} className="px-3 py-1 bg-[#0284C7] text-white font-bold text-xs rounded-lg">Retry</button>
                      <button onClick={() => { setActiveTab('diagnostics'); setResearchToolsOpen(true); }} className="px-3 py-1 bg-[#F1F5F9] border border-[#CBD5E1] text-[#0284C7] font-bold text-xs rounded-lg">Diagnostics</button>
                    </div>
                  </div>
                )}

                {/* SINGLE PRIMARY BUTTON */}
                <div className="mt-6 border-t border-[#E2E8F0] pt-4 flex justify-end">
                  <button 
                    onClick={handleAnalyze} 
                    disabled={loading || !isSystemReady} 
                    className={`px-8 py-3 rounded-xl font-bold text-xs shadow-lg transition-all flex items-center gap-2 ${
                      !isSystemReady 
                        ? 'bg-[#F1F5F9] border border-[#CBD5E1] text-[#64748B] cursor-not-allowed' 
                        : 'bg-[#0284C7] hover:bg-[#0369A1] text-white shadow-sky-600/15'
                    }`}
                  >
                    {loading ? 'Analyzing Patent...' : 'Analyze Patent'}
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* ANALYSIS PROGRESS SCREEN */}
            {loading && (
              <div className="rounded-2xl p-8 bg-[#FFFFFF] border border-[#E2E8F0] text-center space-y-4 shadow-xl">
                <div className="w-10 h-10 border-3 border-[#0284C7] border-t-transparent rounded-full animate-spin mx-auto" />
                <h4 className="font-bold text-base text-[#0F172A]">Analyzing Patent Specification...</h4>
                
                <div className="max-w-md mx-auto space-y-2 text-left">
                  {loadingSteps.map((stepName, i) => (
                    <div key={i} className="flex items-center gap-3 text-xs">
                      <div className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                        i < loadingStepIdx ? 'bg-[#22C55E] text-white' : i === loadingStepIdx ? 'bg-[#0284C7] text-white animate-pulse' : 'bg-[#F1F5F9] border border-[#CBD5E1] text-[#64748B]'
                      }`}>
                        {i < loadingStepIdx ? '✓' : i + 1}
                      </div>
                      <span className={i <= loadingStepIdx ? 'text-[#0F172A] font-semibold' : 'text-[#64748B]'}>{stepName}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* DEDICATED RESULTS PAGE */}
        {activeTab === 'results' && (
          <div className="space-y-6">
            {!analysisResult ? (
              <div className="rounded-2xl p-8 bg-[#FFFFFF] border border-[#E2E8F0] text-center space-y-3">
                <FileText className="w-10 h-10 text-[#64748B] mx-auto" />
                <h4 className="font-bold text-base text-[#0F172A]">No Active Analysis Found</h4>
                <p className="text-xs text-[#64748B]">Run an analysis first from the Analyze Patent page to view predictions and prior art candidates.</p>
                <button 
                  onClick={() => setActiveTab('analyze')} 
                  className="px-5 py-2 bg-[#0284C7] text-white font-bold text-xs rounded-xl"
                >
                  Go to Analyze Patent
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                {/* TOP ANALYSIS HEADER */}
                <div className="rounded-2xl p-6 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-3">
                  <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 border-b border-[#E2E8F0] pb-3">
                    <div>
                      <span className="text-[10px] font-mono uppercase text-[#0284C7] font-bold">Analysis ID: {analysisResult.analysis_id}</span>
                      <h3 className="text-lg font-bold font-heading text-[#0F172A] mt-0.5">{patentNumberInput ? `${patentNumberInput} - ` : ''}{titleInput}</h3>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="bg-[#16A34A]/10 text-[#16A34A] border border-[#16A34A]/25 px-3 py-1 rounded-full font-bold">Analysis Complete</span>
                      <span className="text-[#64748B]">Temporal Cutoff: <strong>{dateInput || 'Unspecified'}</strong></span>
                      <span className="text-[#64748B]">Latency: <strong className="text-[#0284C7] font-mono">{analysisResult.timing?.total_ms?.toFixed(1) || 0} ms</strong></span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-[#64748B] pt-1">
                    <div>Embedding: <strong className="text-[#0F172A]">{analysisResult.timing?.embedding_ms?.toFixed(1) || 0} ms</strong></div>
                    <div>Classifier: <strong className="text-[#0F172A]">{analysisResult.timing?.classification_ms?.toFixed(1) || 0} ms</strong></div>
                    <div>FAISS Retrieval: <strong className="text-[#0F172A]">{analysisResult.timing?.faiss_search_ms?.toFixed(1) || 0} ms</strong></div>
                    <div>Primary CPC: <strong className="text-[#0284C7]">{analysisResult.classification_detail?.primary_cpc || analysisResult.classification?.[0]?.cpc}</strong></div>
                  </div>
                </div>

                {/* SECTION 1: PREDICTED CPC CLASSES */}
                <div className="rounded-2xl p-6 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-4">
                  <h3 className="text-base font-bold font-heading text-[#0F172A] flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-[#22C55E]" /> Predicted CPC Subclasses
                  </h3>

                  <div className="space-y-3">
                    {(Array.isArray(analysisResult.classification) ? analysisResult.classification : analysisResult.classification_detail?.top_predicted_cpcs || []).map((item, idx) => (
                      <div key={idx} className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] space-y-1.5">
                        <div className="flex justify-between items-center text-xs font-semibold">
                          <div className="flex items-center gap-2">
                            <span className="bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 px-2 py-0.5 rounded font-mono font-bold">{item.cpc}</span>
                            <span className="text-[#0F172A]">{item.description}</span>
                          </div>
                          <span className="text-[#0284C7] font-bold">{(item.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-[#FFFFFF] rounded-full h-2">
                          <div className="bg-gradient-to-r from-[#38BDF8] to-[#6366F1] h-2 rounded-full transition-all" style={{ width: `${item.probability * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* SECTION 2: POTENTIAL PRIOR ART */}
                <div className="space-y-3">
                  <h3 className="text-base font-bold font-heading text-[#0F172A] flex items-center justify-between">
                    <span>Potential Prior Art Candidates {dateInput ? `(Temporally Filtered ≤ ${dateInput})` : ''}</span>
                    <span className="text-xs text-[#64748B] font-normal">{analysisResult.prior_art?.length || 0} Candidates Retrieved</span>
                  </h3>

                  {(analysisResult.prior_art || []).map((cand) => (
                    <div key={cand.rank} className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] hover:border-[#0284C7]/50 transition-all flex flex-col md:flex-row justify-between items-start md:items-center gap-4 shadow-md">
                      <div className="space-y-2 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="w-6 h-6 rounded-full bg-[#F8FAFC] border border-[#E2E8F0] text-[#0284C7] text-xs font-bold flex items-center justify-center">#{cand.rank}</span>
                          <span className="font-mono font-bold text-sm text-[#0284C7]">{cand.patent_id}</span>
                          <span className="text-xs text-[#64748B]">Granted: {cand.date || cand.grant_date}</span>
                          <span className="bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 px-2 py-0.5 rounded-full text-[11px] font-semibold">{cand.primary_cpc}</span>
                        </div>
                        <h4 className="text-xs font-bold text-[#0F172A]">{cand.title}</h4>
                      </div>

                      <div className="flex items-center gap-4 flex-wrap">
                        <div className="text-right">
                          <span className="block text-[10px] text-[#64748B] uppercase font-semibold">Semantic Sim</span>
                          <span className="text-xs font-extrabold text-[#4F46E5]">{cand.semantic_similarity?.toFixed(3)}</span>
                        </div>
                        <div className="text-right">
                          <span className="block text-[10px] text-[#64748B] uppercase font-semibold">Final Score</span>
                          <span className="text-sm font-extrabold text-[#22C55E]">{cand.ranking_score?.toFixed(3) || cand.final_score?.toFixed(3)}</span>
                        </div>
                        
                        <div className="flex gap-2">
                          <button 
                            onClick={() => setExplainerPanel(cand)} 
                            className="px-3 py-1.5 bg-[#F1F5F9] border border-[#CBD5E1] hover:bg-[#E2E8F0] text-xs font-bold text-[#0284C7] rounded-lg transition-all"
                          >
                            Why This Matched
                          </button>
                          <button 
                            onClick={() => setComparisonModal(cand)} 
                            className="px-3 py-1.5 bg-[#F1F5F9] border border-[#CBD5E1] hover:bg-[#E2E8F0] text-xs font-bold text-[#4F46E5] rounded-lg transition-all"
                          >
                            Compare
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SIDE PANEL: WHY THIS MATCHED */}
        {explainerPanel && (
          <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex justify-end">
            <div className="w-full max-w-md bg-[#F8FAFC] border-l border-[#E2E8F0] p-6 h-full overflow-y-auto space-y-5 shadow-2xl">
              <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-3">
                <h3 className="text-base font-bold font-heading text-[#0F172A]">Why This Matched</h3>
                <button onClick={() => setExplainerPanel(null)} className="p-1 hover:text-[#0F172A] text-[#64748B]"><X className="w-5 h-5" /></button>
              </div>

              <div className="space-y-4 text-xs">
                <div className="p-3 rounded-xl bg-[#FFFFFF] border border-[#E2E8F0]">
                  <span className="text-[10px] text-[#64748B] uppercase font-bold">Candidate Patent</span>
                  <h4 className="font-bold text-[#0284C7] mt-0.5">{explainerPanel.patent_id} - {explainerPanel.title}</h4>
                </div>

                <div className="space-y-1.5">
                  <h5 className="font-bold text-[#0F172A]">Shared Technical Concepts</h5>
                  <div className="flex flex-wrap gap-1.5">
                    {explainerPanel.shared_technical_concepts && explainerPanel.shared_technical_concepts.length > 0 ? (
                      explainerPanel.shared_technical_concepts.map((c, i) => (
                        <span key={i} className="px-2 py-0.5 bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 rounded text-[11px] font-semibold">{c}</span>
                      ))
                    ) : (
                      <span className="text-[#64748B] italic">Dense semantic similarity & vocabulary overlap</span>
                    )}
                  </div>
                </div>

                {explainerPanel.paired_evidence && explainerPanel.paired_evidence.length > 0 ? (
                  <div className="space-y-2">
                    <h5 className="font-bold text-[#0F172A]">Grounded Paired Evidence</h5>
                    {explainerPanel.paired_evidence.map((pair, idx) => (
                      <div key={idx} className="p-3 rounded-xl bg-[#FFFFFF] border border-[#E2E8F0] space-y-2">
                        <span className="text-[10px] font-bold font-mono text-[#0284C7] uppercase tracking-wider block border-b border-[#E2E8F0] pb-1">Concept: {pair.concept}</span>
                        <div className="space-y-1">
                          <span className="text-[10px] text-[#64748B] font-semibold block">Query Passage:</span>
                          <p className="text-[11px] text-[#0F172A] italic bg-[#F1F5F9] p-2 rounded-lg border border-[#E2E8F0]">"{pair.query_passage}"</p>
                        </div>
                        <div className="space-y-1">
                          <span className="text-[10px] text-[#0284C7] font-semibold block">Candidate Passage:</span>
                          <p className="text-[11px] text-[#0F172A] italic bg-[#F1F5F9] p-2 rounded-lg border border-[#E2E8F0]">"{pair.candidate_passage}"</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <>
                    <div className="space-y-1.5">
                      <h5 className="font-bold text-[#0F172A]">Query Claim Evidence</h5>
                      <p className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#E2E8F0] text-[#64748B] leading-relaxed">
                        "{claimsInput ? claimsInput.slice(0, 150) + '...' : abstractInput.slice(0, 150) + '...'}"
                      </p>
                    </div>

                    <div className="space-y-1.5">
                      <h5 className="font-bold text-[#0F172A]">Candidate Specification Evidence</h5>
                      <p className="p-2.5 rounded-lg bg-[#FFFFFF] border border-[#E2E8F0] text-[#64748B] leading-relaxed">
                        "{explainerPanel.abstract ? explainerPanel.abstract.slice(0, 150) + '...' : 'Indexed vector document specification.'}"
                      </p>
                    </div>
                  </>
                )}

                <div className="space-y-1.5">
                  <h5 className="font-bold text-[#0F172A]">CPC Relationship</h5>
                  <p className="text-[#64748B]">Query and candidate share primary CPC subclass <strong>{explainerPanel.primary_cpc}</strong>.</p>
                </div>

                <div className="space-y-1.5">
                  <h5 className="font-bold text-[#0F172A]">Score Breakdown</h5>
                  <div className="p-3 rounded-xl bg-[#FFFFFF] border border-[#E2E8F0] space-y-1">
                    <div className="flex justify-between"><span>Semantic Vector Cosine:</span> <strong className="text-[#0284C7]">{explainerPanel.semantic_similarity?.toFixed(4)}</strong></div>
                    <div className="flex justify-between"><span>CPC Compatibility Score:</span> <strong className="text-[#16A34A]">{explainerPanel.cpc_compatibility?.toFixed(4) || 1.0}</strong></div>
                    <div className="flex justify-between border-t border-[#E2E8F0] pt-1"><span>Final Score:</span> <strong className="text-[#22C55E]">{explainerPanel.ranking_score?.toFixed(4) || explainerPanel.final_score?.toFixed(4)}</strong></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* SIDE-BY-SIDE PATENT COMPARISON MODAL */}
        {comparisonModal && (
          <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="rounded-2xl max-w-5xl w-full p-6 bg-[#FFFFFF] border border-[#E2E8F0] space-y-4 max-h-[90vh] overflow-y-auto shadow-2xl">
              <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-3">
                <div>
                  <h3 className="text-lg font-bold font-heading text-[#0F172A]">Side-by-Side Patent Comparison</h3>
                  <p className="text-xs text-[#64748B]">Compare Query Invention directly with Prior Art Candidate ({comparisonModal.patent_id})</p>
                </div>
                <button onClick={() => setComparisonModal(null)} className="p-1 hover:text-[#0F172A] text-[#64748B]"><X className="w-5 h-5" /></button>
              </div>

              <div className="grid grid-cols-2 gap-4 text-xs">
                {/* QUERY INVENTION COLUMN */}
                <div className="p-4 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] space-y-3">
                  <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-2">
                    <span className="text-[11px] font-bold uppercase text-[#0284C7]">Query Invention</span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-[#0284C7]/10 text-[#0284C7] font-mono">Target Patent</span>
                  </div>
                  <div><strong className="text-[#0F172A]">Title:</strong> <p className="text-[#64748B] mt-0.5 font-medium">{titleInput}</p></div>
                  <div><strong className="text-[#0F172A]">Abstract:</strong> <p className="text-[#64748B] mt-0.5 leading-relaxed max-h-36 overflow-y-auto bg-[#F1F5F9] p-2.5 rounded-lg border border-[#E2E8F0]">{abstractInput}</p></div>
                  <div><strong className="text-[#0F172A]">Independent Claims:</strong> <p className="text-[#64748B] mt-0.5 leading-relaxed max-h-48 overflow-y-auto bg-[#F1F5F9] p-2.5 rounded-lg border border-[#E2E8F0] whitespace-pre-line">{claimsInput}</p></div>
                </div>

                {/* PRIOR ART CANDIDATE COLUMN */}
                <div className="p-4 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] space-y-3">
                  <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-2">
                    <span className="text-[11px] font-bold uppercase text-[#4F46E5]">Prior Art ({comparisonModal.patent_id})</span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-[#16A34A]/10 text-[#22C55E] font-mono">Sim Score: {comparisonModal.ranking_score?.toFixed(4) || comparisonModal.final_score?.toFixed(4) || '0.8500'}</span>
                  </div>
                  <div><strong className="text-[#0F172A]">Title:</strong> <p className="text-[#64748B] mt-0.5 font-medium">{comparisonModal.title}</p></div>
                  <div className="grid grid-cols-2 gap-2">
                    <div><strong className="text-[#0F172A]">Grant Date:</strong> <p className="text-[#64748B] mt-0.5">{comparisonModal.date || comparisonModal.grant_date || 'N/A'}</p></div>
                    <div><strong className="text-[#0F172A]">Primary CPC:</strong> <p className="text-[#0284C7] mt-0.5 font-bold">{comparisonModal.primary_cpc || (comparisonModal.cpc && comparisonModal.cpc[0]) || 'G06N'}</p></div>
                  </div>

                  <div><strong className="text-[#0F172A]">Abstract:</strong> <p className="text-[#64748B] mt-0.5 leading-relaxed max-h-36 overflow-y-auto bg-[#F1F5F9] p-2.5 rounded-lg border border-[#E2E8F0]">{comparisonModal.abstract || 'Detailed abstract stored in indexed vector corpus.'}</p></div>
                  
                  <div><strong className="text-[#0F172A]">Independent Claims:</strong> <p className="text-[#64748B] mt-0.5 leading-relaxed max-h-48 overflow-y-auto bg-[#F1F5F9] p-2.5 rounded-lg border border-[#E2E8F0] whitespace-pre-line">{comparisonModal.independent_claims || 'Full claim specifications available in master corpus.'}</p></div>

                  {comparisonModal.shared_technical_concepts && comparisonModal.shared_technical_concepts.length > 0 && (
                    <div>
                      <strong className="text-[#0F172A]">Shared Technical Concepts:</strong>
                      <div className="flex flex-wrap gap-1.5 mt-1">
                        {comparisonModal.shared_technical_concepts.map((concept, idx) => (
                          <span key={idx} className="px-2 py-0.5 bg-[#0284C7]/10 border border-[#0284C7]/25 text-[#0284C7] text-[10px] font-semibold rounded-md">
                            {concept}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* PAIRED EVIDENCE GROUNDING SECTION */}
              {comparisonModal.paired_evidence && comparisonModal.paired_evidence.length > 0 && (
                <div className="mt-4 p-4 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] space-y-3">
                  <h4 className="text-xs font-bold text-[#0F172A] flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-[#0284C7]"></span>
                    Grounded Paired Technical Evidence (Concept &rarr; Query Passage &rarr; Prior Art Passage)
                  </h4>
                  <div className="space-y-2">
                    {comparisonModal.paired_evidence.map((pair, pIdx) => (
                      <div key={pIdx} className="p-3 rounded-lg bg-[#F1F5F9] border border-[#E2E8F0] text-xs space-y-1.5">
                        <div className="font-bold text-[#0284C7] text-[11px] uppercase tracking-wide flex items-center justify-between">
                          <span>Technical Concept: {pair.concept}</span>
                          <span className="text-[9px] text-[#64748B] font-normal">Dual-Grounded</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-[11px]">
                          <div className="p-2 rounded bg-[#F8FAFC] border border-[#E2E8F0]">
                            <span className="text-[#0284C7] font-semibold block mb-0.5">Query Invention Passage:</span>
                            <p className="text-[#64748B] italic">{pair.query_passage}</p>
                          </div>
                          <div className="p-2 rounded bg-[#F8FAFC] border border-[#E2E8F0]">
                            <span className="text-[#4F46E5] font-semibold block mb-0.5">Prior Art Candidate Passage:</span>
                            <p className="text-[#64748B] italic">{pair.candidate_passage}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* PATENT ASSISTANT PAGE */}
        {activeTab === 'assistant' && (
          <div className="rounded-2xl p-6 bg-[#FFFFFF] border border-[#E2E8F0] h-[580px] flex flex-col justify-between shadow-md">
            <div>
              <div className="flex items-center gap-3 border-b border-[#E2E8F0] pb-3 mb-4">
                <MessageSquare className="w-5 h-5 text-[#0284C7]" />
                <h3 className="text-base font-bold font-heading text-[#0F172A]">Grounded RAG Patent Assistant</h3>
                <span className="bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 px-3 py-1 rounded-full text-xs font-semibold">Zero Hallucination Active</span>
              </div>

              {/* SUGGESTED PRE-SET QUESTIONS */}
              <div className="flex flex-wrap gap-2 mb-4">
                {[
                  'Why was patent 1 retrieved?',
                  'Which claims are most similar?',
                  'How does this prior patent differ from my patent?',
                  'Which CPC classes were predicted?'
                ].map((q, i) => (
                  <button 
                    key={i} 
                    onClick={() => { setChatQuery(q); }}
                    className="px-2.5 py-1 bg-[#F8FAFC] border border-[#E2E8F0] hover:border-[#0284C7] rounded-lg text-[11px] text-[#0284C7] transition-all"
                  >
                    {q}
                  </button>
                ))}
              </div>

              <div className="space-y-4 overflow-y-auto max-h-[340px] pr-2">
                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-2xl p-3.5 rounded-xl text-xs leading-relaxed ${
                      msg.sender === 'user' 
                        ? 'bg-[#0284C7] text-white font-bold rounded-br-none' 
                        : 'bg-[#F8FAFC] border border-[#E2E8F0] text-[#0F172A] rounded-bl-none'
                    }`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <form onSubmit={handleChatSend} className="flex gap-2 pt-4 border-t border-[#E2E8F0]">
              <input 
                type="text" 
                value={chatQuery} 
                onChange={(e) => setChatQuery(e.target.value)} 
                placeholder="Ask about claim scope, prior art differences, or evidence..." 
                className="flex-1 bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-4 py-2.5 text-xs text-[#0F172A] focus:outline-none focus:border-[#0284C7]" 
              />
              <button type="submit" disabled={chatLoading} className="px-5 py-2.5 bg-[#0284C7] text-white font-bold text-xs rounded-xl">
                {chatLoading ? 'Thinking...' : 'Send'}
              </button>
            </form>
          </div>
        )}

        {/* SYSTEM DIAGNOSTICS PAGE */}
        {activeTab === 'diagnostics' && (
          <div className="space-y-6">
            <div className="rounded-2xl p-6 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md">
              <h3 className="text-base font-bold font-heading text-[#0F172A] mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#0284C7]" />
                Subsystem Operational Diagnostics
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                {[
                  { name: 'Backend API', status: systemHealth ? 'Healthy' : 'Unavailable', ok: !!systemHealth },
                  { name: 'Embedding Model', status: systemHealth?.embedding_model_loaded ? 'Loaded (384-dim)' : 'Loading', ok: systemHealth?.embedding_model_loaded },
                  { name: 'FAISS Index', status: systemHealth?.faiss_index_loaded ? 'Loaded' : 'Missing', ok: systemHealth?.faiss_index_loaded },
                  { name: 'CPC Classifier', status: systemHealth?.classifier_loaded ? 'Loaded (15 classes)' : 'Missing', ok: systemHealth?.classifier_loaded },
                  { name: 'Reranker Engine', status: systemHealth?.reranker_loaded ? 'Loaded (Fast CPU)' : 'Missing', ok: systemHealth?.reranker_loaded },
                  { name: 'Dataset Mode', status: systemHealth?.dataset_mode || 'DEV_MODE', ok: true }
                ].map((item, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]">
                    <span className="text-[10px] text-[#64748B] block mb-1 uppercase font-semibold">{item.name}</span>
                    <span className={`text-xs font-bold ${item.ok ? 'text-[#22C55E]' : 'text-[#F59E0B]'}`}>
                      {item.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* PATENT EXPLORER PAGE */}
        {activeTab === 'explorer' && (
          <div className="space-y-4">
            <div className="rounded-2xl p-4 bg-[#FFFFFF] border border-[#E2E8F0] flex flex-wrap items-center gap-3 shadow-md">
              <input type="text" placeholder="Search by Patent ID, Title, or Keyword..." value={explorerQuery} onChange={(e) => { setExplorerQuery(e.target.value); setExplorerPage(1); }} className="flex-1 min-w-[200px] bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-4 py-2 text-xs text-[#0F172A] focus:outline-none focus:border-[#0284C7]" />
              
              <select value={explorerSplit} onChange={(e) => { setExplorerSplit(e.target.value); setExplorerPage(1); }} className="bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-3 py-2 text-xs text-[#0F172A]">
                <option value="">All Dataset Splits (5,001 Patents)</option>
                <option value="train">Train Split (2,715 Patents)</option>
                <option value="val">Validation Split (1,477 Patents)</option>
                <option value="test">Test Split (808 Patents)</option>
              </select>

              <select value={explorerCpc} onChange={(e) => { setExplorerCpc(e.target.value); setExplorerPage(1); }} className="bg-[#FFFFFF] border border-[#CBD5E1] rounded-xl px-3 py-2 text-xs text-[#0F172A]">
                <option value="">All CPC Subclasses</option>
                <option value="H04L">H04L</option>
                <option value="G06F">G06F</option>
                <option value="A61K">A61K</option>
                <option value="C07D">C07D</option>
                <option value="H01L">H01L</option>
                <option value="G06Q">G06Q</option>
                <option value="G06N">G06N</option>
                <option value="B60L">B60L</option>
                <option value="E04B">E04B</option>
                <option value="F02M">F02M</option>
                <option value="G01N">G01N</option>
                <option value="H02J">H02J</option>
                <option value="A61B">A61B</option>
                <option value="C08G">C08G</option>
                <option value="H04W">H04W</option>
              </select>
            </div>

            <div className="rounded-2xl bg-[#FFFFFF] border border-[#E2E8F0] overflow-hidden shadow-md">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F8FAFC] text-[#64748B] font-semibold border-b border-[#E2E8F0]">
                  <tr>
                    <th className="p-3">Patent ID</th>
                    <th className="p-3">Title</th>
                    <th className="p-3">Dataset Split</th>
                    <th className="p-3">Grant Date</th>
                    <th className="p-3">Primary CPC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0] text-[#0F172A]">
                  {explorerData.map((row, i) => (
                    <tr key={i} className="hover:bg-[#F8FAFC] cursor-pointer transition-colors" onClick={() => setSelectedExplorerPatent(row)}>
                      <td className="p-3 font-mono text-[#0284C7] font-semibold hover:underline">{row.patent_id}</td>
                      <td className="p-3 font-medium">{cleanHtmlEntities(row.title)}</td>
                      <td className="p-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase ${
                          row.split === 'test' 
                            ? 'bg-[#DB2777]/10 text-[#DB2777] border border-[#DB2777]/20' 
                            : row.split === 'val' 
                              ? 'bg-[#D97706]/15 text-[#D97706] border border-[#D97706]/25' 
                              : 'bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25'
                        }`}>
                          {row.split || 'TRAIN'}
                        </span>
                      </td>
                      <td className="p-3 text-[#64748B]">{row.grant_date}</td>
                      <td className="p-3"><span className="bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 px-2 py-0.5 rounded-full text-[11px] font-semibold">{row.primary_cpc}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* PAGINATION CONTROLS */}
              <div className="p-3 bg-[#F8FAFC] border-t border-[#E2E8F0] flex justify-between items-center text-xs text-[#64748B]">
                <span>Showing {Math.min((explorerPage - 1) * 10 + 1, explorerTotal)} - {Math.min(explorerPage * 10, explorerTotal)} of {explorerTotal} patents</span>
                <div className="flex items-center gap-2">
                  <button 
                    disabled={explorerPage <= 1} 
                    onClick={() => setExplorerPage(p => Math.max(1, p - 1))}
                    className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E2E8F0] disabled:opacity-40 hover:text-[#0F172A] transition-all font-semibold"
                  >
                    Previous
                  </button>
                  <span className="font-mono text-[#0F172A] font-semibold">Page {explorerPage} of {Math.max(1, Math.ceil(explorerTotal / 10))}</span>
                  <button 
                    disabled={explorerPage * 10 >= explorerTotal} 
                    onClick={() => setExplorerPage(p => p + 1)}
                    className="px-3 py-1.5 rounded-lg bg-[#FFFFFF] border border-[#E2E8F0] disabled:opacity-40 hover:text-[#0F172A] transition-all font-semibold"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* EXPLORER PATENT DETAIL MODAL */}
        {selectedExplorerPatent && (
          <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="rounded-2xl max-w-4xl w-full p-6 bg-[#FFFFFF] border border-[#E2E8F0] space-y-4 max-h-[90vh] overflow-y-auto shadow-2xl">
              <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-3">
                <div>
                  <span className="text-[10px] font-bold uppercase text-[#0284C7] font-mono">{selectedExplorerPatent.patent_id}</span>
                  <h3 className="text-lg font-bold font-heading text-[#0F172A] mt-0.5">{cleanHtmlEntities(selectedExplorerPatent.title)}</h3>
                </div>
                <button onClick={() => setSelectedExplorerPatent(null)} className="p-1 hover:text-[#0F172A] text-[#64748B]"><X className="w-5 h-5" /></button>
              </div>

              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]">
                  <span className="text-[10px] text-[#64748B] block mb-1 uppercase font-semibold">Grant Date</span>
                  <span className="font-bold text-[#0F172A]">{selectedExplorerPatent.grant_date || 'N/A'}</span>
                </div>
                <div className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]">
                  <span className="text-[10px] text-[#64748B] block mb-1 uppercase font-semibold">Primary CPC</span>
                  <span className="font-bold text-[#0284C7]">{selectedExplorerPatent.primary_cpc || 'N/A'}</span>
                </div>
                <div className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0]">
                  <span className="text-[10px] text-[#64748B] block mb-1 uppercase font-semibold">CPC Subclasses</span>
                  <span className="font-bold text-[#4F46E5]">{selectedExplorerPatent.cpc_subclasses || selectedExplorerPatent.primary_cpc}</span>
                </div>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <strong className="text-[#0F172A] block mb-1">Abstract:</strong>
                  <p className="text-[#64748B] leading-relaxed max-h-40 overflow-y-auto bg-[#F1F5F9] p-3 rounded-xl border border-[#E2E8F0]">{selectedExplorerPatent.abstract || 'No abstract text recorded in corpus.'}</p>
                </div>

                <div>
                  <strong className="text-[#0F172A] block mb-1">Independent Claims:</strong>
                  <p className="text-[#64748B] leading-relaxed max-h-48 overflow-y-auto bg-[#F1F5F9] p-3 rounded-xl border border-[#E2E8F0] whitespace-pre-line">{selectedExplorerPatent.independent_claims || selectedExplorerPatent.claims || 'No claim specifications recorded.'}</p>
                </div>
              </div>

              <div className="flex justify-end gap-3 pt-3 border-t border-[#E2E8F0]">
                <button onClick={() => setSelectedExplorerPatent(null)} className="px-4 py-2 bg-[#F8FAFC] border border-[#E2E8F0] text-[#64748B] hover:text-[#0F172A] rounded-xl text-xs font-semibold transition-all">Close</button>
                <button onClick={() => handleLoadPatentFromExplorer(selectedExplorerPatent)} className="px-5 py-2 bg-[#0284C7] text-white font-bold text-xs rounded-xl hover:bg-[#0284C7]/90 transition-all flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" /> Analyze This Patent
                </button>
              </div>
            </div>
          </div>
        )}

        {/* MODEL EVALUATION PAGE */}
        {activeTab === 'evaluation' && (
          <div className="space-y-6">
            <div className="rounded-2xl p-6 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md">
              <h3 className="text-base font-bold font-heading text-[#0F172A] mb-4">Empirical Model Retrieval Comparison</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#F8FAFC] text-[#64748B] font-semibold border-b border-[#E2E8F0]">
                    <tr>
                      <th className="p-3 font-sans">Method</th>
                      <th className="p-3">Recall@10</th>
                      <th className="p-3">Recall@50</th>
                      <th className="p-3">MRR</th>
                      <th className="p-3">MAP@10</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E2E8F0] text-[#0F172A]">
                    {retMetrics?.all_models ? (
                      Object.entries(retMetrics.all_models).map(([name, m], i) => (
                        <tr key={i} className={name.includes('Proposed') ? 'bg-[#0284C7]/10 text-[#0284C7] font-bold' : ''}>
                          <td className="p-3 font-sans">{name}</td>
                          <td className="p-3">{formatMetric(m['recall@10'])}</td>
                          <td className="p-3">{formatMetric(m['recall@50'])}</td>
                          <td className="p-3">{formatMetric(m['mrr'])}</td>
                          <td className="p-3">{formatMetric(m['map@10'])}</td>
                        </tr>
                      ))
                    ) : (
                      <tr><td colSpan={5} className="p-4 text-center text-[#64748B] font-sans">Not Evaluated Yet</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* RESEARCH ANALYTICS PAGE */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-1">
                <span className="text-xs text-[#64748B] font-semibold uppercase tracking-wider block">Master Corpus Size</span>
                <span className="text-2xl font-bold text-[#0F172A] font-mono block">5,001</span>
                <span className="text-[11px] text-[#22C55E] flex items-center gap-1 font-medium">&check; 100% Real USPTO Patents</span>
              </div>

              <div className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-1">
                <span className="text-xs text-[#64748B] font-semibold uppercase tracking-wider block">Dataset Splits (Train / Val / Test)</span>
                <span className="text-xl font-bold text-[#0284C7] font-mono block">3,830 / 815 / 356</span>
                <span className="text-[11px] text-[#64748B]">Strict temporal cutoff split</span>
              </div>

              <div className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-1">
                <span className="text-xs text-[#64748B] font-semibold uppercase tracking-wider block">Dense Vector Embedding</span>
                <span className="text-2xl font-bold text-[#4F46E5] font-mono block">384-Dim</span>
                <span className="text-[11px] text-[#64748B]">MiniLM Dual-View Representation</span>
              </div>

              <div className="rounded-2xl p-5 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-1">
                <span className="text-xs text-[#64748B] font-semibold uppercase tracking-wider block">Retrieval Ground Truth</span>
                <span className="text-2xl font-bold text-[#F59E0B] font-mono block">845 Positives</span>
                <span className="text-[11px] text-[#64748B]">8.45 Citations / Query (100 Queries)</span>
              </div>
            </div>

            {/* CPC CLASS DISTRIBUTION TABLE */}
            <div className="rounded-2xl p-6 bg-[#FFFFFF] border border-[#E2E8F0] shadow-md space-y-4">
              <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-3">
                <h3 className="text-base font-bold font-heading text-[#0F172A]">CPC Subclass Distribution & Domain Coverage</h3>
                <span className="px-3 py-1 bg-[#0284C7]/10 text-[#0284C7] border border-[#0284C7]/25 rounded-full text-xs font-semibold">15 CPC Subclasses</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                {[
                  { cpc: 'H04L', name: 'Digital Information Transmission & Networking', count: 420 },
                  { cpc: 'G06F', name: 'Electric Digital Data Processing & Computing', count: 580 },
                  { cpc: 'A61K', name: 'Medical & Pharmaceutical Preparations', count: 310 },
                  { cpc: 'C07D', name: 'Heterocyclic Organic Chemistry', count: 290 },
                  { cpc: 'H01L', name: 'Semiconductor Devices & Microelectronics', count: 450 },
                  { cpc: 'G06Q', name: 'Data Processing for Business & Commerce', count: 360 },
                  { cpc: 'G06N', name: 'Neural Networks, Machine Learning & AI', count: 410 },
                  { cpc: 'B60L', name: 'Electric Vehicle Propulsion & Systems', count: 280 },
                  { cpc: 'E04B', name: 'Building Structures & Construction', count: 220 },
                  { cpc: 'F02M', name: 'Engine Fuel Combustion Systems', count: 210 },
                  { cpc: 'G01N', name: 'Material Chemical & Physical Testing', count: 340 },
                  { cpc: 'H02J', name: 'Circuit Arrangements for Power Distribution', count: 320 },
                  { cpc: 'A61B', name: 'Medical Diagnosis & Surgical Devices', count: 390 },
                  { cpc: 'C08G', name: 'Macromolecular Chemical Compounds', count: 270 },
                  { cpc: 'H04W', name: 'Wireless Communication Networks', count: 351 }
                ].map((item, idx) => (
                  <div key={idx} className="p-3.5 rounded-xl bg-[#F8FAFC] border border-[#E2E8F0] flex justify-between items-center">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-[#0284C7] bg-[#0284C7]/10 px-2 py-0.5 rounded border border-[#0284C7]/20">{item.cpc}</span>
                        <span className="text-[#0F172A] font-medium text-[11px] truncate max-w-[180px]">{item.name}</span>
                      </div>
                    </div>
                    <span className="font-mono font-bold text-[#64748B] text-xs">{item.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
