import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Sliders, FileSpreadsheet, Play, Download, Upload, 
  CheckCircle, AlertCircle, RefreshCw, BarChart2, ShieldAlert, Sparkles, Database, History, Eye
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, Cell, CartesianGrid } from 'recharts';

const Predict = ({ apiBaseUrl }) => {
  const [activeTab, setActiveTab] = useState('single');
  
  // ==========================================
  // DATASET INGESTION STATES
  // ==========================================
  const [ingestFile, setIngestFile] = useState(null);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestSuccess, setIngestSuccess] = useState(null);
  const [ingestError, setIngestError] = useState(null);

  const handleIngestFileDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.xls'))) {
      setIngestFile(file);
      setIngestError(null);
      setIngestSuccess(null);
    } else {
      setIngestError("Please drop a valid .csv, .xlsx, or .xls file.");
    }
  };

  const handleIngestFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.xls'))) {
      setIngestFile(file);
      setIngestError(null);
      setIngestSuccess(null);
    } else {
      setIngestError("Please select a valid .csv, .xlsx, or .xls file.");
    }
  };

  const runDatasetIngestion = async () => {
    if (!ingestFile) return;
    setIngestLoading(true);
    setIngestProgress(10);
    setIngestSuccess(null);
    setIngestError(null);

    const formData = new FormData();
    formData.append("file", ingestFile);

    const interval = setInterval(() => {
      setIngestProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 15;
      });
    }, 200);

    try {
      const res = await fetch(`${apiBaseUrl}/api/dataset/upload`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Dataset ingestion failed.");
      }

      const data = await res.json();
      clearInterval(interval);
      setIngestProgress(100);
      setIngestSuccess(data);
      setIngestLoading(false);
      setIngestFile(null);
    } catch (err) {
      console.error(err);
      clearInterval(interval);
      setIngestError(err.message || "Failed to ingest dataset into the pipeline.");
      setIngestLoading(false);
      setIngestProgress(0);
    }
  };

  const downloadDataset = async (type) => {
    try {
      const res = await fetch(`${apiBaseUrl}/api/dataset/download/${type}`);
      if (!res.ok) throw new Error(`Download of ${type} dataset failed.`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', type === 'raw' ? 'OnlineRetail.xlsx' : 'cleaned_retail.csv');
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (err) {
      console.error(err);
      alert(err.message);
    }
  };

  // ==========================================
  // UNIFIED PREDICTION HISTORY LOGS STATES
  // ==========================================
  const [historyLogs, setHistoryLogs] = useState([]);
  const [historyCount, setHistoryCount] = useState(0);
  const [historyOffset, setHistoryOffset] = useState(0);
  const [historyLimit] = useState(15);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  const fetchHistoryLogs = async (offsetVal = 0) => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch(`${apiBaseUrl}/api/predict/history/all?limit=${historyLimit}&offset=${offsetVal}`);
      if (!res.ok) throw new Error("Failed to load prediction history logs.");
      const resJson = await res.json();
      if (resJson.success) {
        setHistoryLogs(resJson.data.records);
        setHistoryCount(resJson.data.total);
        setHistoryOffset(offsetVal);
      } else {
        throw new Error(resJson.error || "Failed to load prediction history logs.");
      }
    } catch (err) {
      console.error(err);
      setHistoryError(err.message || "An unknown error occurred.");
    } finally {
      setHistoryLoading(false);
    }
  };

  const exportHistoryLogs = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/api/predict/history/export`);
      if (!res.ok) throw new Error("Failed to export prediction history logs.");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `predictiq_inference_history_${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (err) {
      console.error(err);
      alert(err.message);
    }
  };

  const formatTimestamp = (ts) => {
    if (!ts) return "N/A";
    try {
      const date = new Date(ts);
      if (isNaN(date.getTime())) return ts;
      return date.toLocaleString();
    } catch (e) {
      return ts;
    }
  };

  // ==========================================
  // CUSTOMER SHAP EXPLAINABILITY STATES
  // ==========================================
  const [shapCustomerId, setShapCustomerId] = useState('');
  const [shapLoading, setShapLoading] = useState(false);
  const [shapData, setShapData] = useState(null);
  const [shapError, setShapError] = useState(null);

  const fetchShapData = async (custId) => {
    if (!custId) return;
    setShapLoading(true);
    setShapError(null);
    setShapData(null);
    try {
      const res = await fetch(`${apiBaseUrl}/api/predict/shap/${encodeURIComponent(custId)}`);
      const data = await res.json();
      if (data.success) {
        setShapData(data.data);
      } else {
        setShapError(data.error || `Customer ID ${custId} was not found in registries.`);
      }
    } catch (err) {
      console.error(err);
      setShapError("Network error. Make sure FastAPI server is running.");
    } finally {
      setShapLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'history') {
      fetchHistoryLogs(0);
    }
  }, [activeTab]);

  // ==========================================
  // SINGLE PREDICTION STATES
  // ==========================================
  const [singleInputs, setSingleInputs] = useState({
    Recency: 30,
    Frequency: 5,
    Monetary: 250,
    AvgOrderValue: 50,
    UniqueProducts: 15,
    ReturnRate: 0.05,
    CustomerLifetimeDays: 180,
    PurchaseFrequencyMonthly: 1.5,
    AvgQuantityPerOrder: 12,
    CustomerID: '17850'
  });
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResults, setSingleResults] = useState(null);
  const [singleError, setSingleError] = useState(null);
  const [showSingleRawShap, setShowSingleRawShap] = useState(false);

  // ==========================================
  // BATCH PREDICTION STATES
  // ==========================================
  const downloadTemplate = () => {
    const headers = [
      'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts', 'ReturnRate',
      'CustomerLifetimeDays', 'PurchaseFrequencyMonthly', 'AvgQuantityPerOrder'
    ];
    const sampleRows = [
      [30, 5, 250.0, 50.0, 15, 0.05, 180, 1.5, 12],
      [120, 2, 80.0, 40.0, 5, 0.0, 90, 0.6, 8],
      [5, 25, 1200.0, 48.0, 65, 0.12, 320, 3.1, 24]
    ];
    const csvContent = [
      headers.join(','),
      ...sampleRows.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "predictiq_batch_template.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const [batchFile, setBatchFile] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchResults, setBatchResults] = useState(null);
  const [batchCsvData, setBatchCsvData] = useState(null);
  const [batchError, setBatchError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 8;

  // Handle single input sliders change
  const handleInputChange = (field, value) => {
    setSingleInputs(prev => {
      const updated = { ...prev, [field]: parseFloat(value) };
      
      // Auto-calculate AvgOrderValue = Monetary / Frequency unless manually changed
      if (field === 'Monetary' || field === 'Frequency') {
        const freq = field === 'Frequency' ? parseFloat(value) : prev.Frequency;
        const mon = field === 'Monetary' ? parseFloat(value) : prev.Monetary;
        if (freq > 0) {
          updated.AvgOrderValue = parseFloat((mon / freq).toFixed(2));
        }
      }
      return updated;
    });
  };

  // Submit Single Prediction
  const runSinglePrediction = async (e) => {
    e.preventDefault();
    setSingleLoading(true);
    setSingleResults(null);
    setSingleError(null);
    setShowSingleRawShap(false);

    try {
      const res = await fetch(`${apiBaseUrl}/api/predict/single`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(singleInputs)
      });
      const data = await res.json();
      
      if (data.success) {
        setSingleResults(data.data);
      } else {
        setSingleError(data.error || "Inference failed.");
      }
    } catch (err) {
      console.error(err);
      setSingleError("Network error. Make sure FastAPI server is running.");
    } finally {
      setSingleLoading(false);
    }
  };

  // Handle Drag & Drop File
  const handleFileDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      setBatchFile(file);
      setBatchError(null);
    } else {
      setBatchError("Please drop a valid .csv file.");
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.csv')) {
      setBatchFile(file);
      setBatchError(null);
    } else {
      setBatchError("Please select a valid .csv file.");
    }
  };

  // Run Batch Prediction
  const runBatchPrediction = async () => {
    if (!batchFile) return;
    setBatchLoading(true);
    setBatchProgress(10);
    setBatchResults(null);
    setBatchError(null);

    const formData = new FormData();
    formData.append("file", batchFile);

    // Simulate progress bar increments
    const interval = setInterval(() => {
      setBatchProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 15;
      });
    }, 200);

    try {
      const res = await fetch(`${apiBaseUrl}/api/predict/batch`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Batch conversion failed.");
      }
      
      const csvBlob = await res.blob();
      setBatchCsvData(csvBlob);
      
      // Parse CSV client-side to render inside the dashboard table
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const rows = text.split('\n').filter(r => r.trim().length > 0);
        const headers = rows[0].split(',');
        
        const parsedData = rows.slice(1).map(rowStr => {
          const cells = rowStr.split(',');
          const obj = {};
          headers.forEach((h, idx) => {
            obj[h.trim()] = cells[idx] ? cells[idx].trim() : '';
          });
          return obj;
        });
        
        setBatchResults(parsedData);
        setBatchProgress(100);
        setBatchLoading(false);
        clearInterval(interval);
      };
      reader.readAsText(csvBlob);

    } catch (err) {
      console.error(err);
      clearInterval(interval);
      setBatchError(err.message || "Failed running batch parser.");
      setBatchLoading(false);
      setBatchProgress(0);
    }
  };

  // Download Batch CSV Result
  const downloadBatchResults = () => {
    if (!batchCsvData) return;
    const url = window.URL.createObjectURL(batchCsvData);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `batch_predictions_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
  };

  // Segment metadata
  const segmentActions = {
    "Champions": "VIP Reward Action: Offer early access, zero-shipping loyalty programs, and high-value brand advocacy items.",
    "Loyal Customers": "Growth Strategy: Cross-sell premium categories, trigger monthly email updates, and run referral perks.",
    "At-Risk": "Re-activation Campaign: Send a high-discount win-back offer ('We miss you, take 30% off') within 48 hours.",
    "Lost Customers": "Retention Save: Automated low-cost reactivation. Avoid high spend on acquisition channels."
  };

  const segmentBadges = {
    "Champions": "bg-success/10 text-success border-success/20",
    "Loyal Customers": "bg-primary/10 text-primary border-primary/20",
    "At-Risk": "bg-warning/10 text-warning border-warning/20",
    "Lost Customers": "bg-danger/10 text-danger border-danger/20"
  };

  // Pagination helper
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentBatchItems = batchResults ? batchResults.slice(indexOfFirstItem, indexOfLastItem) : [];
  const totalPages = batchResults ? Math.ceil(batchResults.length / itemsPerPage) : 0;

  const chartData = shapData ? Object.entries(shapData.contributions).map(([name, value]) => ({
    name,
    value: parseFloat(value)
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Customer Purchase Inference Console</h1>
        <p className="text-textMuted text-sm mt-1">
          Perform real-time customer behavior predictions using our trained supervised models.
        </p>
      </div>

      {/* Tabs Menu */}
      <div className="flex gap-4 border-b border-white/10 pb-px">
        <button
          onClick={() => setActiveTab('single')}
          className={`flex items-center gap-2 pb-3 text-sm font-semibold transition-all border-b-2 px-2 ${
            activeTab === 'single'
              ? 'border-primary text-white'
              : 'border-transparent text-textMuted hover:text-white'
          }`}
        >
          <Sliders className="w-4 h-4" /> Single Prediction
        </button>
        <button
          onClick={() => setActiveTab('batch')}
          className={`flex items-center gap-2 pb-3 text-sm font-semibold transition-all border-b-2 px-2 ${
            activeTab === 'batch'
              ? 'border-primary text-white'
              : 'border-transparent text-textMuted hover:text-white'
          }`}
        >
          <FileSpreadsheet className="w-4 h-4" /> Batch Prediction
        </button>
        <button
          onClick={() => setActiveTab('update_dataset')}
          className={`flex items-center gap-2 pb-3 text-sm font-semibold transition-all border-b-2 px-2 ${
            activeTab === 'update_dataset'
              ? 'border-primary text-white'
              : 'border-transparent text-textMuted hover:text-white'
          }`}
        >
          <Database className="w-4 h-4" /> Ingest Retail Data
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center gap-2 pb-3 text-sm font-semibold transition-all border-b-2 px-2 ${
            activeTab === 'history'
              ? 'border-primary text-white'
              : 'border-transparent text-textMuted hover:text-white'
          }`}
        >
          <History className="w-4 h-4" /> Prediction History Logs
        </button>
        <button
          onClick={() => setActiveTab('shap')}
          className={`flex items-center gap-2 pb-3 text-sm font-semibold transition-all border-b-2 px-2 ${
            activeTab === 'shap'
              ? 'border-primary text-white'
              : 'border-transparent text-textMuted hover:text-white'
          }`}
        >
          <BarChart2 className="w-4 h-4" /> Explainability Dashboard (SHAP)
        </button>
      </div>

      {/* Tab Contents */}
      <div>
        {activeTab === 'single' ? (
          // ==========================================
          // SINGLE PREDICTION VIEW
          // ==========================================
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Input Form Column (col-span-2) */}
            <div className="glass-panel p-6 lg:col-span-2 space-y-6">
              <div>
                <h3 className="text-base font-bold text-white">Customer RFM Metrics</h3>
                <p className="text-textMuted text-xs mt-1">Adjust sliders or inputs to configure a client profile.</p>
              </div>

              <form onSubmit={runSinglePrediction} className="space-y-5">
                {/* Customer ID */}
                <div className="space-y-1.5">
                  <label className="text-xs text-textMuted font-semibold flex items-center justify-between">
                    <span>Customer ID (Optional)</span>
                    <span className="text-[10px] text-textMuted/70 font-normal">Logged for audit explainability</span>
                  </label>
                  <input 
                    type="text"
                    placeholder="e.g. 17850 or Admin Manual"
                    value={singleInputs.CustomerID || ''}
                    onChange={(e) => setSingleInputs(prev => ({ ...prev, CustomerID: e.target.value }))}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                  />
                </div>

                {/* Recency */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <label className="text-textPrimary font-semibold">Recency (Days since Last Order)</label>
                    <span className="text-secondary font-bold">{singleInputs.Recency} days</span>
                  </div>
                  <input 
                    type="range" min="1" max="365" step="1"
                    value={singleInputs.Recency}
                    onChange={(e) => handleInputChange('Recency', e.target.value)}
                    className="w-full h-1.5 bg-white/5 rounded-lg appearance-none cursor-pointer accent-secondary"
                  />
                </div>

                {/* Frequency */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <label className="text-textPrimary font-semibold">Frequency (Unique Orders)</label>
                    <span className="text-secondary font-bold">{singleInputs.Frequency} purchases</span>
                  </div>
                  <input 
                    type="range" min="1" max="50" step="1"
                    value={singleInputs.Frequency}
                    onChange={(e) => handleInputChange('Frequency', e.target.value)}
                    className="w-full h-1.5 bg-white/5 rounded-lg appearance-none cursor-pointer accent-secondary"
                  />
                </div>

                {/* Monetary */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <label className="text-textPrimary font-semibold">Monetary Spend (£)</label>
                    <span className="text-secondary font-bold">£{singleInputs.Monetary}</span>
                  </div>
                  <input 
                    type="range" min="1" max="5000" step="10"
                    value={singleInputs.Monetary}
                    onChange={(e) => handleInputChange('Monetary', e.target.value)}
                    className="w-full h-1.5 bg-white/5 rounded-lg appearance-none cursor-pointer accent-secondary"
                  />
                </div>

                {/* Grid for other features */}
                <div className="grid grid-cols-2 gap-4">
                  {/* Unique Products */}
                  <div className="space-y-1.5">
                    <label className="text-xs text-textMuted font-semibold">Unique Products</label>
                    <input 
                      type="number" min="1" max="200"
                      value={singleInputs.UniqueProducts}
                      onChange={(e) => handleInputChange('UniqueProducts', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                    />
                  </div>

                  {/* Return Rate */}
                  <div className="space-y-1.5">
                    <label className="text-xs text-textMuted font-semibold">Return Rate (0 to 1)</label>
                    <input 
                      type="number" min="0" max="1" step="0.01"
                      value={singleInputs.ReturnRate}
                      onChange={(e) => handleInputChange('ReturnRate', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                    />
                  </div>
                </div>

                {/* Grid for new engineered features */}
                <div className="grid grid-cols-3 gap-3">
                  {/* Customer Lifetime Days */}
                  <div className="space-y-1.5">
                    <label className="text-xs text-textMuted font-semibold">Lifetime (Days)</label>
                    <input 
                      type="number" min="0" max="1000"
                      value={singleInputs.CustomerLifetimeDays}
                      onChange={(e) => handleInputChange('CustomerLifetimeDays', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-2.5 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                    />
                  </div>

                  {/* Purchase Frequency Monthly */}
                  <div className="space-y-1.5">
                    <label className="text-xs text-textMuted font-semibold">Freq / Month</label>
                    <input 
                      type="number" min="0" max="100" step="0.1"
                      value={singleInputs.PurchaseFrequencyMonthly}
                      onChange={(e) => handleInputChange('PurchaseFrequencyMonthly', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-2.5 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                    />
                  </div>

                  {/* Avg Quantity Per Order */}
                  <div className="space-y-1.5">
                    <label className="text-xs text-textMuted font-semibold">Qty / Order</label>
                    <input 
                      type="number" min="1" max="1000"
                      value={singleInputs.AvgQuantityPerOrder}
                      onChange={(e) => handleInputChange('AvgQuantityPerOrder', e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-2.5 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                    />
                  </div>
                </div>

                {/* Average Order Value (Auto Calculated) */}
                <div className="p-3.5 rounded-xl border border-white/5 bg-white/[0.01] flex justify-between items-center text-xs">
                  <span className="text-textMuted font-semibold">Calculated Average Order Value:</span>
                  <span className="text-white font-bold font-mono">£{singleInputs.AvgOrderValue}</span>
                </div>

                {/* Submit button */}
                <button
                  type="submit"
                  disabled={singleLoading}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-white accent-gradient accent-gradient-hover text-sm"
                >
                  {singleLoading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" /> Run Purchase Prediction
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* Prediction Results Panel (col-span-3) */}
            <div className="lg:col-span-3 flex flex-col justify-between">
              <AnimatePresence mode="wait">
                {singleResults ? (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                    className="glass-panel p-6 h-full flex flex-col justify-between space-y-6"
                  >
                    {/* Panel Header */}
                    <div className="flex justify-between items-start border-b border-white/5 pb-4">
                      <div>
                        <h4 className="text-base font-bold text-white">Inference Classification Details</h4>
                        <p className="text-textMuted text-xs mt-1">Computed customer segment and multi-classifier outputs.</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full border text-xs font-bold ${segmentBadges[singleResults.assigned_segment] || 'bg-white/5 text-white'}`}>
                        {singleResults.assigned_segment}
                      </span>
                    </div>

                    {/* Classifier Scores */}
                    <div className="space-y-4">
                      <h5 className="text-xs font-bold text-textMuted uppercase tracking-wider">Classifier Consensus (Probabilities)</h5>
                      
                      <div className="space-y-3">
                        {Object.entries(singleResults.predictions).map(([model, res]) => {
                          const percentage = res.probability * 100;
                          const willBuy = res.label === 1;
                          
                          return (
                            <div key={model} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="uppercase font-semibold text-textPrimary">{model.replace('_', ' ')}</span>
                                <span className={`font-bold ${willBuy ? 'text-success' : 'text-danger'}`}>
                                  {willBuy ? 'Will Repurchase' : 'Will Churn'} ({percentage ? percentage.toFixed(0) : '0'}%)
                                </span>
                              </div>
                              <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                                <div 
                                  className={`h-full rounded-full transition-all duration-500 ${willBuy ? 'bg-success' : 'bg-danger'}`}
                                  style={{ width: `${percentage}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* SHAP explanation */}
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <h5 className="text-xs font-bold text-textMuted uppercase tracking-wider flex items-center gap-1">
                          <BarChart2 className="w-3.5 h-3.5 text-secondary" /> SHAP Feature Impact (Top 3)
                        </h5>
                        {singleResults.raw_shap_contributions && (
                          <button
                            type="button"
                            onClick={() => setShowSingleRawShap(!showSingleRawShap)}
                            className="text-[10px] text-secondary hover:underline font-bold transition-all focus:outline-none"
                          >
                            {showSingleRawShap ? 'Hide Full List' : 'View Full List'}
                          </button>
                        )}
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-center">
                        {singleResults.top_shap_features.map((sf, idx) => {
                          const isPos = sf.influence === 'positive';
                          return (
                            <div key={idx} className="p-3 rounded-xl border border-white/5 bg-white/[0.01] space-y-1">
                              <span className="block text-[10px] font-bold text-textMuted uppercase tracking-wider truncate">{sf.feature.replace('_log', '')}</span>
                              <span className={`text-sm font-extrabold block ${isPos ? 'text-success' : 'text-danger'}`}>
                                {isPos ? '+' : ''}{(sf?.shap_value ?? 0).toFixed(2)}
                              </span>
                              <span className="text-[9px] text-textMuted font-medium block">
                                {isPos ? 'Push to Repurchase' : 'Push to Churn'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                      {showSingleRawShap && singleResults.raw_shap_contributions && (
                        <div className="mt-3 p-3 rounded-xl border border-white/5 bg-white/[0.02] space-y-2 max-h-48 overflow-y-auto">
                          <div className="text-[10px] font-bold text-textMuted uppercase tracking-wider mb-2">All Individual Feature Influences</div>
                          {Object.entries(singleResults.raw_shap_contributions)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .map(([feat, val]) => {
                              const isPos = val >= 0;
                              return (
                                <div key={feat} className="flex justify-between items-center text-xs py-1 border-b border-white/5 last:border-b-0">
                                  <span className="text-textMuted font-semibold font-mono">{feat.replace('_log', '')}</span>
                                  <span className={`font-mono font-bold ${isPos ? 'text-success' : 'text-danger'}`}>
                                    {isPos ? '+' : ''}{(parseFloat(val) || 0).toFixed(4)}
                                  </span>
                                </div>
                              );
                            })}
                        </div>
                      )}
                    </div>

                    {/* Action Text */}
                    <div className="p-4 rounded-xl border border-secondary/20 bg-secondary/[0.02] flex items-start gap-3">
                      <Sparkles className="w-5 h-5 text-secondary shrink-0 mt-0.5" />
                      <div>
                        <h5 className="text-xs font-bold text-white uppercase tracking-wider">Marketing Advisor Action</h5>
                        <p className="text-textPrimary text-xs mt-1.5 leading-relaxed font-semibold">
                          {segmentActions[singleResults.assigned_segment] || "Profile initialized. Deploy campaigns."}
                        </p>
                      </div>
                    </div>

                    {singleInputs.CustomerID && singleInputs.CustomerID !== 'Admin Manual' && (
                      <button
                        onClick={() => {
                          setShapCustomerId(singleInputs.CustomerID);
                          fetchShapData(singleInputs.CustomerID);
                          setActiveTab('shap');
                        }}
                        className="w-full flex items-center justify-center gap-1.5 py-2.5 bg-secondary/15 hover:bg-secondary/25 border border-secondary/20 text-secondary rounded-xl text-xs font-semibold transition-all mt-2"
                      >
                        <BarChart2 className="w-3.5 h-3.5" /> Explain this prediction in SHAP Dashboard
                      </button>
                    )}

                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="glass-panel p-6 h-full flex flex-col items-center justify-center text-center space-y-4 text-textMuted py-24"
                  >
                    <Sliders className="w-12 h-12 text-white/10" />
                    <div>
                      <h4 className="text-white font-bold text-sm">Prediction Pending</h4>
                      <p className="text-xs max-w-xs mt-1 leading-relaxed">
                        Adjust features on the left and run the inference tool to visualize behavior forecasts.
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        ) : activeTab === 'batch' ? (
          // ==========================================
          // BATCH PREDICTION VIEW
          // ==========================================
          <div className="space-y-6">
            <div className="glass-panel p-6 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-white">Batch Customer Predictor</h3>
                  <p className="text-textMuted text-xs mt-1">Upload a CSV containing customer profiles to run batch inference using the Stacking Classifier.</p>
                </div>
                <button
                  onClick={downloadTemplate}
                  type="button"
                  className="px-3.5 py-2 rounded-xl border border-white/10 hover:border-secondary hover:text-secondary text-xs font-semibold text-textMuted bg-white/[0.02] hover:bg-white/[0.05] transition-all flex items-center gap-1.5 shrink-0 self-start sm:self-center"
                >
                  <Download className="w-3.5 h-3.5" /> Download CSV Template
                </button>
              </div>

              {/* Drag and Drop Zone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
                className="flex flex-col items-center justify-center border-2 border-dashed border-white/15 hover:border-primary/40 rounded-2xl p-10 bg-white/[0.01] hover:bg-white/[0.02] cursor-pointer transition-all"
              >
                <input 
                  type="file" accept=".csv" id="batchCsvInput"
                  onChange={handleFileSelect}
                  className="hidden" 
                />
                <label htmlFor="batchCsvInput" className="flex flex-col items-center gap-3 cursor-pointer">
                  <Upload className="w-10 h-10 text-textMuted" />
                  <span className="text-sm font-semibold text-white">
                    {batchFile ? batchFile.name : "Drag & Drop CSV file here, or click to browse"}
                  </span>
                  <span className="text-[10px] text-textMuted text-center">
                    CSV columns must include: Recency, Frequency, Monetary, AvgOrderValue, UniqueProducts, ReturnRate, CustomerLifetimeDays, PurchaseFrequencyMonthly, AvgQuantityPerOrder
                  </span>
                </label>
              </div>

              {/* Progress bar and Run action */}
              {batchFile && (
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.01]">
                  <div className="flex-1 space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="font-semibold text-white">Uploading & Analyzing CSV...</span>
                      <span className="font-bold text-secondary">{batchProgress}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-secondary transition-all duration-300"
                        style={{ width: `${batchProgress}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={runBatchPrediction}
                      disabled={batchLoading}
                      className="px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-primary hover:bg-primary/80 transition-all flex items-center gap-1.5"
                    >
                      {batchLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-white" />} Run Batch
                    </button>
                    {batchResults && (
                      <button
                        onClick={downloadBatchResults}
                        className="px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-success hover:bg-success/80 transition-all flex items-center gap-1.5"
                      >
                        <Download className="w-3.5 h-3.5" /> Download Results
                      </button>
                    )}
                  </div>
                </div>
              )}

              {batchError && (
                <div className="p-4 rounded-xl border border-danger/25 bg-danger/[0.02] flex items-center gap-3 text-xs text-danger font-medium">
                  <AlertCircle className="w-5 h-5 shrink-0" />
                  <span>{batchError}</span>
                </div>
              )}
            </div>

            {/* Batch Results Table */}
            {batchResults && (
              <div className="glass-panel p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-base font-bold text-white">Batch Analysis Records</h3>
                    <p className="text-textMuted text-xs mt-1">Processed predictions and assigned segments.</p>
                  </div>
                  <span className="text-xs text-textMuted font-bold uppercase tracking-wider">
                    Total Records: {batchResults.length}
                  </span>
                </div>

                <div className="overflow-x-auto w-full">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-white/10 text-textMuted font-bold">
                        <th className="py-2.5">Row ID</th>
                        <th className="py-2.5">Recency</th>
                        <th className="py-2.5">Frequency</th>
                        <th className="py-2.5">Monetary</th>
                        <th className="py-2.5">Avg Order Val</th>
                        <th className="py-2.5">Return Rate</th>
                        <th className="py-2.5">Assigned Segment</th>
                        <th className="py-2.5">Prediction</th>
                        <th className="py-2.5 text-right">Probability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentBatchItems.map((item, idx) => {
                        const buyLabel = item.Predicted_Purchase_Label === '1' ? 'Repurchase' : 'Churn';
                        const buyColor = item.Predicted_Purchase_Label === '1' ? 'bg-success/10 text-success border-success/20' : 'bg-danger/10 text-danger border-danger/20';
                        
                        return (
                          <tr key={idx} className="border-b border-white/5 text-textPrimary hover:bg-white/[0.01]">
                            <td className="py-3 font-semibold text-textMuted">#{indexOfFirstItem + idx + 1}</td>
                            <td className="py-3">{item.Recency}d</td>
                            <td className="py-3">{item.Frequency}x</td>
                            <td className="py-3">£{(parseFloat(item.Monetary) || 0).toFixed(0)}</td>
                            <td className="py-3">£{(parseFloat(item.AvgOrderValue) || 0).toFixed(0)}</td>
                            <td className="py-3">{((parseFloat(item.ReturnRate) || 0) * 100).toFixed(0)}%</td>
                            <td className="py-3">
                              <span className={`px-2 py-0.5 rounded-full border text-[10px] font-semibold ${segmentBadges[item.Predicted_Segment] || 'bg-white/5 text-white'}`}>
                                {item.Predicted_Segment}
                              </span>
                            </td>
                            <td className="py-3">
                              <span className={`px-2 py-0.5 rounded-md border text-[10px] font-bold ${buyColor}`}>
                                {buyLabel}
                              </span>
                            </td>
                            <td className="py-3 text-right font-mono font-bold text-secondary">
                              {((parseFloat(item.Purchase_Probability) || 0) * 100).toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Pagination Controls */}
                {totalPages > 1 && (
                  <div className="flex justify-between items-center pt-4 border-t border-white/5">
                    <button
                      onClick={() => setCurrentPage(p => Math.max(p - 1, 1))}
                      disabled={currentPage === 1}
                      className="px-3 py-1.5 rounded-lg border border-white/10 text-xs text-white hover:bg-white/5 disabled:opacity-30 transition-all"
                    >
                      Previous
                    </button>
                    <span className="text-xs text-textMuted font-bold">
                      Page {currentPage} of {totalPages}
                    </span>
                    <button
                      onClick={() => setCurrentPage(p => Math.min(p + 1, totalPages))}
                      disabled={currentPage === totalPages}
                      className="px-3 py-1.5 rounded-lg border border-white/10 text-xs text-white hover:bg-white/5 disabled:opacity-30 transition-all"
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : activeTab === 'update_dataset' ? (
          // ==========================================
          // DATASET INGEST PREVIEW
          // ==========================================
          <div className="space-y-6">
            <div className="glass-panel p-6 space-y-6">
              <div>
                <h3 className="text-base font-bold text-white">Ingest Raw Transactions</h3>
                <p className="text-textMuted text-xs mt-1">
                  Upload a CSV or Excel (.xlsx/.xls) file of raw retail transactions. Valid rows are appended to the main UCI retail model database, immediately updating the cleaned cache. The backend data science pipeline and drift models will automatically reconfigure themselves.
                </p>
              </div>

              {/* Drag and Drop Zone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleIngestFileDrop}
                className="flex flex-col items-center justify-center border-2 border-dashed border-white/15 hover:border-primary/45 rounded-2xl p-10 bg-white/[0.01] hover:bg-white/[0.02] cursor-pointer transition-all"
              >
                <input 
                  type="file" accept=".csv,.xlsx,.xls" id="datasetUploadInput"
                  onChange={handleIngestFileSelect}
                  className="hidden" 
                />
                <label htmlFor="datasetUploadInput" className="flex flex-col items-center gap-3 cursor-pointer select-none">
                  <Upload className="w-10 h-10 text-textMuted" />
                  <span className="text-sm font-semibold text-white">
                    {ingestFile ? ingestFile.name : "Drag & Drop CSV/Excel file here, or click to browse"}
                  </span>
                  <span className="text-[10px] text-textMuted text-center max-w-lg leading-relaxed">
                    Required fields (case-insensitive): <strong>CustomerID</strong>, <strong>InvoiceNo</strong>, <strong>Quantity</strong>, <strong>UnitPrice</strong>, <strong>InvoiceDate</strong>.<br />
                    Optional fields: StockCode, Description, Country. Supports large spreadsheets.
                  </span>
                </label>
              </div>

              {/* Ingestion progress, error, and actions */}
              {ingestFile && (
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.01] bg-opacity-30">
                  <div className="flex-1 space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="font-semibold text-white">Ingesting & Processing File...</span>
                      <span className="font-bold text-secondary">{ingestProgress}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-secondary transition-all duration-300"
                        style={{ width: `${ingestProgress}%` }}
                      />
                    </div>
                  </div>
                  
                  <button
                    onClick={runDatasetIngestion}
                    disabled={ingestLoading}
                    className="px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-primary hover:bg-primary/80 transition-all flex items-center gap-1.5 whitespace-nowrap self-start md:self-center"
                  >
                    {ingestLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-white" />} Ingest to Pipeline
                  </button>
                </div>
              )}

              {ingestError && (
                <div className="p-4 rounded-xl border border-danger/25 bg-danger/[0.02] flex items-center gap-3 text-xs text-danger font-medium">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{ingestError}</span>
                </div>
              )}

              {ingestSuccess && (
                <div className="p-4 rounded-xl border border-success/25 bg-success/[0.02] flex flex-col gap-2 text-xs text-success font-medium">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-4 h-4 shrink-0 text-success" />
                    <span className="font-bold text-white">{ingestSuccess.message}</span>
                  </div>
                  <p className="text-textMuted text-xs pl-8">
                    Ingested total rows: <strong className="text-white">{ingestSuccess.rows_ingested}</strong>
                  </p>
                </div>
              )}
            </div>

            {/* Dataset Administration & Exports */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass-panel p-5 space-y-4 flex flex-col justify-between">
                <div className="space-y-1.5">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Database className="w-4 h-4 text-primary" /> Master Raw Dataset
                  </h4>
                  <p className="text-xs text-textMuted leading-relaxed">
                    Download the complete base Online Retail transaction-level registry sheet (.xlsx format) to review historical runs or load into local analysis tools.
                  </p>
                </div>
                <button
                  onClick={() => downloadDataset('raw')}
                  className="w-full flex items-center justify-center gap-1.5 py-2.5 border border-white/10 hover:border-primary/50 text-white rounded-xl text-xs font-semibold hover:bg-white/5 transition-all"
                >
                  <Download className="w-3.5 h-3.5" /> Download raw database (.xlsx)
                </button>
              </div>

              <div className="glass-panel p-5 space-y-4 flex flex-col justify-between">
                <div className="space-y-1.5">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-secondary" /> Preprocessed Cleaned Dataset
                  </h4>
                  <p className="text-xs text-textMuted leading-relaxed">
                    Download the curated Uk-filtered, outlier-removed, and parsed dataset containing computed total spend metrics (.csv format) used directly for segment clustering.
                  </p>
                </div>
                <button
                  onClick={() => downloadDataset('cleaned')}
                  className="w-full flex items-center justify-center gap-1.5 py-2.5 border border-white/10 hover:border-secondary/50 text-white rounded-xl text-xs font-semibold hover:bg-white/5 transition-all"
                >
                  <Download className="w-3.5 h-3.5" /> Download cleaned database (.csv)
                </button>
              </div>
            </div>
          </div>
        ) : activeTab === 'history' ? (
          // ==========================================
          // PREDICTION HISTORY LOGS VIEW
          // ==========================================
          <div className="space-y-6">
            <div className="glass-panel p-6 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-white">Inference History Registry</h3>
                  <p className="text-textMuted text-xs mt-1">
                    Complete audit log of all predictions run by the system, including real-time simulator steps and manual single/batch testing.
                  </p>
                </div>
                <button
                  onClick={exportHistoryLogs}
                  className="px-4 py-2.5 rounded-xl text-xs font-semibold text-white bg-primary hover:bg-primary/80 transition-all flex items-center gap-1.5 self-start sm:self-center"
                >
                  <Download className="w-3.5 h-3.5" /> Export Prediction Logs (CSV)
                </button>
              </div>

              {/* Status or errors */}
              {historyError && (
                <div className="p-4 rounded-xl border border-danger/25 bg-danger/[0.02] flex items-center gap-3 text-xs text-danger font-medium">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{historyError}</span>
                </div>
              )}

              {/* Data Table */}
              <div className="overflow-x-auto w-full rounded-xl border border-white/5 bg-white/[0.01]">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-white/5 bg-white/[0.02] text-textMuted uppercase font-bold text-[10px] tracking-wider">
                      <th className="py-3 px-4">Timestamp</th>
                      <th className="py-3 px-4">CustomerID</th>
                      <th className="py-3 px-4 text-right">Recency</th>
                      <th className="py-3 px-4 text-right">Frequency</th>
                      <th className="py-3 px-4 text-right">Monetary</th>
                      <th className="py-3 px-4">Predicted Segment</th>
                      <th className="py-3 px-4 text-center">Repurchase Predict</th>
                      <th className="py-3 px-4 text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historyLoading ? (
                      [...Array(6)].map((_, idx) => (
                        <tr key={idx} className="border-b border-white/5 animate-pulse">
                          <td className="py-3.5 px-4"><div className="h-3 w-28 bg-white/10 rounded" /></td>
                          <td className="py-3.5 px-4"><div className="h-3 w-16 bg-white/10 rounded" /></td>
                          <td className="py-3.5 px-4 text-right"><div className="h-3 w-16 bg-white/10 rounded ml-auto" /></td>
                          <td className="py-3.5 px-4 text-right"><div className="h-3 w-16 bg-white/10 rounded ml-auto" /></td>
                          <td className="py-3.5 px-4 text-right"><div className="h-3 w-16 bg-white/10 rounded ml-auto" /></td>
                          <td className="py-3.5 px-4"><div className="h-5 w-24 bg-white/10 rounded-full" /></td>
                          <td className="py-3.5 px-4"><div className="h-5 w-20 bg-white/10 rounded-full mx-auto" /></td>
                          <td className="py-3.5 px-4"><div className="h-5 w-12 bg-white/10 rounded mx-auto" /></td>
                        </tr>
                      ))
                    ) : historyLogs.length === 0 ? (
                      <tr>
                        <td colSpan="8" className="py-10 text-center text-textMuted text-xs">
                          No logged predictions found. Run some inferences or turn on the live stream to populate logs.
                        </td>
                      </tr>
                    ) : (
                      historyLogs.map((row) => (
                        <tr key={row.id} className="border-b border-white/5 hover:bg-white/[0.01] transition-all text-white">
                          <td className="py-3 px-4 text-textMuted">{formatTimestamp(row.timestamp)}</td>
                          <td className="py-3 px-4 font-semibold">{row.customer_id}</td>
                          <td className="py-3 px-4 text-right text-textMuted">{row.recency}d</td>
                          <td className="py-3 px-4 text-right text-textMuted">{row.frequency}</td>
                          <td className="py-3 px-4 text-right font-medium text-secondary">${(parseFloat(row.monetary) || 0).toFixed(2)}</td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${segmentBadges[row.predicted_segment] || 'bg-white/5 text-white'}`}>
                              {row.predicted_segment}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-center">
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${row.stacking_pred === 1 ? 'bg-success/15 text-success border border-success/20' : 'bg-danger/15 text-danger border border-danger/20'}`}>
                              {row.stacking_pred === 1 ? 'Will Repurchase' : 'Will Churn'}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-center">
                            <button
                              onClick={() => {
                                setShapCustomerId(row.customer_id);
                                fetchShapData(row.customer_id);
                                setActiveTab('shap');
                              }}
                              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-secondary/15 hover:bg-secondary/25 text-secondary border border-secondary/20 text-[10px] font-bold transition-all"
                            >
                              <BarChart2 className="w-3 h-3" />
                              <span>Explain</span>
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination Section */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-textMuted pt-2">
                <div>
                  Showing <span className="font-semibold text-white">{historyLogs.length > 0 ? historyOffset + 1 : 0}</span> to{' '}
                  <span className="font-semibold text-white">
                    {Math.min(historyOffset + historyLimit, historyCount)}
                  </span>{' '}
                  of <span className="font-semibold text-white">{historyCount}</span> stored inferences
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => fetchHistoryLogs(Math.max(0, historyOffset - historyLimit))}
                    disabled={historyOffset === 0 || historyLoading}
                    className="px-3.5 py-1.5 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] disabled:opacity-40 disabled:hover:bg-white/[0.02] text-white transition-all font-semibold"
                  >
                    Previous
                  </button>
                  <button
                    disabled={historyOffset + historyLimit >= historyCount || historyLoading}
                    onClick={() => fetchHistoryLogs(historyOffset + historyLimit)}
                    className="px-3.5 py-1.5 rounded-lg border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] disabled:opacity-40 disabled:hover:bg-white/[0.02] text-white transition-all font-semibold"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          // ==========================================
          // ACTIVE SHAP EXPLAINABILITY VIEW
          // ==========================================
          <div className="space-y-6">
            <div className="glass-panel p-6 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-secondary" /> Interactive SHAP Explainability Dashboard
                  </h3>
                  <p className="text-textMuted text-xs mt-1">
                    Visualize real-time feature contributions using raw SHAP values to explain individual churn/repurchase forecasts.
                  </p>
                </div>
              </div>

              {/* Customer ID Search Form */}
              <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] space-y-4">
                <div className="flex flex-col sm:flex-row gap-4 items-end">
                  <div className="flex-1 space-y-1.5">
                    <label className="text-xs text-textMuted font-semibold">Enter Customer ID</label>
                    <input
                      type="text"
                      placeholder="e.g. 17850, 13047..."
                      value={shapCustomerId}
                      onChange={(e) => setShapCustomerId(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-secondary"
                    />
                  </div>
                  <button
                    onClick={() => fetchShapData(shapCustomerId)}
                    disabled={shapLoading || !shapCustomerId}
                    className="px-5 py-2 rounded-xl text-xs font-semibold text-white bg-primary hover:bg-primary/80 disabled:opacity-50 disabled:hover:bg-primary transition-all flex items-center justify-center gap-1.5 h-[34px]"
                  >
                    {shapLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-white" />}
                    <span>Get Explanation</span>
                  </button>
                </div>

                <div className="flex flex-wrap items-center gap-2 text-xs pt-1">
                  <span className="text-textMuted">Sample Customer Registries:</span>
                  {['17850', '13047', '12583', '17841', '12347'].map((id) => (
                    <button
                      key={id}
                      onClick={() => {
                        setShapCustomerId(id);
                        fetchShapData(id);
                      }}
                      className="px-2.5 py-1 rounded bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 text-white font-mono text-[10px] transition-all"
                    >
                      {id}
                    </button>
                  ))}
                </div>
              </div>

              {/* Results Display */}
              {shapLoading && (
                <div className="py-24 flex flex-col items-center justify-center space-y-4 animate-pulse">
                  <RefreshCw className="w-8 h-8 text-primary animate-spin" />
                  <span className="text-xs text-textMuted font-semibold">Interpreting model behavior via SHAP trees...</span>
                </div>
              )}

              {shapError && (
                <div className="p-4 rounded-xl border border-danger/25 bg-danger/[0.02] flex items-center gap-3 text-xs text-danger font-semibold">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{shapError}</span>
                </div>
              )}

              {shapData && (
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                  {/* Chart contribution column */}
                  <div className="lg:col-span-3 glass-panel p-5 border border-white/5 bg-white/[0.01] flex flex-col justify-between">
                    <div>
                      <h4 className="text-xs uppercase font-bold text-textMuted tracking-wider">SHAP Feature Influence Balance</h4>
                      <p className="text-[10px] text-textMuted mt-0.5">Directional impact (probability space) absolute sorted.</p>
                    </div>

                    <div className="h-96 w-full mt-4 flex items-center justify-center">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={chartData}
                          margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" horizontal={false} />
                          <XAxis type="number" stroke="#94A3B8" fontSize={9} tickLine={false} />
                          <YAxis dataKey="name" type="category" stroke="#94A3B8" fontSize={9} tickLine={false} width={130} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#020617', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            labelStyle={{ color: '#F1F5F9' }}
                            formatter={(value) => [`${value >= 0 ? '+' : ''}${((value || 0) * 100).toFixed(2)}%`, 'Repurchase Likelihood Impact']}
                          />
                          <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
                          <Bar dataKey="value" strokeWidth={0} radius={[0, 4, 4, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#10B981' : '#EF4444'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Summary Details column */}
                  <div className="lg:col-span-2 space-y-6">
                    <div className="glass-panel p-5 border border-white/5 bg-white/[0.01] space-y-4">
                      <div>
                        <h4 className="text-xs uppercase font-bold text-textMuted tracking-wider">Forecast Verdict</h4>
                        <p className="text-[10px] text-textMuted mt-0.5 font-medium">Derived from final Stacking/XGBoost ensemble.</p>
                      </div>

                      <div className="flex gap-4 items-center">
                        <div className="relative flex items-center justify-center w-20 h-20 shrink-0">
                          {/* Radial border indicator */}
                          <svg className="absolute w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                            <path
                              className="text-white/5"
                              strokeWidth="2.5"
                              stroke="currentColor"
                              fill="none"
                              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            />
                            <path
                              className={shapData.prediction >= 0.5 ? "text-success" : "text-danger"}
                              strokeDasharray={`${((shapData?.prediction ?? 0) * 100).toFixed(0)}, 100`}
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              stroke="currentColor"
                              fill="none"
                              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            />
                          </svg>
                          <span className="text-sm font-extrabold text-white">{((shapData?.prediction ?? 0) * 100).toFixed(0)}%</span>
                        </div>
                        <div className="space-y-1">
                          <span className="text-xs text-textMuted font-semibold">Repurchase Probability</span>
                          <span className={`block text-sm font-extrabold ${shapData.prediction >= 0.5 ? 'text-success' : 'text-danger'}`}>
                            {shapData.prediction >= 0.5 ? 'WILL REPURCHASE (Champions/Loyal)' : 'WILL CHURN (At-Risk/Lost)'}
                          </span>
                        </div>
                      </div>

                      <div className="pt-2 border-t border-white/5 space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-textMuted font-medium">Population Base Value:</span>
                          <span className="text-white font-bold font-mono">{((shapData?.base_value ?? 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-textMuted font-medium">Customer Prediction Value:</span>
                          <span className="text-white font-bold font-mono">{((shapData?.prediction ?? 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-textMuted font-medium">Behavioral Deviation:</span>
                          <span className={`font-bold font-mono ${((shapData?.prediction ?? 0) - (shapData?.base_value ?? 0)) >= 0 ? 'text-success' : 'text-danger'}`}>
                            {((shapData?.prediction ?? 0) - (shapData?.base_value ?? 0)) >= 0 ? '+' : ''}{(((shapData?.prediction ?? 0) - (shapData?.base_value ?? 0)) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Customer attributes */}
                    <div className="glass-panel p-5 border border-white/5 bg-white/[0.01] space-y-4">
                      <div>
                        <h4 className="text-xs uppercase font-bold text-textMuted tracking-wider">Raw Attributes Summary</h4>
                        <p className="text-[10px] text-textMuted mt-0.5">Recorded behavior inputs parsed before scaling.</p>
                      </div>

                      <div className="grid grid-cols-2 gap-3.5 text-xs text-textMuted">
                        <div className="p-2 rounded bg-white/[0.01] border border-white/5">
                          <span className="block font-bold text-white text-sm">{(shapData?.features?.Recency ?? 0).toFixed(0)}d</span>
                          Recency
                        </div>
                        <div className="p-2 rounded bg-white/[0.01] border border-white/5">
                          <span className="block font-bold text-white text-sm">{(shapData?.features?.Frequency ?? 0).toFixed(0)}x</span>
                          Frequency
                        </div>
                        <div className="p-2 rounded bg-white/[0.01] border border-white/5">
                          <span className="block font-bold text-white text-sm">£{(shapData?.features?.Monetary ?? 0).toFixed(0)}</span>
                          Monetary Spend
                        </div>
                        <div className="p-2 rounded bg-white/[0.01] border border-white/5">
                          <span className="block font-bold text-white text-sm">£{(shapData?.features?.AvgOrderValue ?? 0).toFixed(0)}</span>
                          Avg Order Value
                        </div>
                        <div className="p-2 rounded bg-white/[0.01] border border-white/5">
                          <span className="block font-bold text-white text-sm">{((shapData?.features?.ReturnRate ?? 0) * 100).toFixed(0)}%</span>
                          Return Rate
                        </div>
                        <div className="p-2 rounded bg-white/[0.01] border border-white/5">
                          <span className="block font-bold text-white text-sm">{(shapData?.features?.CustomerLifetimeDays ?? 0).toFixed(0)}d</span>
                          Lifetime
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {!shapLoading && !shapData && !shapError && (
                <div className="py-24 border border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center text-center space-y-3 text-textMuted">
                  <BarChart2 className="w-12 h-12 text-white/5" />
                  <div>
                    <h4 className="text-white font-bold text-sm">No SHAP Profile Queried</h4>
                    <p className="text-xs max-w-xs mt-1 leading-relaxed">
                      Search for an active customer ID or choose a preset registry to compute explanations.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predict;
