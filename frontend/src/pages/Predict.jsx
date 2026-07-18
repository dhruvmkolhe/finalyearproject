import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Sliders, FileSpreadsheet, Play, Download, Upload, 
  CheckCircle, AlertCircle, RefreshCw, BarChart2, ShieldAlert, Sparkles
} from 'lucide-react';

const Predict = ({ apiBaseUrl }) => {
  const [activeTab, setActiveTab] = useState('single');
  
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
    AvgQuantityPerOrder: 12
  });
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResults, setSingleResults] = useState(null);
  const [singleError, setSingleError] = useState(null);

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
                                  {willBuy ? 'Will Repurchase' : 'Will Churn'} ({percentage.toFixed(0)}%)
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
                      <h5 className="text-xs font-bold text-textMuted uppercase tracking-wider flex items-center gap-1">
                        <BarChart2 className="w-3.5 h-3.5 text-secondary" /> SHAP Feature Impact (Top 3)
                      </h5>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-center">
                        {singleResults.top_shap_features.map((sf, idx) => {
                          const isPos = sf.influence === 'positive';
                          return (
                            <div key={idx} className="p-3 rounded-xl border border-white/5 bg-white/[0.01] space-y-1">
                              <span className="block text-[10px] font-bold text-textMuted uppercase tracking-wider truncate">{sf.feature.replace('_log', '')}</span>
                              <span className={`text-sm font-extrabold block ${isPos ? 'text-success' : 'text-danger'}`}>
                                {isPos ? '+' : ''}{sf.shap_value.toFixed(2)}
                              </span>
                              <span className="text-[9px] text-textMuted font-medium block">
                                {isPos ? 'Push to Repurchase' : 'Push to Churn'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
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
        ) : (
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
                            <td className="py-3">£{parseFloat(item.Monetary).toFixed(0)}</td>
                            <td className="py-3">£{parseFloat(item.AvgOrderValue).toFixed(0)}</td>
                            <td className="py-3">{(parseFloat(item.ReturnRate) * 100).toFixed(0)}%</td>
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
                              {(parseFloat(item.Purchase_Probability) * 100).toFixed(1)}%
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
        )}
      </div>
    </div>
  );
};

export default Predict;
