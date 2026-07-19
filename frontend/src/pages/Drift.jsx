import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, AlertTriangle, AlertCircle, RefreshCw, 
  Calendar, Play
} from 'lucide-react';

const Drift = ({ apiBaseUrl }) => {
  const [loading, setLoading] = useState(true);
  const [runningCheck, setRunningCheck] = useState(false);
  const [driftData, setDriftData] = useState(null);
  const [error, setError] = useState(null);

  // MLOps Retraining State
  const [retrainState, setRetrainState] = useState({
    status: 'idle', // idle, running, success, failed
    progress: 0,
    error: null,
    started_at: null,
    finished_at: null
  });

  const fetchDriftStatus = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/api/drift/status`);
      const result = await res.json();
      if (result.success) {
        setDriftData(result.data);
        setError(null);
      } else {
        setError(result.error || "Failed to load drift status.");
      }
    } catch (err) {
      console.error(err);
      setError("Error connecting to drift detection API.");
    } finally {
      setLoading(false);
    }
  };

  const fetchRetrainStatus = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/api/models/retrain/status`);
      const result = await res.json();
      if (result.success) {
        setRetrainState(result.data);
      }
    } catch (err) {
      console.error("Error fetching retrain status:", err);
    }
  };
  const formatTimestamp = (isoString) => {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return isoString;
      return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ', ' + 
             date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch (e) {
      return isoString;
    }
  };

  useEffect(() => {
    fetchDriftStatus();
    fetchRetrainStatus();
  }, [apiBaseUrl]);

  // Poll retraining progress in the background if active
  useEffect(() => {
    let interval = null;
    if (retrainState.status === 'running') {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${apiBaseUrl}/api/models/retrain/status`);
          const result = await res.json();
          if (result.success) {
            setRetrainState(result.data);
            if (result.data.status === 'success') {
              clearInterval(interval);
              fetchDriftStatus(); // Reload drift report since baseline re-calibrated
            } else if (result.data.status === 'failed') {
              clearInterval(interval);
            }
          }
        } catch (err) {
          console.error("Error polling retrain progress:", err);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [retrainState.status, apiBaseUrl]);

  const triggerRetrain = async () => {
    try {
      setRetrainState(prev => ({ ...prev, status: 'running', progress: 5, error: null }));
      const res = await fetch(`${apiBaseUrl}/api/models/retrain`, {
        method: 'POST'
      });
      const result = await res.json();
      if (result.success) {
        setRetrainState(result.data);
      } else {
        setRetrainState(prev => ({ ...prev, status: 'failed', error: result.error || "Failed to trigger retraining." }));
      }
    } catch (err) {
      console.error("Error triggering retraining:", err);
      setRetrainState(prev => ({ ...prev, status: 'failed', error: "Network error triggering retrain." }));
    }
  };

  // Triggers recalculation on python pipeline
  const runDriftCheck = async () => {
    try {
      setRunningCheck(true);
      setError(null);
      
      const res = await fetch(`${apiBaseUrl}/api/drift/check`, {
        method: 'POST'
      });
      const result = await res.json();
      
      if (result.success) {
        // Fetch new status
        await fetchDriftStatus();
      } else {
        setError(result.error || "Recalculation pipeline failed.");
      }
    } catch (err) {
      console.error(err);
      setError("Connection error. Could not run recalculation pipeline.");
    } finally {
      setRunningCheck(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-32 rounded-2xl bg-white/5 border border-white/5" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-44 rounded-2xl bg-white/5 border border-white/5" />
          ))}
        </div>
      </div>
    );
  }

  const overallStatus = driftData?.overall_status || 'Stable';
  const timestamp = driftData?.timestamp || '';
  const metrics = driftData?.metrics || [];

  const getStatusBadge = (status) => {
    if (status === 'Stable') {
      return {
        text: 'Stable',
        class: 'bg-success/10 text-success border-success/20',
        cardBorder: 'border-white/10 hover:border-success/30 bg-white/[0.01]',
        icon: ShieldCheck,
        colorHex: '#10B981'
      };
    } else if (status === 'Monitor') {
      return {
        text: 'Monitor',
        class: 'bg-warning/10 text-warning border-warning/20',
        cardBorder: 'border-warning/30 bg-warning/[0.01]',
        icon: AlertTriangle,
        colorHex: '#F59E0B'
      };
    } else {
      return {
        text: 'Retrain Alert',
        class: 'bg-danger/10 text-danger border-danger/20',
        cardBorder: 'border-danger/30 bg-danger/[0.01]',
        icon: AlertCircle,
        colorHex: '#EF4444'
      };
    }
  };

  const overallBadge = getStatusBadge(overallStatus);
  const OverallIcon = overallBadge.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Data Drift Monitor</h1>
          <p className="text-textMuted text-sm mt-1">
            Tracks distribution shifts between base training features and live production batches using Population Stability Index (PSI).
          </p>
        </div>
        
        <button
          onClick={runDriftCheck}
          disabled={runningCheck}
          className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-primary hover:bg-primary/80 disabled:opacity-50 transition-all shadow-lg shadow-primary/10 self-start md:self-auto"
        >
          {runningCheck ? (
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <RefreshCw className="w-3.5 h-3.5" />
          )}
          Run Drift Check
        </button>
      </div>

      {/* Error Panel */}
      {error && (
        <div className="p-4 rounded-xl border border-danger/25 bg-danger/[0.02] flex items-center gap-3 text-xs text-danger font-medium">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Overall Health Card */}
      <div className="glass-panel p-6 border border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="flex items-start gap-4">
          <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 shrink-0 flex items-center justify-center" style={{ color: overallBadge.colorHex }}>
            <OverallIcon className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <span className="text-xs font-bold text-textMuted uppercase tracking-wider">System Stability Status</span>
            <div className="flex items-center gap-3">
              <h3 className="text-xl font-bold text-white tracking-tight">
                {overallStatus === 'Stable' ? 'Data Distribution Consistent' : overallStatus === 'Monitor' ? 'Minor Drift Detected' : 'Model Retraining Required'}
              </h3>
              <span className={`px-2.5 py-0.5 rounded-full border text-[10px] font-bold ${overallBadge.class}`}>
                {overallBadge.text}
              </span>
            </div>
            <p className="text-xs text-textMuted max-w-xl leading-relaxed">
              {overallStatus === 'Stable' 
                ? 'The current live production data matches the training distribution closely. No actions required.' 
                : overallStatus === 'Monitor' 
                ? 'Some features are exhibiting minor deviations. Keep monitoring prediction logs.' 
                : 'Warning: Major features have drifted significantly. Classification models are likely degraded and require immediate retraining.'}
            </p>
          </div>
        </div>

        {/* Timestamp */}
        <div className="flex items-center gap-2 text-xs text-textMuted border-t md:border-t-0 md:border-l border-white/5 pt-4 md:pt-0 md:pl-6 shrink-0">
          <Calendar className="w-4 h-4" />
          <div>
            <span className="block font-medium">Last Checked</span>
            <span className="font-bold text-white font-mono">{timestamp ? timestamp.substring(0, 19).replace('T', ' ') : 'Never'}</span>
          </div>
        </div>
      </div>

      {/* Retraining Console Card */}
      <div className="glass-panel p-6 border border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="flex items-start gap-4">
          <div className="p-4 rounded-2xl bg-secondary/10 border border-secondary/20 shrink-0 flex items-center justify-center text-secondary">
            <RefreshCw className={`w-8 h-8 ${retrainState.status === 'running' ? 'animate-spin' : ''}`} />
          </div>
          <div className="space-y-1 flex-1">
            <span className="text-xs font-bold text-textMuted uppercase tracking-wider">Automated MLOps Retraining Console</span>
            <h3 className="text-lg font-bold text-white tracking-tight">
              {retrainState.status === 'running' 
                ? `Retraining pipeline in progress... (${retrainState.progress}%)` 
                : retrainState.status === 'success'
                ? 'Retraining Pipeline Completed'
                : retrainState.status === 'failed'
                ? 'Retraining Pipeline Failed'
                : 'Trigger Full Model Retraining (Option 1)'}
            </h3>
            <p className="text-xs text-textMuted max-w-xl leading-relaxed">
              {retrainState.status === 'running' 
                ? 'Executing dataset reconstruction, re-segmentation, and RandomizedSearchCV hyperparameter search in background. Reloads models dynamically upon completion.'
                : retrainState.status === 'success'
                ? `Pipeline successfully completed at ${formatTimestamp(retrainState.finished_at)}. Models reloaded dynamically in RAM.`
                : retrainState.status === 'failed'
                ? `Error during execution: ${retrainState.error || 'Unknown error'}`
                : 'Cleans transaction logs, calculates 9D RFM metrics, recalculates K-Means clusters, retrains Stacking Ensemble and base classifiers, and updates drift baselines.'}
            </p>
            
            {/* Progress bar */}
            {retrainState.status === 'running' && (
              <div className="mt-3 w-full sm:max-w-md space-y-1">
                <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full bg-secondary transition-all duration-500"
                    style={{ width: `${retrainState.progress}%` }}
                  />
                </div>
                <div className="flex justify-between text-[9px] font-bold text-textMuted uppercase">
                  <span>Start</span>
                  <span>Progress: {retrainState.progress}%</span>
                  <span>Generalizing</span>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <button
          onClick={triggerRetrain}
          disabled={retrainState.status === 'running'}
          className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-secondary hover:bg-secondary/80 disabled:opacity-50 transition-all shadow-lg shadow-secondary/15 shrink-0 self-start md:self-auto"
        >
          {retrainState.status === 'running' ? (
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Play className="w-3.5 h-3.5 fill-white" />
          )}
          {retrainState.status === 'running' ? 'Retraining...' : 'Run Pipeline Retrain'}
        </button>
      </div>

      {/* PSI Gauges Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((item) => {
          const config = getStatusBadge(item.status);
          const Icon = config.icon;
          const percentage = Math.min((item.psi_value / 0.5) * 100, 100); // Scale 0 to 0.5 for visual bar

          return (
            <div key={item.feature} className={`glass-panel p-6 border transition-all duration-300 ${config.cardBorder}`}>
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="text-sm font-bold text-white uppercase tracking-tight">{item.feature}</h4>
                  <span className="text-[10px] text-textMuted font-medium uppercase">Population Stability Index</span>
                </div>
                <span className={`px-2 py-0.5 rounded-full border text-[9px] font-bold ${config.class}`}>
                  {config.text}
                </span>
              </div>

              {/* Display Value */}
              <div className="mt-6 flex items-baseline gap-2">
                <span className="text-3xl font-extrabold text-white tracking-tight font-mono">{item.psi_value.toFixed(4)}</span>
                <span className="text-xs text-textMuted font-medium">PSI score</span>
              </div>

              {/* Progress bar representing PSI severity */}
              <div className="mt-4 space-y-1">
                <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <div 
                    className="h-full rounded-full transition-all duration-500"
                    style={{ 
                      width: `${percentage}%`,
                      backgroundColor: config.colorHex 
                    }}
                  />
                </div>
                <div className="flex justify-between text-[9px] text-textMuted font-medium">
                  <span>Stable (0.0)</span>
                  <span>Monitor (0.1)</span>
                  <span>Retrain (0.25+)</span>
                </div>
              </div>

              {/* Mean Comparisons */}
              <div className="mt-6 pt-4 border-t border-white/5 grid grid-cols-2 gap-4 text-xs">
                <div className="p-2.5 rounded-lg border border-white/5 bg-white/[0.01]">
                  <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider block">Training Mean</span>
                  <span className="text-sm font-bold text-white font-mono mt-0.5 block">{item.training_mean.toFixed(2)}</span>
                </div>
                <div className="p-2.5 rounded-lg border border-white/5 bg-white/[0.01]">
                  <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider block">Production Mean</span>
                  <span className="text-sm font-bold text-white font-mono mt-0.5 block" style={{ color: config.colorHex }}>{item.production_mean.toFixed(2)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>



    </div>
  );
};

export default Drift;
