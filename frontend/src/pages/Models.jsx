import React, { useState, useEffect } from 'react';
import { Shield, Sparkles, HelpCircle, RefreshCw, BarChart2 } from 'lucide-react';

const Models = ({ apiBaseUrl, setActiveModalImage }) => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [activeConfusionModel, setActiveConfusionModel] = useState('stacking_ensemble');
  const [error, setError] = useState(null);

  const fetchData = async (force = false) => {
    try {
      setLoading(true);
      setError(null);
      
      if (!force) {
        const cachedModels = sessionStorage.getItem('predictiq_models');
        if (cachedModels) {
          setData(JSON.parse(cachedModels));
          setLoading(false);
          return;
        }
      }
      
      const res = await fetch(`${apiBaseUrl}/api/models/metrics`);
      const result = await res.json();
      if (result.success) {
        setData(result.data);
        sessionStorage.setItem('predictiq_models', JSON.stringify(result.data));
      } else {
        setError(result.error || "Failed to fetch model metrics.");
      }
    } catch (err) {
      console.error(err);
      setError("Error connecting to models validation API.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData(false);
  }, [apiBaseUrl]);

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-32 rounded-2xl bg-white/5 border border-white/5" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-[400px] rounded-2xl bg-white/5 border border-white/5" />
          <div className="h-[400px] rounded-2xl bg-white/5 border border-white/5" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] text-center space-y-4">
        <div className="p-4 rounded-full bg-danger/10 border border-danger/20 text-danger">
          <Shield className="w-10 h-10" />
        </div>
        <h3 className="text-lg font-bold text-white">Validation Module Issue</h3>
        <p className="text-textMuted text-sm">{error}</p>
      </div>
    );
  }

  const modelMetrics = data?.metrics || {};
  const recommendedModel = data?.recommended_model || '';
  const recommendationReason = data?.recommendation_reason || '';

  // Extract best values per metric to highlight in green
  const getBestVal = (metric) => {
    let best = -1.0;
    Object.values(modelMetrics).forEach((m) => {
      if (m[metric] > best) best = m[metric];
    });
    return best;
  };

  const bestMetrics = {
    accuracy: getBestVal('accuracy'),
    precision: getBestVal('precision'),
    recall: getBestVal('recall'),
    f1_score: getBestVal('f1_score'),
    roc_auc: getBestVal('roc_auc'),
    pr_auc: getBestVal('pr_auc'),
    mcc: getBestVal('mcc')
  };

  // Maps model IDs to confusion chart query param names
  const confusionMapping = {
    'logistic_regression': 'confusion_lr',
    'random_forest': 'confusion_rf',
    'xgboost': 'confusion_xgb',
    'lightgbm': 'confusion_lgb',
    'stacking_ensemble': 'confusion_stacking'
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Model Validation Arena</h1>
          <p className="text-textMuted text-sm mt-1">
            Comparative analysis and explainability metrics for 5 customer purchase behavior classifiers.
          </p>
        </div>
        <button
          onClick={() => fetchData(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-textMuted hover:text-white bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 transition-all duration-200 self-start sm:self-auto"
          title="Force refresh validation stats cache"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Data</span>
        </button>
      </div>

      {/* Recommended Model Badge */}
      {recommendedModel && (
        <div className="glass-panel p-6 border border-success/30 bg-success/[0.02] flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="flex h-2.5 w-2.5 rounded-full bg-success animate-pulse" />
              <span className="text-xs font-bold text-success uppercase tracking-wider">Top Recommended Classifier</span>
            </div>
            <h3 className="text-xl font-bold text-white uppercase tracking-tight">
              {recommendedModel.replace('_', ' ')}
            </h3>
            <p className="text-xs text-textMuted max-w-2xl leading-relaxed">{recommendationReason}</p>
          </div>
          <div className="flex items-center gap-2 p-3 rounded-xl bg-success/10 border border-success/20 text-success text-sm font-semibold self-start md:self-auto">
            <Sparkles className="w-4 h-4" /> Final Selection
          </div>
        </div>
      )}

      {/* Metrics Table */}
      <div className="glass-panel p-6">
        <h3 className="text-base font-bold text-white">Cross-Model Metrics Comparison</h3>
        <p className="text-textMuted text-xs mt-1">Evaluations performed on the 20% stratified test set. Highlighted values represent top performers.</p>
        
        <div className="overflow-x-auto mt-6 w-full">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-white/10 text-textMuted font-bold">
                <th className="py-3 pr-4">Model Classifier</th>
                 <th className="py-3 text-center">Accuracy</th>
                <th className="py-3 text-center">Precision</th>
                <th className="py-3 text-center">Recall</th>
                <th className="py-3 text-center">F1-Score</th>
                <th className="py-3 text-center">Test ROC-AUC</th>
                <th className="py-3 text-center">PR-AUC</th>
                <th className="py-3 text-center">MCC</th>
                <th className="py-3 text-center pr-2">5-Fold CV (ROC-AUC)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(modelMetrics).map(([name, m]) => {
                const isRecommended = name === recommendedModel;
                return (
                  <tr 
                    key={name} 
                    className={`border-b border-white/5 text-textPrimary hover:bg-white/[0.01] ${
                      isRecommended ? 'bg-white/[0.01]' : ''
                    }`}
                  >
                    <td className="py-3.5 pr-4 font-bold uppercase tracking-tight text-textPrimary">
                      {name.replace('_', ' ')} {isRecommended && <span className="text-secondary text-[10px] lowercase font-medium ml-1.5">(recommended)</span>}
                    </td>
                    <td className={`py-3.5 text-center font-semibold ${m.accuracy === bestMetrics.accuracy ? 'text-success font-bold' : ''}`}>
                      {m.accuracy != null ? (m.accuracy * 100).toFixed(2) : 'N/A'}%
                    </td>
                    <td className={`py-3.5 text-center font-semibold ${m.precision === bestMetrics.precision ? 'text-success font-bold' : ''}`}>
                      {m.precision != null ? (m.precision * 100).toFixed(2) : 'N/A'}%
                    </td>
                    <td className={`py-3.5 text-center font-semibold ${m.recall === bestMetrics.recall ? 'text-success font-bold' : ''}`}>
                      {m.recall != null ? (m.recall * 100).toFixed(2) : 'N/A'}%
                    </td>
                    <td className={`py-3.5 text-center font-semibold ${m.f1_score === bestMetrics.f1_score ? 'text-success font-bold' : ''}`}>
                      {m.f1_score != null ? (m.f1_score * 100).toFixed(2) : 'N/A'}%
                    </td>
                     <td className={`py-3.5 text-center font-semibold font-mono ${m.roc_auc === bestMetrics.roc_auc ? 'text-success font-bold' : ''}`}>
                      {m.roc_auc != null ? m.roc_auc.toFixed(4) : 'N/A'}
                    </td>
                    <td className={`py-3.5 text-center font-semibold font-mono ${m.pr_auc === bestMetrics.pr_auc ? 'text-success font-bold' : ''}`}>
                      {m.pr_auc != null ? m.pr_auc.toFixed(4) : 'N/A'}
                    </td>
                    <td className={`py-3.5 text-center font-semibold font-mono ${m.mcc === bestMetrics.mcc ? 'text-success font-bold' : ''}`}>
                      {m.mcc != null ? m.mcc.toFixed(4) : 'N/A'}
                    </td>
                    <td className="py-3.5 text-center font-semibold font-mono text-textMuted pr-2">
                      {m.cv_roc_auc_mean != null ? m.cv_roc_auc_mean.toFixed(4) : 'N/A'}{' '}
                      {m.cv_roc_auc_std != null ? `± ${m.cv_roc_auc_std.toFixed(3)}` : ''}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison Plots Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC curve */}
        <div className="glass-panel p-6">
          <h3 className="text-base font-bold text-white">Receiver Operating Characteristic (ROC)</h3>
          <p className="text-textMuted text-xs mt-1">Visualizes sensitivity trade-offs across all classification thresholds.</p>
          <div className="flex justify-center p-4 min-h-[300px] mt-4 rounded-xl border border-white/5 bg-white/[0.01]">
            <img 
              src={`${apiBaseUrl}/api/charts/roc?t=${new Date().getTime()}`} 
              alt="ROC Curve overlay" 
              className="max-h-[350px] object-contain rounded-lg cursor-pointer hover:opacity-90 hover:scale-[1.01] transition-all duration-200"
              onClick={() => setActiveModalImage({
                src: `${apiBaseUrl}/api/charts/roc?t=${new Date().getTime()}`,
                label: "Receiver Operating Characteristic (ROC)",
                description: "Visualizes sensitivity trade-offs across all classification thresholds."
              })}
            />
          </div>
        </div>

        {/* PR Curve */}
        <div className="glass-panel p-6">
          <h3 className="text-base font-bold text-white">Precision-Recall Curve Overlay</h3>
          <p className="text-textMuted text-xs mt-1">Key validator for evaluating models under class imbalance.</p>
          <div className="flex justify-center p-4 min-h-[300px] mt-4 rounded-xl border border-white/5 bg-white/[0.01]">
            <img 
              src={`${apiBaseUrl}/api/charts/pr?t=${new Date().getTime()}`} 
              alt="Precision-Recall Curve overlay" 
              className="max-h-[350px] object-contain rounded-lg cursor-pointer hover:opacity-90 hover:scale-[1.01] transition-all duration-200"
              onClick={() => setActiveModalImage({
                src: `${apiBaseUrl}/api/charts/pr?t=${new Date().getTime()}`,
                label: "Precision-Recall Curve Overlay",
                description: "Key validator for evaluating models under class imbalance."
              })}
            />
          </div>
        </div>
      </div>

      {/* Confusion Matrix Viewer & Feature Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Confusion Matrix dropdown viewer */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Confusion Matrices</h3>
              <select
                value={activeConfusionModel}
                onChange={(e) => setActiveConfusionModel(e.target.value)}
                className="bg-darkbg border border-white/10 rounded-lg px-2 py-1 text-xs text-white focus:outline-none focus:border-primary"
              >
                 {Object.keys(modelMetrics).filter(name => name !== 'baseline_majority').map(name => (
                  <option key={name} value={name}>{name.replace('_', ' ').toUpperCase()}</option>
                ))}
              </select>
            </div>
            <p className="text-textMuted text-xs mt-1">Toggle to inspect actual vs predicted breakdowns.</p>
          </div>
          
          <div className="flex-1 flex items-center justify-center p-4 min-h-[220px] mt-4 rounded-xl border border-white/5 bg-white/[0.01]">
            <img 
              src={`${apiBaseUrl}/api/charts/${confusionMapping[activeConfusionModel]}?t=${new Date().getTime()}`} 
              alt={`${activeConfusionModel} confusion matrix`}
              className="max-h-[250px] object-contain rounded-lg cursor-pointer hover:opacity-90 hover:scale-[1.01] transition-all duration-200"
              onClick={() => setActiveModalImage({
                src: `${apiBaseUrl}/api/charts/${confusionMapping[activeConfusionModel]}?t=${new Date().getTime()}`,
                label: `${activeConfusionModel.replace('_', ' ').toUpperCase()} Confusion Matrix`,
                description: "Actual vs predicted classification matrix."
              })}
            />
          </div>
        </div>

        {/* Random Forest Feature Importance */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white">Feature Importances (RF)</h3>
            <p className="text-textMuted text-xs mt-1">Relative feature contribution according to Random Forest Gini impurity.</p>
          </div>
          <div className="flex-1 flex items-center justify-center p-4 min-h-[220px] mt-4 rounded-xl border border-white/5 bg-white/[0.01]">
            <img 
              src={`${apiBaseUrl}/api/charts/rf_importance?t=${new Date().getTime()}`} 
              alt="Random forest feature importance"
              className="max-h-[250px] object-contain rounded-lg cursor-pointer hover:opacity-90 hover:scale-[1.01] transition-all duration-200"
              onClick={() => setActiveModalImage({
                src: `${apiBaseUrl}/api/charts/rf_importance?t=${new Date().getTime()}`,
                label: "Random Forest Feature Importances",
                description: "Relative feature contribution calculated from Gini impurity indices."
              })}
            />
          </div>
        </div>

        {/* SHAP Summary Plot */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white">SHAP Beeswarm Explanations</h3>
            <p className="text-textMuted text-xs mt-1">Global feature importance distributions computed via game-theory (XGBoost).</p>
          </div>
          <div className="flex-1 flex items-center justify-center p-4 min-h-[220px] mt-4 rounded-xl border border-white/5 bg-white/[0.01]">
            <img 
              src={`${apiBaseUrl}/api/charts/shap_summary?t=${new Date().getTime()}`} 
              alt="SHAP summary beeswarm plot"
              className="max-h-[250px] object-contain rounded-lg cursor-pointer hover:opacity-90 hover:scale-[1.01] transition-all duration-200"
              onClick={() => setActiveModalImage({
                src: `${apiBaseUrl}/api/charts/shap_summary?t=${new Date().getTime()}`,
                label: "SHAP Beeswarm Explanations",
                description: "Global feature impact distributions mapping XGBoost input correlations."
              })}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Models;
