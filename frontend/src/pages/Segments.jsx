import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { 
  Users, TrendingUp, AlertTriangle, Trash2, 
  HelpCircle, BarChart3, Image as ImageIcon, Sparkles, RefreshCw
} from 'lucide-react';

const Segments = ({ apiBaseUrl, setActiveModalImage }) => {
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [activeSegment, setActiveSegment] = useState('Champions');
  const [error, setError] = useState(null);

  const fetchData = async (force = false) => {
    try {
      setLoading(true);
      setError(null);

      if (!force) {
        const cachedOverview = sessionStorage.getItem('predictiq_segments_overview');
        const cachedCustomers = sessionStorage.getItem('predictiq_segments_customers');
        if (cachedOverview && cachedCustomers) {
          const overviewData = JSON.parse(cachedOverview);
          setOverview(overviewData);
          setCustomers(JSON.parse(cachedCustomers));
          
          const firstSeg = overviewData.distribution[0]?.segment || 'Champions';
          setActiveSegment(firstSeg);
          setLoading(false);
          return;
        }
      }
      
      // Fetch segment overview
      const ovRes = await fetch(`${apiBaseUrl}/api/segments/overview`);
      const ovData = await ovRes.json();
      
      // Fetch raw customer list for histogram calculations
      const custRes = await fetch(`${apiBaseUrl}/api/segments/customers`);
      const custData = await custRes.json();
      
      if (ovData.success && custData.success) {
        setOverview(ovData.data);
        setCustomers(custData.data);
        
        sessionStorage.setItem('predictiq_segments_overview', JSON.stringify(ovData.data));
        sessionStorage.setItem('predictiq_segments_customers', JSON.stringify(custData.data));
        
        // Select default active segment (usually the first one, or Champions)
        const firstSeg = ovData.data.distribution[0]?.segment || 'Champions';
        setActiveSegment(firstSeg);
      } else {
        setError("Failed to load customer segments details.");
      }
    } catch (err) {
      console.error(err);
      setError("Error connecting to segment APIs.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData(false);
  }, [apiBaseUrl]);

  // Client-side histogram calculator — uses quantile-based edges for skewed data
  const getHistogramData = (segment, field, numBins = 10) => {
    if (!customers || customers.length === 0) return [];

    // Filter customers of the active segment
    const segmentCusts = customers.filter(c => c.Segment === segment);
    const segmentVals = segmentCusts.map(c => c[field]).filter(v => v != null);
    if (segmentVals.length === 0) return [];

    // Sort values to compute quantile edges
    const sorted = [...segmentVals].sort((a, b) => a - b);
    const n = sorted.length;

    // Build unique quantile edges (percentile boundaries)
    const edgesSet = new Set();
    for (let i = 0; i <= numBins; i++) {
      const idx = Math.min(Math.floor((i / numBins) * n), n - 1);
      edgesSet.add(sorted[idx]);
    }
    // Always include the very last value
    edgesSet.add(sorted[n - 1]);
    const edges = [...edgesSet].sort((a, b) => a - b);

    // If there are very few unique values, fall back to simple value-count bins
    if (edges.length <= 2) {
      const uniqueVals = [...new Set(sorted)];
      return uniqueVals.map(v => ({
        rangeName: field === 'Monetary' ? `£${v.toFixed(0)}` : `${v.toFixed(0)}`,
        binStart: v,
        binEnd: v,
        count: segmentVals.filter(sv => sv === v).length
      }));
    }

    // Build bins from quantile edges
    const bins = [];
    for (let i = 0; i < edges.length - 1; i++) {
      const start = edges[i];
      const end = edges[i + 1];
      // Skip zero-width duplicate edges
      if (start === end && i < edges.length - 2) continue;
      bins.push({
        rangeName: field === 'Monetary' ? `£${start.toFixed(0)}` : `${start.toFixed(0)}`,
        binStart: start,
        binEnd: end,
        count: 0
      });
    }
    if (bins.length === 0) return [];

    // Populate bins
    segmentVals.forEach(val => {
      // Find the right bin (last bin is inclusive on both ends)
      for (let i = bins.length - 1; i >= 0; i--) {
        if (val >= bins[i].binStart) {
          bins[i].count++;
          break;
        }
      }
    });

    return bins;
  };

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-40 rounded-2xl bg-white/5 border border-white/5" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-[450px] rounded-2xl bg-white/5 border border-white/5" />
          <div className="h-[450px] rounded-2xl bg-white/5 border border-white/5" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] text-center space-y-4">
        <div className="p-4 rounded-full bg-danger/10 border border-danger/20 text-danger">
          <AlertTriangle className="w-10 h-10" />
        </div>
        <h3 className="text-lg font-bold text-white">Segment Module Error</h3>
        <p className="text-textMuted text-sm">{error}</p>
      </div>
    );
  }

  // Map segment names to appropriate configurations
  const segmentConfigs = {
    "Champions": {
      icon: Sparkles,
      color: "success",
      colorHex: "#10B981",
      badgeClass: "bg-success/10 text-success border-success/20",
      description: "Your most valuable customers. They buy frequently, spend heavily, and purchased very recently.",
      action: "Reward with loyalty points, VIP previews, and personalized thank-you offers. Avoid spamming."
    },
    "Loyal Customers": {
      icon: TrendingUp,
      color: "primary",
      colorHex: "#6366F1",
      badgeClass: "bg-primary/10 text-primary border-primary/20",
      description: "Consistent and responsive buyers. They spend well and buy regularly.",
      action: "Upsell premium products, offer subscription packages, and ask for reviews or referrals."
    },
    "At-Risk": {
      icon: AlertTriangle,
      color: "warning",
      colorHex: "#F59E0B",
      badgeClass: "bg-warning/10 text-warning border-warning/20",
      description: "Used to be frequent buyers, but haven't purchased in a long time. Churn threat.",
      action: "Send Win-back email campaigns, offer steep discounts, and conduct surveys to address issues."
    },
    "Lost Customers": {
      icon: Trash2,
      color: "danger",
      colorHex: "#EF4444",
      badgeClass: "bg-danger/10 text-danger border-danger/20",
      description: "Inactive for a very long time, low frequency, and minimal monetary spend.",
      action: "Run low-cost automated re-activation campaigns. Reallocate core marketing budget elsewhere."
    }
  };

  const getCardBorder = (segment) => {
    if (activeSegment === segment) {
      if (segment === "Champions") return "border-success bg-success/[0.03]";
      if (segment === "Loyal Customers") return "border-primary bg-primary/[0.03]";
      if (segment === "At-Risk") return "border-warning bg-warning/[0.03]";
      if (segment === "Lost Customers") return "border-danger bg-danger/[0.03]";
    }
    return "border-white/10 hover:border-white/20 bg-white/[0.02]";
  };

  // Compute histograms for the active segment
  const recencyHist = getHistogramData(activeSegment, 'Recency', 10);
  const frequencyHist = getHistogramData(activeSegment, 'Frequency', 10);
  const monetaryHist = getHistogramData(activeSegment, 'Monetary', 10);
  const activeConfig = segmentConfigs[activeSegment] || segmentConfigs['Champions'];
  const activeOverview = overview?.centroids.find(c => c.Segment === activeSegment);
  const activeDist = overview?.distribution.find(d => d.segment === activeSegment);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Customer Segmentation Explorer</h1>
          <p className="text-textMuted text-sm mt-1">
            Profiling segments generated via 9D K-Means clustering. Click cards below to drill down.
          </p>
        </div>
        <button
          onClick={() => fetchData(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-textMuted hover:text-white bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 transition-all duration-200 self-start sm:self-auto"
          title="Force refresh segments stats cache"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Data</span>
        </button>
      </div>

      {/* Segment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {overview?.distribution.map((item) => {
          const segName = item.segment;
          const config = segmentConfigs[segName] || { icon: HelpCircle, badgeClass: 'bg-white/5 border-white/10 text-white', colorHex: '#94A3B8' };
          const Icon = config.icon;
          const centroid = overview.centroids.find(c => c.Segment === segName);

          return (
            <motion.div
              key={segName}
              whileHover={{ scale: 1.01 }}
              onClick={() => setActiveSegment(segName)}
              className={`glass-panel p-6 cursor-pointer border transition-all duration-300 ${getCardBorder(segName)}`}
            >
              <div className="flex justify-between items-center">
                <span className={`px-2.5 py-0.5 rounded-full border text-[10px] font-bold ${config.badgeClass}`}>
                  {segName}
                </span>
                <Icon className="w-4 h-4" style={{ color: config.colorHex }} />
              </div>
              <div className="mt-4">
                <h4 className="text-3xl font-extrabold text-white tracking-tight">{item.count}</h4>
                <p className="text-textMuted text-xs mt-0.5">{(item.percentage ?? 0).toFixed(1)}% of userbase</p>
              </div>
              
              {/* Avg values indicator */}
              <div className="mt-4 pt-3 border-t border-white/5 grid grid-cols-3 text-[10px] text-textMuted gap-1 text-center">
                <div>
                  <span className="block font-bold text-white">{(centroid?.Recency ?? 0).toFixed(0)}d</span>
                  Recency
                </div>
                <div>
                  <span className="block font-bold text-white">{(centroid?.Frequency ?? 0).toFixed(1)}x</span>
                  Freq
                </div>
                <div>
                  <span className="block font-bold text-white">£{(centroid?.Monetary ?? 0).toFixed(0)}</span>
                  Monetary
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Detail Analysis Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Card: PCA Plot */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white">2D PCA Cluster Representation</h3>
            <p className="text-textMuted text-xs mt-1">Dimensionality reduction mapping 9D feature profiles to 2D space.</p>
          </div>
          <div className="flex-1 flex items-center justify-center p-4 min-h-[300px] mt-4 rounded-xl border border-white/5 bg-white/[0.01]">
            <img 
              src={`${apiBaseUrl}/api/charts/pca?t=${new Date().getTime()}`} 
              alt="Customer Segment PCA" 
              className="max-h-[350px] object-contain rounded-lg shadow-2xl cursor-pointer hover:opacity-90 hover:scale-[1.01] transition-all duration-200"
              onClick={() => setActiveModalImage({
                src: `${apiBaseUrl}/api/charts/pca?t=${new Date().getTime()}`,
                label: "2D PCA Cluster Representation",
                description: "Dimensionality reduction mapping 9D feature profiles to 2D space."
              })}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div className="hidden flex-col items-center gap-2 text-textMuted text-xs">
              <ImageIcon className="w-8 h-8 text-white/20" />
              <span>Centroid plot image currently loading or unavailable...</span>
            </div>
          </div>
        </div>

        {/* Right Card: Segment Profile Insights */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h3 className="text-base font-bold text-white">Segment Profile:</h3>
              <span className={`px-3 py-1 rounded-full border text-xs font-bold ${activeConfig.badgeClass}`}>
                {activeSegment}
              </span>
            </div>
            <p className="text-textMuted text-xs mt-1">Tailored marketing strategies and cohort metrics.</p>
          </div>

          <div className="flex-1 space-y-6 mt-6">
            {/* Characteristic description */}
            <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] space-y-2">
              <h5 className="text-xs font-bold text-white uppercase tracking-wider">Cohort Characterization</h5>
              <p className="text-textMuted text-xs leading-relaxed">{activeConfig.description}</p>
            </div>

            {/* Action Recommendations */}
            <div className="p-4 rounded-xl border border-secondary/20 bg-secondary/[0.02] space-y-2">
              <h5 className="text-xs font-bold text-secondary uppercase tracking-wider flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5" /> Recommended SaaS Action
              </h5>
              <p className="text-textPrimary text-xs leading-relaxed font-medium">{activeConfig.action}</p>
            </div>

            {/* In-depth centroids profile */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-3 rounded-xl border border-white/5 bg-white/[0.01] text-center">
                <span className="block text-lg font-bold text-white">{(activeOverview?.Recency ?? 0).toFixed(1)}d</span>
                <span className="text-[10px] text-textMuted uppercase tracking-wider font-semibold">Avg Recency</span>
              </div>
              <div className="p-3 rounded-xl border border-white/5 bg-white/[0.01] text-center">
                <span className="block text-lg font-bold text-white">{(activeOverview?.Frequency ?? 0).toFixed(1)} orders</span>
                <span className="text-[10px] text-textMuted uppercase tracking-wider font-semibold">Avg Frequency</span>
              </div>
              <div className="p-3 rounded-xl border border-white/5 bg-white/[0.01] text-center">
                <span className="block text-lg font-bold text-white">£{(activeOverview?.Monetary ?? 0).toFixed(2)}</span>
                <span className="text-[10px] text-textMuted uppercase tracking-wider font-semibold">Avg Monetary</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Histograms Distribution Charts */}
      <div className="glass-panel p-6">
        <div>
          <h3 className="text-base font-bold text-white">RFM Feature Distribution Bins for {activeSegment}</h3>
          <p className="text-textMuted text-xs mt-1">Histogram profiles showing distribution densities for customer counts.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          {/* Recency Histogram */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-white text-center">Recency Distribution (Days since Last Order)</h4>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={recencyHist}>
                  <XAxis dataKey="rangeName" stroke="#94A3B8" fontSize={9} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0A0F1E', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    labelStyle={{ color: '#F1F5F9' }}
                    formatter={(val) => [`${val} Customers`, 'Count']}
                  />
                  <Bar dataKey="count" fill={activeConfig.colorHex} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Frequency Histogram */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-white text-center">Frequency Distribution (Unique Invoices)</h4>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={frequencyHist}>
                  <XAxis dataKey="rangeName" stroke="#94A3B8" fontSize={9} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0A0F1E', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    labelStyle={{ color: '#F1F5F9' }}
                    formatter={(val) => [`${val} Customers`, 'Count']}
                  />
                  <Bar dataKey="count" fill={activeConfig.colorHex} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Monetary Histogram */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-white text-center">Monetary Distribution (Total Spend Amount)</h4>
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monetaryHist}>
                  <XAxis dataKey="rangeName" stroke="#94A3B8" fontSize={9} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0A0F1E', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    labelStyle={{ color: '#F1F5F9' }}
                    formatter={(val) => [`${val} Customers`, 'Count']}
                  />
                  <Bar dataKey="count" fill={activeConfig.colorHex} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Segments;
