import React, { useState, useEffect, useRef } from 'react';
import { ResponsiveContainer, RadialBarChart, RadialBar, Cell } from 'recharts';
import { 
  Play, Square, Activity, ShieldCheck, AlertTriangle, 
  HelpCircle, RefreshCw, Layers
} from 'lucide-react';

const Live = ({ apiBaseUrl }) => {
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [feed, setFeed] = useState([]);
  
  // Running stats
  const [stats, setStats] = useState({
    totalProcessed: 0,
    correct: 0,
    accuracy: 100,
    pps: 0, // predictions per second
    segmentDistribution: {
      "Champions": 0,
      "Loyal Customers": 0,
      "At-Risk": 0,
      "Lost Customers": 0
    }
  });

  const wsRef = useRef(null);
  const feedContainerRef = useRef(null);
  const ppsTimerRef = useRef(null);
  const processedLastSecRef = useRef(0);

  // Auto-scroll to bottom of the feed (non-intrusive container-level scroll)
  useEffect(() => {
    const container = feedContainerRef.current;
    if (container) {
      // Check if user is close to the bottom (within 80px)
      const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
      
      // Auto-scroll ONLY if they are at the bottom or if it is the first record
      if (isAtBottom || feed.length <= 1) {
        container.scrollTop = container.scrollHeight;
      }
    }
  }, [feed]);

  // PPS calculations
  useEffect(() => {
    if (running) {
      ppsTimerRef.current = setInterval(() => {
        setStats(prev => ({
          ...prev,
          pps: processedLastSecRef.current
        }));
        processedLastSecRef.current = 0;
      }, 1000);
    } else {
      clearInterval(ppsTimerRef.current);
      setStats(prev => ({ ...prev, pps: 0 }));
      processedLastSecRef.current = 0;
    }

    return () => clearInterval(ppsTimerRef.current);
  }, [running]);

  // Clean up socket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const startStream = () => {
    if (wsRef.current) wsRef.current.close();
    
    // Clear feed and reset stats
    setFeed([]);
    setStats({
      totalProcessed: 0,
      correct: 0,
      accuracy: 100,
      pps: 0,
      segmentDistribution: {
        "Champions": 0,
        "Loyal Customers": 0,
        "At-Risk": 0,
        "Lost Customers": 0
      }
    });
    processedLastSecRef.current = 0;

    // Connect to WebSocket (convert http endpoint to ws)
    const wsUrl = apiBaseUrl.replace('http', 'ws') + '/ws/realtime-predict';
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    setRunning(true);

    ws.onopen = () => {
      setConnected(true);
      console.log("WebSocket connected successfully.");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error(data.error);
        ws.close();
        return;
      }

      // Add to feed list, capping at 50 items
      setFeed(prev => {
        const updated = [...prev, data];
        if (updated.length > 50) updated.shift();
        return updated;
      });

      // Update counters
      processedLastSecRef.current += 1;

      setStats(prev => {
        const seg = data.segment || 'Unknown';
        const updatedSegDist = { ...prev.segmentDistribution };
        if (updatedSegDist[seg] !== undefined) {
          updatedSegDist[seg] += 1;
        }

        return {
          totalProcessed: data.total_processed,
          correct: prev.correct + (data.prediction === data.true_label ? 1 : 0),
          accuracy: data.running_accuracy,
          segmentDistribution: updatedSegDist,
          pps: prev.pps
        };
      });
    };

    ws.onclose = () => {
      setConnected(false);
      setRunning(false);
      console.log("WebSocket closed.");
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.close();
    };
  };

  const stopStream = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  // Recharts RadialBarChart formatting
  const radialData = [
    {
      name: 'Accuracy',
      value: stats.accuracy,
      fill: stats.accuracy >= 80 ? '#10B981' : stats.accuracy >= 60 ? '#F59E0B' : '#EF4444'
    }
  ];

  const segmentColors = {
    "Champions": "bg-success/10 text-success border-success/20",
    "Loyal Customers": "bg-primary/10 text-primary border-primary/20",
    "At-Risk": "bg-warning/10 text-warning border-warning/20",
    "Lost Customers": "bg-danger/10 text-danger border-danger/20"
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Real-Time Prediction Stream</h1>
          <p className="text-textMuted text-sm mt-1">
            Simulates a live transactional feed by replaying model testing cohorts row-by-row.
          </p>
        </div>
        
        {/* Controls */}
        <div className="flex items-center gap-3">
          {/* Connection status indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/5 bg-white/[0.02]">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-success animate-pulse' : 'bg-danger'}`} />
            <span className="text-[10px] font-bold text-textPrimary uppercase tracking-wider">
              {connected ? 'Streaming' : 'Disconnected'}
            </span>
          </div>

          {!running ? (
            <button
              onClick={startStream}
              className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-success hover:bg-success/80 transition-all shadow-lg shadow-success/15"
            >
              <Play className="w-3.5 h-3.5 fill-white" /> Start Stream
            </button>
          ) : (
            <button
              onClick={stopStream}
              className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-danger hover:bg-danger/80 transition-all shadow-lg shadow-danger/15"
            >
              <Square className="w-3.5 h-3.5 fill-white" /> Stop Stream
            </button>
          )}
        </div>
      </div>

      {/* Stats Board & Accuracy Gauge */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Accuracy Gauge Card */}
        <div className="glass-panel p-6 flex flex-col justify-between items-center text-center">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Running Accuracy</h3>
            <p className="text-textMuted text-[10px] mt-0.5">Classification correctness score.</p>
          </div>
          
          <div className="h-40 w-40 relative flex items-center justify-center mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart 
                cx="50%" cy="50%" 
                innerRadius="65%" outerRadius="90%" 
                barSize={12} 
                data={radialData}
                startAngle={180} endAngle={-180}
              >
                <RadialBar background clockWise dataKey="value" cornerRadius={6} />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="absolute text-center">
              <span className="text-3xl font-extrabold text-white tracking-tight">
                {stats.accuracy.toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Dynamic Running Counters */}
        <div className="glass-panel p-6 grid grid-cols-2 gap-4 lg:col-span-2">
          {/* Total Processed */}
          <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] flex flex-col justify-center">
            <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider">Total Processed</span>
            <span className="text-2xl font-extrabold text-white mt-1 font-mono">{stats.totalProcessed}</span>
          </div>

          {/* Predictions/sec */}
          <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] flex flex-col justify-center">
            <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider">Inference Speed</span>
            <span className="text-2xl font-extrabold text-white mt-1 font-mono">{stats.pps} p/s</span>
          </div>

          {/* Correct Predictions */}
          <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] flex flex-col justify-center">
            <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider">Correct Predictions</span>
            <span className="text-2xl font-extrabold text-success mt-1 font-mono">{stats.correct}</span>
          </div>

          {/* System Load status */}
          <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01] flex flex-col justify-center">
            <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider">WebSocket Buffer</span>
            <span className="text-2xl font-extrabold text-secondary mt-1 font-mono">200ms delay</span>
          </div>
        </div>

        {/* Live Segment Distribution */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Cohort Yield</h3>
            <p className="text-textMuted text-[10px] mt-0.5">Discovered segments in this session.</p>
          </div>
          
          <div className="space-y-2 mt-4 flex-1 flex flex-col justify-center">
            {Object.entries(stats.segmentDistribution).map(([seg, count]) => {
              const total = stats.totalProcessed || 1;
              const pct = (count / total) * 100;
              
              const barColors = {
                "Champions": "bg-success",
                "Loyal Customers": "bg-primary",
                "At-Risk": "bg-warning",
                "Lost Customers": "bg-danger"
              };

              return (
                <div key={seg} className="space-y-0.5">
                  <div className="flex justify-between text-[10px]">
                    <span className="font-semibold text-textMuted">{seg}</span>
                    <span className="font-bold text-white">{count} ({pct.toFixed(0)}%)</span>
                  </div>
                  <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${barColors[seg] || 'bg-white'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Scrolling Prediction Feed */}
      <div className="glass-panel p-6 flex flex-col h-[400px]">
        <div>
          <h3 className="text-base font-bold text-white">Live Transactions Activity Log</h3>
          <p className="text-textMuted text-xs mt-1">Scroll feed detailing input profiles, target repurchase events, and model classifications.</p>
        </div>

        <div ref={feedContainerRef} className="flex-1 overflow-y-auto border border-white/5 bg-black/20 rounded-xl mt-6 p-4 font-mono text-xs space-y-2.5">
          {feed.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-textMuted gap-2">
              <Activity className="w-8 h-8 text-white/5 animate-pulse" />
              <span>Inference pipeline is idle. Start stream to activate feed logs.</span>
            </div>
          ) : (
            feed.map((row, idx) => {
              const isCorrect = row.prediction === row.true_label;
              const buyLabel = row.prediction === 1 ? 'Repurchase' : 'Churn';
              const trueBuyLabel = row.true_label === 1 ? 'Repurchase' : 'Churn';
              
              return (
                <div 
                  key={idx} 
                  className={`flex flex-col md:flex-row md:items-center justify-between border-b border-white/5 pb-2 gap-2 text-textPrimary hover:bg-white/[0.01]`}
                >
                  <div className="flex flex-wrap items-center gap-3">
                    {/* Timestamp */}
                    <span className="text-textMuted">[{new Date().toLocaleTimeString()}]</span>
                    
                    {/* Customer Id */}
                    <span className="text-white font-bold">Cust #{row.customer_id}</span>
                    
                    {/* Segment Badge */}
                    <span className={`px-2 py-px rounded border text-[9px] font-semibold ${segmentColors[row.segment] || 'bg-white/5'}`}>
                      {row.segment}
                    </span>

                    {/* Inputs */}
                    <span className="text-textMuted">
                      (R:{row.features.Recency}d | F:{row.features.Frequency}x | M:£{row.features.Monetary.toFixed(0)})
                    </span>
                  </div>

                  <div className="flex items-center gap-3 self-end md:self-auto">
                    {/* true label */}
                    <span className="text-textMuted">Actual: <span className="text-white font-semibold">{trueBuyLabel}</span></span>

                    {/* prediction result */}
                    <span className={`px-2 py-px rounded font-bold ${
                      row.prediction === 1 
                        ? 'bg-success/10 text-success border border-success/20' 
                        : 'bg-danger/10 text-danger border border-danger/20'
                    }`}>
                      Pred: {buyLabel} ({(row.probability * 100).toFixed(0)}%)
                    </span>

                    {/* correctness check */}
                    {isCorrect ? (
                      <span className="text-success flex items-center gap-0.5"><ShieldCheck className="w-3.5 h-3.5" /> OK</span>
                    ) : (
                      <span className="text-warning flex items-center gap-0.5"><AlertTriangle className="w-3.5 h-3.5" /> ERR</span>
                    )}
                  </div>
                </div>
              );
            })
          )}
          {/* Clean scroller wrapper */}
        </div>
      </div>

    </div>
  );
};

export default Live;
