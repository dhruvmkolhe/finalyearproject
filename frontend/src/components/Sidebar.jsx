import React from 'react';
import { 
  LayoutDashboard, 
  Users, 
  Cpu, 
  Sliders, 
  Activity, 
  ShieldAlert,
  MessageSquare
} from 'lucide-react';

// Custom coded SVG logo: neural network nodes + prediction trend line
const PredictIQLogo = () => (
  <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-10 h-10 drop-shadow-xl">
    <defs>
      <linearGradient id="piq-bg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
        <stop stopColor="#6366F1" />
        <stop offset="1" stopColor="#22D3EE" />
      </linearGradient>
      <filter id="piq-glow">
        <feGaussianBlur stdDeviation="1.5" result="blur" />
        <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
    </defs>

    {/* Rounded background with gradient */}
    <rect width="40" height="40" rx="10" fill="url(#piq-bg)" />
    {/* Inner highlight border */}
    <rect x="1" y="1" width="38" height="38" rx="9" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />

    {/* Neural network — diamond of 4 nodes */}
    {/* Top → Left */}
    <line x1="20" y1="8" x2="8" y2="22" stroke="rgba(255,255,255,0.28)" strokeWidth="1.1" strokeLinecap="round" />
    {/* Top → Right */}
    <line x1="20" y1="8" x2="32" y2="22" stroke="rgba(255,255,255,0.28)" strokeWidth="1.1" strokeLinecap="round" />
    {/* Left → Bottom */}
    <line x1="8" y1="22" x2="20" y2="33" stroke="rgba(255,255,255,0.18)" strokeWidth="1.1" strokeLinecap="round" />
    {/* Right → Bottom */}
    <line x1="32" y1="22" x2="20" y2="33" stroke="rgba(255,255,255,0.18)" strokeWidth="1.1" strokeLinecap="round" />
    {/* Left ↔ Right */}
    <line x1="8" y1="22" x2="32" y2="22" stroke="rgba(255,255,255,0.13)" strokeWidth="1.1" strokeLinecap="round" />
    {/* Top → Bottom (vertical) */}
    <line x1="20" y1="8" x2="20" y2="33" stroke="rgba(255,255,255,0.1)" strokeWidth="1.1" strokeLinecap="round" />

    {/* Prediction trend line — bold upward stroke with glow */}
    <polyline
      filter="url(#piq-glow)"
      points="6,32 12,24 19,27 34,12"
      stroke="white" strokeWidth="2.3"
      strokeLinecap="round" strokeLinejoin="round"
    />
    {/* Arrowhead */}
    <polyline
      points="29,10 34,12 32,17"
      stroke="white" strokeWidth="2.3"
      strokeLinecap="round" strokeLinejoin="round"
    />

    {/* Nodes */}
    <circle cx="20" cy="8" r="3" fill="white" fillOpacity="0.95" />
    <circle cx="8" cy="22" r="2.2" fill="white" fillOpacity="0.6" />
    <circle cx="32" cy="22" r="2.2" fill="white" fillOpacity="0.6" />
    <circle cx="20" cy="33" r="1.7" fill="white" fillOpacity="0.35" />

    {/* Bright dot at trend peak (top-right) */}
    <circle cx="34" cy="12" r="2" fill="white" fillOpacity="0.9" />
  </svg>
);

const Sidebar = ({ currentPage, setCurrentPage, isOpen, setIsOpen }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'segments', label: 'Customer Segments', icon: Users },
    { id: 'models', label: 'Model Performance', icon: Cpu },
    { id: 'predict', label: 'Predict Behavior', icon: Sliders },
    { id: 'live', label: 'Real-time Feed', icon: Activity },
    { id: 'drift', label: 'Data Drift Monitor', icon: ShieldAlert },
    { id: 'chat', label: 'AI Chatbot', icon: MessageSquare },
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/60 md:hidden backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar Container */}
      <aside 
        className={`fixed top-0 bottom-0 left-0 z-50 flex flex-col w-64 border-r border-white/10 bg-darkbg/95 md:bg-darkbg/60 backdrop-blur-xl transition-transform duration-300 md:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header/Logo */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-white/10">
          <PredictIQLogo />
          <div>
            <h1 className="font-bold text-lg text-white tracking-wide">PredictIQ</h1>
            <span className="text-[10px] text-secondary font-semibold uppercase tracking-wider">Enterprise Analytics</span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => {
                  setCurrentPage(item.id);
                  setIsOpen(false); // Close mobile menu after clicking
                }}
                className={`flex items-center gap-3 w-full px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                  isActive 
                    ? 'bg-gradient-to-r from-primary/20 to-secondary/10 border border-primary/30 text-white shadow-lg shadow-primary/5' 
                    : 'text-textMuted hover:text-white hover:bg-white/5 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-secondary' : ''}`} />
                {item.label}
              </button>
            );
          })}
        </nav>



      </aside>
    </>
  );
};

export default Sidebar;
