import React from 'react';
import { 
  LayoutDashboard, 
  Users, 
  Cpu, 
  Sliders, 
  Activity, 
  ShieldAlert,
  MessageSquare,
  LogOut
} from 'lucide-react';

const PredictIQLogo = () => (
  <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-10 h-10 shrink-0">
    <defs>
      <linearGradient id="sidebar-wave-gradient" x1="0" y1="100" x2="100" y2="0" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#00C9FF" />
        <stop offset="100%" stopColor="#7B2FFF" />
      </linearGradient>
    </defs>
    {/* Background with Navy Blue */}
    <rect width="100" height="100" rx="24" fill="#0B1D3A" />
    
    {/* Stylized Q (Drawn in Crisp White) */}
    <circle cx="50" cy="48" r="22" stroke="#FFFFFF" strokeWidth="5.5" strokeLinecap="round" />
    <path d="M 64,62 L 76,74" stroke="#FFFFFF" strokeWidth="5.5" strokeLinecap="round" />
    
    {/* Rising Waveform passing through the center of Q */}
    <path 
      d="M 18,72 Q 32,72 44,52 T 72,28" 
      stroke="url(#sidebar-wave-gradient)" 
      strokeWidth="5" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
    />
    
    {/* Highlight nodes */}
    <circle cx="44" cy="52" r="3.5" fill="#00C9FF" />
    <circle cx="72" cy="28" r="4" fill="#7B2FFF" />
  </svg>
);

const Sidebar = ({ currentPage, setCurrentPage, isOpen, setIsOpen, onLogout }) => {
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

        {/* User / Logout Section at Bottom */}
        <div className="p-4 border-t border-white/10 bg-white/[0.01]">
          <div className="flex items-center justify-between gap-3 px-2 py-2">
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-white truncate">
                {localStorage.getItem('authUser') || 'Administrator'}
              </p>
              <p className="text-[10px] text-textMuted uppercase font-medium tracking-wider">
                Session Active
              </p>
            </div>
            <button
              onClick={onLogout}
              title="Logout Session"
              className="p-2 rounded-xl text-textMuted hover:text-danger hover:bg-danger/10 border border-transparent hover:border-danger/20 transition-all duration-200 cursor-pointer"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>

      </aside>
    </>
  );
};

export default Sidebar;
