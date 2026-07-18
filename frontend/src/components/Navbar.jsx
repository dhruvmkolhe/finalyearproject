import React from 'react';
import { Menu, Server, ShieldAlert } from 'lucide-react';

const Navbar = ({ setIsSidebarOpen, systemStatus }) => {
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-16 px-6 border-b border-white/10 bg-darkbg/60 backdrop-blur-xl">
      {/* Left side: Hamburger button + Page Title */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => setIsSidebarOpen(true)}
          className="p-2 rounded-lg text-textMuted hover:text-white hover:bg-white/5 md:hidden"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="hidden sm:block">
          <h2 className="text-sm font-semibold text-white tracking-wide">PredictIQ Analytics Console</h2>
        </div>
      </div>

      {/* Right side: Academic project info & system health */}
      <div className="flex items-center gap-6">
        {/* Project Authors / Advisor info */}
        <div className="hidden lg:flex flex-col text-right border-r border-white/10 pr-6">
          <div className="text-[11px] font-semibold text-white">
            By: Dhruv, Aditi Kesarkar, Aum Patel, Manav Patel
          </div>
          <div className="text-[10px] text-textMuted font-medium">
            Supervised by: <span className="text-secondary font-semibold">Dr. Sneha</span>
          </div>
        </div>

        {/* System status dot */}
        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-full border border-white/5 bg-white/[0.02] text-xs">
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
              systemStatus === 'healthy' ? 'bg-success' : 'bg-warning'
            }`}></span>
            <span className={`relative inline-flex rounded-full h-2 w-2 ${
              systemStatus === 'healthy' ? 'bg-success' : 'bg-warning'
            }`}></span>
          </span>
          <span className="font-semibold text-textPrimary text-[11px]">
            API: {systemStatus === 'healthy' ? 'CONNECTED' : 'DEGRADED'}
          </span>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
