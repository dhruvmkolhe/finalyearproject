import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Segments from './pages/Segments';
import Models from './pages/Models';
import Predict from './pages/Predict';
import Live from './pages/Live';
import Drift from './pages/Drift';
import Chat from './pages/Chat';
import Login from './pages/Login';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    localStorage.getItem('isAuthenticated') === 'true'
  );
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState('degraded');
  const [apiBaseUrl] = useState(import.meta.env.VITE_API_URL || 'http://localhost:8000');
  const [activeModalImage, setActiveModalImage] = useState(null);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setActiveModalImage(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (!isAuthenticated) return;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/health`);
        const data = await res.json();
        if (data.success && data.data.status === 'healthy') {
          setSystemStatus('healthy');
        } else {
          setSystemStatus('degraded');
        }
      } catch (err) {
        setSystemStatus('degraded');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [apiBaseUrl, isAuthenticated]);

  if (!isAuthenticated) {
    return <Login apiBaseUrl={apiBaseUrl} onLoginSuccess={() => setIsAuthenticated(true)} />;
  }

  return (
    <div className="flex min-h-screen bg-[#0A0F1E] text-[#F1F5F9] font-sans">
      {/* Sidebar Navigation */}
      <Sidebar 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage} 
        isOpen={isSidebarOpen} 
        setIsOpen={setIsSidebarOpen} 
        onLogout={() => {
          localStorage.removeItem('isAuthenticated');
          localStorage.removeItem('authUser');
          localStorage.removeItem('authToken');
          setIsAuthenticated(false);
        }}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col md:pl-64 min-w-0">
        <Navbar 
          setIsSidebarOpen={setIsSidebarOpen} 
          systemStatus={systemStatus} 
        />
        <main className="flex-1 p-6 overflow-y-auto">
          {currentPage === 'dashboard' && (
            <Home apiBaseUrl={apiBaseUrl} active={true} />
          )}
          {currentPage === 'segments' && (
            <Segments apiBaseUrl={apiBaseUrl} setActiveModalImage={setActiveModalImage} />
          )}
          {currentPage === 'models' && (
            <Models apiBaseUrl={apiBaseUrl} setActiveModalImage={setActiveModalImage} />
          )}
          {currentPage === 'predict' && (
            <Predict apiBaseUrl={apiBaseUrl} />
          )}
          {currentPage === 'live' && (
            <Live apiBaseUrl={apiBaseUrl} />
          )}
          {currentPage === 'drift' && (
            <Drift apiBaseUrl={apiBaseUrl} />
          )}
          {currentPage === 'chat' && (
            <Chat apiBaseUrl={apiBaseUrl} />
          )}
        </main>
      </div>

      {/* Global Fullscreen Image Preview Modal */}
      {activeModalImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-fade-in cursor-pointer"
          onClick={() => setActiveModalImage(null)}
        >
          <div 
            className="relative max-w-4xl w-full bg-[#0E1325] border border-white/10 rounded-2xl p-6 shadow-2xl flex flex-col items-center gap-4 cursor-default animate-zoom-in"
            onClick={(e) => e.stopPropagation()}
          >
            <button 
              onClick={() => setActiveModalImage(null)}
              className="absolute top-4 right-4 text-white/50 hover:text-white transition-colors text-xl font-bold p-2"
              aria-label="Close modal"
            >
              ✕
            </button>
            <h3 className="text-xl font-bold text-white text-center">{activeModalImage.label}</h3>
            <p className="text-textMuted text-sm text-center -mt-2 max-w-2xl">{activeModalImage.description}</p>
            <div className="w-full flex items-center justify-center p-2 bg-white/[0.02] rounded-xl border border-white/5 mt-2">
              <img 
                src={activeModalImage.src} 
                alt={activeModalImage.label} 
                className="max-h-[65vh] object-contain rounded-lg shadow-xl"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
