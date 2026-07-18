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

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState('degraded');
  const [apiBaseUrl] = useState('http://localhost:8000');

  useEffect(() => {
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
  }, [apiBaseUrl]);

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Home apiBaseUrl={apiBaseUrl} />;
      case 'segments':
        return <Segments apiBaseUrl={apiBaseUrl} />;
      case 'models':
        return <Models apiBaseUrl={apiBaseUrl} />;
      case 'predict':
        return <Predict apiBaseUrl={apiBaseUrl} />;
      case 'live':
        return <Live apiBaseUrl={apiBaseUrl} />;
      case 'drift':
        return <Drift apiBaseUrl={apiBaseUrl} />;
      case 'chat':
        return <Chat apiBaseUrl={apiBaseUrl} />;
      default:
        return <Home apiBaseUrl={apiBaseUrl} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-[#0A0F1E] text-[#F1F5F9] font-sans">
      {/* Sidebar Navigation */}
      <Sidebar 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage} 
        isOpen={isSidebarOpen} 
        setIsOpen={setIsSidebarOpen} 
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col md:pl-64 min-w-0">
        <Navbar 
          setIsSidebarOpen={setIsSidebarOpen} 
          systemStatus={systemStatus} 
        />
        <main className="flex-1 p-6 overflow-y-auto">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}

export default App;
