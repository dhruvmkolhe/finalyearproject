import React, { useState } from 'react';
import { Lock, User, Eye, EyeOff, LogIn, AlertCircle } from 'lucide-react';

// Custom PredictIQ Logo for Login Header
const PredictIQLogoBig = () => (
  <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-20 h-20 mx-auto">
    <defs>
      <linearGradient id="wave-gradient" x1="0" y1="100" x2="100" y2="0" gradientUnits="userSpaceOnUse">
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
      stroke="url(#wave-gradient)" 
      strokeWidth="5" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
    />
    
    {/* Highlight nodes */}
    <circle cx="44" cy="52" r="3.5" fill="#00C9FF" />
    <circle cx="72" cy="28" r="4" fill="#7B2FFF" />
  </svg>
);

const Login = ({ apiBaseUrl = 'http://localhost:8000', onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          password
        })
      });

      const data = await response.json();

      if (response.ok && data.success) {
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('authToken', data.access_token);
        localStorage.setItem('authUser', data.user.username);
        onLoginSuccess();
      } else {
        setError(data.detail || 'Incorrect password or username.');
      }
    } catch (err) {
      console.error(err);
      setError('Network error. Check if the server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center bg-[#0A0F1E] overflow-hidden px-4">
      {/* Decorative gradient spheres/glows */}
      <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-80 h-80 rounded-full bg-primary/20 blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-96 h-96 rounded-full bg-secondary/15 blur-[120px] pointer-events-none" />

      {/* Main Login Card */}
      <div className="w-full max-w-md glass-panel relative z-10 p-8 md:p-10 shadow-2xl border border-white/10 rounded-2xl flex flex-col justify-center">
        
        {/* Header/Branding */}
        <div className="text-center mb-8">
          <PredictIQLogoBig />
          <h2 className="mt-4 text-2xl md:text-3xl font-extrabold text-white tracking-tight">
            Predict<span className="text-secondary bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">IQ</span> Portal
          </h2>
          <p className="mt-2 text-xs md:text-sm text-textMuted font-medium">
            Authorized analytics and pipeline console access
          </p>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Username */}
          <div className="space-y-2">
            <label htmlFor="username" className="text-xs font-semibold text-textMuted uppercase tracking-wider block">
              Username
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-textMuted">
                <User className="h-4 w-4" />
              </span>
              <input
                id="username"
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username (e.g. admin)"
                className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/[0.03] hover:bg-white/[0.05] border border-white/10 focus:border-primary/50 text-white placeholder-textMuted/60 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all duration-200"
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-2">
            <label htmlFor="password" className="text-xs font-semibold text-textMuted uppercase tracking-wider block">
              Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-textMuted">
                <Lock className="h-4 w-4" />
              </span>
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-10 py-3 rounded-xl bg-white/[0.03] hover:bg-white/[0.05] border border-white/10 focus:border-primary/50 text-white placeholder-textMuted/60 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all duration-200"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-textMuted hover:text-white transition-colors duration-150"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-2.5 p-3 rounded-xl bg-danger/10 border border-danger/20 text-danger text-xs font-medium animate-shake">
              <AlertCircle className="h-4.5 w-4.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-white font-semibold text-sm transition-all duration-300 accent-gradient accent-gradient-hover cursor-pointer"
          >
            {isLoading ? (
              <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <>
                <LogIn className="h-4.5 w-4.5" />
                <span>Enter Console</span>
              </>
            )}
          </button>
        </form>

        {/* Footer Info */}
        <div className="mt-8 pt-6 border-t border-white/5 text-center">
          <p className="text-[11px] text-textMuted">
            Secure JWT-based authentication console keys.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
