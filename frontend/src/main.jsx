import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Intercept native fetch to inject the JWT bear token automatically
const originalFetch = window.fetch;
window.fetch = async (input, init = {}) => {
  const token = localStorage.getItem('authToken');
  if (token) {
    const headers = init.headers ? { ...init.headers } : {};
    headers['Authorization'] = `Bearer ${token}`;
    init.headers = headers;
  }
  return originalFetch(input, init);
};

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
