import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Bot, 
  User, 
  Sparkles, 
  Trash2, 
  HelpCircle, 
  MessageSquare,
  AlertTriangle,
  Loader2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const Chat = ({ apiBaseUrl }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello! I am your PredictIQ Analytics Assistant. I have real-time access to the customer database, segment breakdowns, and machine learning models. How can I help you optimize your retail marketing and predictive pipelines today?"
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);

  const suggestionChips = [
    { label: "Paradox", text: "Explain the Baseline F1-Score Paradox in PredictIQ.", icon: AlertTriangle, color: "from-amber-500/20 to-orange-500/20 text-orange-400 border-orange-500/30" },
    { label: "Compare Models", text: "Compare the models in the classifier arena. Which one is recommended and why?", icon: Sparkles, color: "from-primary/20 to-indigo-500/20 text-[#22D3EE] border-primary/30" },
    { label: "Segment Details", text: "Give me an overview of the customer segments and their centroids.", icon: HelpCircle, color: "from-success/20 to-emerald-500/20 text-emerald-400 border-success/30" },
    { label: "Top Products", text: "What are the top 10 products by revenue and quantities sold?", icon: MessageSquare, color: "from-secondary/20 to-cyan-500/20 text-cyan-400 border-secondary/30" },
    { label: "VIP Strategy", text: "Suggest targeted marketing campaigns for each customer segment.", icon: Sparkles, color: "from-[#F59E0B]/20 to-yellow-500/20 text-yellow-400 border-yellow-500/30" }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (textToSend) => {
    const text = textToSend || input;
    if (!text.trim()) return;

    if (!textToSend) setInput('');
    setError(null);
    setIsLoading(true);

    const userMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);

    try {
      const history = messages.slice(1); // Exclude the initial greeting
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: text,
          history: history
        }),
      });

      const data = await response.json();

      if (data.success && data.reply) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
      } else {
        setError(data.error || "Failed to retrieve response from the assistant.");
      }
    } catch (err) {
      console.error(err);
      setError("Unable to connect to the backend server. Make sure the FastAPI application is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: "Hello! I am your PredictIQ Analytics Assistant. I have real-time access to the customer database, segment breakdowns, and machine learning models. How can I help you optimize your retail marketing and predictive pipelines today?"
      }
    ]);
    setError(null);
  };

  const markdownComponents = {
    h1: ({ children }) => <h1 className="text-lg font-bold text-white mt-3 mb-1">{children}</h1>,
    h2: ({ children }) => <h2 className="text-base font-bold text-white mt-3 mb-1">{children}</h2>,
    h3: ({ children }) => <h3 className="text-sm font-bold text-cyan-300 mt-2 mb-1">{children}</h3>,
    p: ({ children }) => <p className="mb-2 leading-relaxed">{children}</p>,
    strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
    em: ({ children }) => <em className="italic text-slate-300">{children}</em>,
    ul: ({ children }) => <ul className="my-2 space-y-1 pl-1">{children}</ul>,
    ol: ({ children }) => <ol className="my-2 space-y-1 pl-1 list-decimal list-inside">{children}</ol>,
    li: ({ children }) => (
      <li className="flex items-start gap-2">
        <span className="text-cyan-400 mt-1.5 shrink-0 text-[8px]">●</span>
        <span>{children}</span>
      </li>
    ),
    code: ({ inline, children }) => inline
      ? <code className="bg-black/40 text-cyan-300 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
      : <pre className="bg-black/30 border border-white/10 p-3 rounded-lg overflow-x-auto my-3 text-xs font-mono text-slate-200 whitespace-pre-wrap"><code>{children}</code></pre>,
    hr: () => <hr className="border-white/10 my-3" />,
    blockquote: ({ children }) => <blockquote className="border-l-2 border-cyan-500/50 pl-3 my-2 text-slate-400 italic">{children}</blockquote>,
    table: ({ children }) => (
      <div className="overflow-x-auto my-4 rounded-xl border border-white/10 bg-white/[0.02]">
        <table className="min-w-full divide-y divide-white/10 text-xs text-left text-slate-300">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-white/5 text-white font-bold">{children}</thead>,
    tbody: ({ children }) => <tbody className="divide-y divide-white/5">{children}</tbody>,
    tr: ({ children }) => <tr className="hover:bg-white/[0.01] transition-colors">{children}</tr>,
    th: ({ children }) => <th className="px-4 py-2.5 font-semibold text-slate-200">{children}</th>,
    td: ({ children }) => <td className="px-4 py-2.5 text-slate-300">{children}</td>,
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)] max-w-5xl mx-auto space-y-4">
      {/* Header Panel */}
      <div className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/10 rounded-2xl backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-secondary shadow-lg shadow-primary/20">
            <Bot className="w-5 h-5 text-white animate-pulse" />
          </div>
          <div>
            <h2 className="font-bold text-base text-white tracking-wide flex items-center gap-1.5">
              PredictIQ Copilot <span className="text-[10px] bg-secondary/20 text-secondary border border-secondary/30 px-1.5 py-0.5 rounded font-mono">GROQ</span>
            </h2>
            <p className="text-xs text-textMuted">Ask questions about models, data, and segment strategy</p>
          </div>
        </div>
        <button 
          onClick={clearChat}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-white/10 text-xs text-textMuted hover:text-red-400 hover:bg-red-500/10 hover:border-red-500/20 transition-all duration-200"
          title="Clear Chat History"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Reset</span>
        </button>
      </div>

      {/* Chat Messages Log */}
      <div className="flex-1 overflow-y-auto p-4 rounded-2xl bg-white/[0.01] border border-white/5 flex flex-col space-y-4 min-h-0 relative">
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'self-end flex-row-reverse' : 'self-start'}`}
            >
              {/* Avatar icon */}
              <div className={`flex items-center justify-center w-8 h-8 rounded-lg shrink-0 ${
                msg.role === 'user' 
                  ? 'bg-gradient-to-br from-primary to-indigo-600' 
                  : 'bg-white/5 border border-white/10'
              }`}>
                {msg.role === 'user' ? (
                  <User className="w-4 h-4 text-white" />
                ) : (
                  <Bot className="w-4 h-4 text-secondary" />
                )}
              </div>

              {/* Message bubble */}
              <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed border ${
                msg.role === 'user'
                  ? 'bg-gradient-to-r from-primary/30 to-indigo-600/20 border-primary/40 text-white rounded-tr-none'
                  : 'bg-white/[0.03] border-white/10 text-textPrimary rounded-tl-none'
              }`}>
                {msg.role === 'user' ? (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                ) : (
                  <div className="space-y-1 text-slate-200 prose-invert">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{msg.content}</ReactMarkdown>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading Spinner */}
        {isLoading && (
          <div className="flex gap-3 max-w-[80%] self-start">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 border border-white/10 shrink-0">
              <Bot className="w-4 h-4 text-secondary" />
            </div>
            <div className="px-4 py-3 rounded-2xl bg-white/[0.03] border border-white/10 rounded-tl-none flex items-center gap-2 text-xs text-textMuted">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-secondary" />
              <span>Analyzing PredictIQ metrics...</span>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="flex gap-2.5 p-3.5 rounded-xl border border-red-500/20 bg-red-500/10 text-xs text-red-400 self-center max-w-xl text-center">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Reference Anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Prompts Grid */}
      {messages.length === 1 && !isLoading && (
        <div className="space-y-1.5">
          <p className="text-[11px] font-semibold text-textMuted uppercase tracking-wider px-1">Suggested Inquiries</p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {suggestionChips.map((chip, idx) => {
              const Icon = chip.icon;
              return (
                <button
                  key={idx}
                  onClick={() => handleSend(chip.text)}
                  className={`flex items-start gap-2.5 p-2.5 text-left rounded-xl border text-xs bg-gradient-to-r ${chip.color} hover:scale-[1.01] hover:bg-white/[0.03] transition-all duration-200`}
                >
                  <Icon className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <span className="font-medium">{chip.text}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Input panel */}
      <div className="flex items-center gap-2 p-2.5 bg-white/[0.02] border border-white/10 rounded-2xl backdrop-blur-md">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask a question about the customer dataset or model CV performance..."
          rows="1"
          disabled={isLoading}
          className="flex-1 max-h-24 min-h-[2.5rem] bg-transparent border-0 ring-0 focus:ring-0 focus:outline-none placeholder-textMuted text-sm text-white px-3 py-2 resize-none"
        />
        <button
          onClick={() => handleSend()}
          disabled={isLoading || !input.trim()}
          className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-r from-primary to-secondary text-white shadow-md shadow-primary/10 hover:shadow-primary/20 hover:scale-[1.03] active:scale-[0.98] disabled:opacity-50 disabled:scale-100 disabled:shadow-none transition-all duration-150"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default Chat;
