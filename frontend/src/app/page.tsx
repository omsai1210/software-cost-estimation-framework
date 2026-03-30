"use client";

import React, { useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Calculator, Zap, Server, BrainCircuit, Activity, Clock, ShieldCheck, Loader2, AlertTriangle, FileCode } from 'lucide-react';

const MAE_DATA = [
  { name: 'Smart Hybrid', val: 198.5, color: '#10b981' }, 
  { name: 'Random For.', val: 214.9, color: '#3b82f6' }, 
  { name: 'Deep Lrng', val: 1842.2, color: '#34d399' }, 
  { name: 'COCOMO', val: 14021.4, color: '#ef4444' }, 
];

export default function Dashboard() {
  const [formData, setFormData] = useState({
    extInputs: '15',
    extOutputs: '5',
    extInquiries: '10',
    intLogFiles: '2',
    extInterfaces: '3'
  });
  
  const [isGenerating, setIsGenerating] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [aiEstimate, setAiEstimate] = useState<string | null>(null);
  const [cocomoEstimate, setCocomoEstimate] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setShowResults(false);
    setApiError(null);
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          extInputs: Number(formData.extInputs) || 0,
          extOutputs: Number(formData.extOutputs) || 0,
          extInquiries: Number(formData.extInquiries) || 0,
          intLogFiles: Number(formData.intLogFiles) || 0,
          extInterfaces: Number(formData.extInterfaces) || 0
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to connect to ML Backend API');
      }

      const data = await response.json();
      
      if (data.error) {
        setApiError(data.error);
      }
      
      setAiEstimate(data.ai_estimate.toLocaleString());
      setCocomoEstimate(data.cocomo_estimate.toLocaleString());
      
    } catch (err: any) {
      console.error(err);
      setApiError("Backend not running. Please start the FastAPI server on port 8000.");
    } finally {
      setIsGenerating(false);
      setShowResults(true);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-500/10 p-2 rounded-lg">
              <BrainCircuit className="w-6 h-6 text-blue-500" />
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
              Hybrid Ensemble Cost Estimator
            </h1>
          </div>
          <div className="flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-medium text-emerald-400 tracking-wide">v3.0 - ML Stacking</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Estimator Form */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-xl overflow-hidden">
              <div className="p-6 border-b border-slate-700 bg-slate-800/50">
                <div className="flex items-center space-x-2">
                  <FileCode className="w-5 h-5 text-blue-400" />
                  <h2 className="text-lg font-semibold text-slate-200">FPA Parameters</h2>
                </div>
                <p className="text-sm text-slate-400 mt-1">Provide standard Function Point counts.</p>
              </div>
              
              <form onSubmit={handleGenerate} className="p-6 space-y-5">
                <div className="space-y-4">
                  {/* External Inputs */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-0.5 flex items-center justify-between">
                      External Inputs
                      <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded">Wt: 4</span>
                    </label>
                    <p className="text-xs text-slate-500 mb-2">Forms, Data Entry & User Interactions</p>
                    <input 
                      type="number" 
                      name="extInputs"
                      value={formData.extInputs}
                      required
                      min="0"
                      onChange={handleInputChange}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                      placeholder="e.g. 15"
                    />
                  </div>
                  
                  {/* External Outputs */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-0.5 flex items-center justify-between">
                      External Outputs
                      <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded">Wt: 5</span>
                    </label>
                    <p className="text-xs text-slate-500 mb-2">Reports, Dashboards & Export files</p>
                    <input 
                      type="number" 
                      name="extOutputs"
                      value={formData.extOutputs}
                      required
                      min="0"
                      onChange={handleInputChange}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                      placeholder="e.g. 5"
                    />
                  </div>

                  {/* External Inquiries */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-0.5 flex items-center justify-between">
                      External Inquiries
                      <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded">Wt: 4</span>
                    </label>
                    <p className="text-xs text-slate-500 mb-2">Simple Queries, Fetches, Search inputs</p>
                    <input 
                      type="number" 
                      name="extInquiries"
                      value={formData.extInquiries}
                      required
                      min="0"
                      onChange={handleInputChange}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                      placeholder="e.g. 10"
                    />
                  </div>

                  {/* Internal Logical Files */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-0.5 flex items-center justify-between">
                      Internal Logical Files
                      <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded">Wt: 10</span>
                    </label>
                    <p className="text-xs text-slate-500 mb-2">Internal Database Tables & Data Structures</p>
                    <input 
                      type="number" 
                      name="intLogFiles"
                      value={formData.intLogFiles}
                      required
                      min="0"
                      onChange={handleInputChange}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                      placeholder="e.g. 2"
                    />
                  </div>

                  {/* External Interfaces */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-0.5 flex items-center justify-between">
                      External Interface Files
                      <span className="text-xs bg-slate-700 text-slate-400 px-2 py-0.5 rounded">Wt: 7</span>
                    </label>
                    <p className="text-xs text-slate-500 mb-2">Usage of APIs & 3rd Party Integrations</p>
                    <input 
                      type="number" 
                      name="extInterfaces"
                      value={formData.extInterfaces}
                      required
                      min="0"
                      onChange={handleInputChange}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                      placeholder="e.g. 3"
                    />
                  </div>
                </div>

                <div className="pt-2">
                  <button 
                    type="submit"
                    disabled={isGenerating}
                    className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white font-medium py-3 rounded-xl transition-all hover:shadow-[0_0_20px_rgba(59,130,246,0.4)] disabled:opacity-70 disabled:cursor-not-allowed"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Querying ML API...</span>
                      </>
                    ) : (
                      <>
                        <Activity className="w-5 h-5" />
                        <span>Generate Estimate</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Right Column: Results Dashboard */}
          <div className="lg:col-span-8 space-y-6">
            
            {/* KPI Section */}
            <div className="bg-slate-800 rounded-2xl border border-slate-700 p-8 shadow-xl relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-8 opacity-5">
                <Server className="w-32 h-32" />
              </div>
              <div className="relative z-10">
                <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-6">Estimated Project Effort Showcase</h3>
                
                {apiError && (
                  <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex flex-row items-start space-x-3">
                    <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-red-300 leading-relaxed">{apiError}</p>
                  </div>
                )}

                {showResults && !apiError ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    {/* New AI Prediction */}
                    <div className="bg-slate-900/50 rounded-xl p-6 border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.1)] relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500"></div>
                      <h4 className="text-emerald-400 font-semibold mb-2 flex items-center">
                        <BrainCircuit className="w-4 h-4 mr-2" />
                        Smart Hybrid Stacking AI
                      </h4>
                      <div className="flex items-baseline space-x-2">
                        <span className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-400 tracking-tight">
                          {aiEstimate}
                        </span>
                        <span className="text-xl text-slate-400 font-medium">Hours</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">Powered by Stacking Regressor Meta-Model</p>
                    </div>

                    {/* Old COCOMO Prediction */}
                    <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700 relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-1 h-full bg-slate-600"></div>
                      <h4 className="text-slate-400 font-semibold mb-2 flex items-center">
                        <Calculator className="w-4 h-4 mr-2" />
                        Traditional Old Model
                      </h4>
                      <div className="flex items-baseline space-x-2">
                        <span className="text-4xl font-extrabold text-slate-300 tracking-tight">
                          {cocomoEstimate}
                        </span>
                        <span className="text-xl text-slate-500 font-medium">Hours</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">Organic COCOMO via Unadjusted FPs</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                       <div className="bg-slate-900/30 rounded-xl p-6 border border-slate-700/50 border-dashed">
                          <h4 className="text-slate-500 font-medium mb-2">Smart Hybrid Machine Learning</h4>
                          <span className="text-4xl font-bold text-slate-600">--</span>
                       </div>
                       <div className="bg-slate-900/30 rounded-xl p-6 border border-slate-700/50 border-dashed">
                          <h4 className="text-slate-500 font-medium mb-2">Traditional Old Model</h4>
                          <span className="text-4xl font-bold text-slate-600">--</span>
                       </div>
                    </div>
                    <p className="text-sm text-slate-500 mt-2">Fill out the FPA counts to trigger the Python Backend API.</p>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Leaderboard */}
              <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-xl overflow-hidden flex flex-col">
                <div className="p-5 border-b border-slate-700 flex items-center space-x-2">
                  <Clock className="w-5 h-5 text-emerald-400" />
                  <h3 className="font-semibold text-slate-200">Historical Benchmarks</h3>
                </div>
                <div className="p-5 flex-1">
                  <div className="space-y-4">
                    {[
                      { name: 'Smart Hybrid (Stacking)', score: '198.5', bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/20', highlight: true },
                      { name: 'Random Forest', score: '214.9', bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/20' },
                      { name: 'Deep Learning', score: '1842.2', bg: 'bg-slate-700/50', text: 'text-slate-300', border: 'border-slate-700' },
                      { name: 'Traditional COCOMO', score: '14021.4', bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/20' }
                    ].map((model, i) => (
                      <div key={i} className={`flex items-center justify-between p-3 rounded-lg border ${model.border} ${model.highlight ? 'bg-slate-700/30' : 'bg-slate-900/50'}`}>
                        <span className={`text-sm font-medium ${model.text}`}>{model.name}</span>
                        <div className={`px-2.5 py-1 rounded text-xs font-bold tracking-wider ${model.bg} ${model.text}`}>
                          {model.score}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Chart */}
              <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-xl overflow-hidden flex flex-col">
                <div className="p-5 border-b border-slate-700 flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  <h3 className="font-semibold text-slate-200">Error Comparison</h3>
                </div>
                <div className="p-5 w-full" style={{ height: 250 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={MAE_DATA} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                      <XAxis 
                        dataKey="name" 
                        stroke="#94a3b8" 
                        fontSize={11} 
                        tickLine={false}
                        axisLine={false}
                        tick={{fill: '#94a3b8'}}
                      />
                      <YAxis 
                        stroke="#94a3b8" 
                        fontSize={11}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value}
                      />
                      <RechartsTooltip 
                        cursor={{fill: '#1e293b'}}
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f1f5f9', fontSize: '12px' }}
                        itemStyle={{ color: '#f1f5f9' }}
                      />
                      <Bar dataKey="val" radius={[4, 4, 0, 0]}>
                        {MAE_DATA.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

            </div>
          </div>
          
        </div>
      </main>
    </div>
  );
}
