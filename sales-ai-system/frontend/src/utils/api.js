/**
 * API Client — centralized Axios instance with JWT auth interceptors.
 * All API calls go through this module.
 */

import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ── Request interceptor: attach JWT ──────────────────────────────────────────
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("salesai_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── Response interceptor: handle 401 ─────────────────────────────────────────
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("salesai_token");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

// ── Auth ──────────────────────────────────────────────────────────────────────
export const authAPI = {
  login: (email, password) =>
    api.post("/auth/login", new URLSearchParams({ username: email, password }), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }),
  signup: (data) => api.post("/auth/signup", data),
  me: () => api.get("/auth/me"),
};

// ── Dashboard ─────────────────────────────────────────────────────────────────
export const dashboardAPI = {
  kpis:            () => api.get("/dashboard/kpis"),
  revenueTrend:    () => api.get("/dashboard/revenue-trend"),
  regionPerf:      () => api.get("/dashboard/region-performance"),
};

// ── Forecast ──────────────────────────────────────────────────────────────────
export const forecastAPI = {
  demo:        (days = 30) => api.get(`/forecast/demo?days=${days}`),
  train:       (file, days = 30) => {
    const form = new FormData();
    if (file) form.append("file", file);
    return api.post(`/forecast/train?days=${days}`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  topProducts: () => api.get("/forecast/top-products"),
};

// ── NLP ───────────────────────────────────────────────────────────────────────
export const nlpAPI = {
  demo:        () => api.get("/nlp/demo"),
  analyze:     (reviews) => api.post("/nlp/analyze", { reviews }),
  analyzeCSV:  (file) => {
    const form = new FormData();
    form.append("file", file);
    return api.post("/nlp/analyze-csv", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};

// ── Inventory ─────────────────────────────────────────────────────────────────
export const inventoryAPI = {
  status:   () => api.get("/inventory/status"),
  deadStock:() => api.get("/inventory/dead-stock"),
};

// ── Pricing ───────────────────────────────────────────────────────────────────
export const pricingAPI = {
  overview: () => api.get("/pricing/overview"),
  suggest:  (data) => api.post("/pricing/suggest", data),
};

// ── Chatbot ───────────────────────────────────────────────────────────────────
export const chatbotAPI = {
  ask: (message, history = []) =>
    api.post("/chatbot/ask", { message, history }),
};

// ── Fraud ─────────────────────────────────────────────────────────────────────
export const fraudAPI = {
  alerts:    () => api.get("/fraud/alerts"),
  riskScore: () => api.get("/fraud/risk-score"),
};

// ── Reports ───────────────────────────────────────────────────────────────────
export const reportsAPI = {
  summary:      () => api.get("/reports/summary"),
  downloadPDF:  () =>
    api.get("/reports/download/pdf", { responseType: "blob" }),
  downloadExcel:() =>
    api.get("/reports/download/excel", { responseType: "blob" }),
};

export default api;
