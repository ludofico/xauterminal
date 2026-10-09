// Proxy /api to the FastAPI backend so the browser talks to one origin.
const API = process.env.API_URL || "http://127.0.0.1:8000";
export default { async rewrites() { return [{ source: "/api/:path*", destination: `${API}/api/:path*` }]; } };
