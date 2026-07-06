// Demo storefront configuration -- in a real install this would be the two
// lines a brand pastes from their CartIQ dashboard's Settings tab.
//
// Copy this file to config.js (gitignored -- never commit real API keys)
// and fill in the key printed by `python ml/seed_demo_data.py`.
window.CARTIQ_CONFIG = {
  apiBaseUrl: "http://127.0.0.1:8000",
  apiKey: "PASTE_YOUR_DEMO_API_KEY_HERE",
};
