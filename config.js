// Demo storefront configuration -- in a real install this would be the two
// lines a brand pastes from their CartIQ dashboard's Settings tab.
//
// This key is intentionally public here -- same design as a Stripe
// publishable key or a GA tracking ID. It only grants access to this
// sandboxed "CartIQ Demo Store" brand's own data (see PRD Feature 1:
// the snippet's API key is meant to live in client-side JS).
window.CARTIQ_CONFIG = {
  apiBaseUrl: "https://cartiq-backend-3ubn.onrender.com",
  apiKey: "ciq_6KkSPlvm7JSj5VCCLFe3zeEE4XWpJD2E9KXkXdb_Zqk",
};
