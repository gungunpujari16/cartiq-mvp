/**
 * CartIQ JS Snippet (PRD Feature 1 / TRD "Capture Layer").
 *
 * - Async, non-blocking, zero PII (session id + behavioural signals only).
 * - Batches events and flushes every 500ms (TRD S1).
 * - Checks a consent flag before activating (TRD S5 / GDPR Article 7).
 * - After cart-relevant events, requests a score and -- if the shopper is
 *   flagged low-intent with a real cart value -- a discount (PRD Feature 3).
 *
 * Host pages call window.CartIQ.track(eventType, {...}) for events the
 * snippet can't observe passively (add_to_cart, purchase, etc). Everything
 * else (page_view, scroll depth, exit intent, time on page) is automatic.
 */
(function () {
  const CFG = window.CARTIQ_CONFIG || {};
  const API_BASE = CFG.apiBaseUrl || "http://127.0.0.1:8000";
  const API_KEY = CFG.apiKey || "";
  const FLUSH_INTERVAL_MS = 500;
  const CONSENT_KEY = "cartiq_consent";
  const SESSION_KEY = "cartiq_session_id";

  let queue = [];
  let pageEnteredAt = performance.now();
  let scrollSamples = [];
  let currentPageUrl = location.pathname;

  function uuid() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function getSessionId() {
    let id = sessionStorage.getItem(SESSION_KEY);
    if (!id) {
      id = uuid();
      sessionStorage.setItem(SESSION_KEY, id);
    }
    return id;
  }

  function hasConsent() {
    return localStorage.getItem(CONSENT_KEY) === "1";
  }

  function deviceType() {
    const w = window.innerWidth;
    if (w < 768) return "mobile";
    if (w < 1024) return "tablet";
    return "desktop";
  }

  function trafficSource() {
    const params = new URLSearchParams(location.search);
    const utm = params.get("utm_source");
    if (utm) return utm.toLowerCase();
    if (!document.referrer) return "direct";
    try {
      const host = new URL(document.referrer).hostname;
      if (/google|bing/i.test(host)) return "organic";
      if (/facebook|instagram|tiktok/i.test(host)) return "social media";
      return "referral";
    } catch {
      return "direct";
    }
  }

  function avgScroll() {
    if (!scrollSamples.length) return 0;
    return scrollSamples.reduce((a, b) => a + b, 0) / scrollSamples.length;
  }

  function enqueue(eventType, extra) {
    if (!hasConsent()) return;
    const cart = (window.CartIQStore && window.CartIQStore.getCart()) || { value: 0, items: 0 };
    queue.push({
      session_id: getSessionId(),
      event_type: eventType,
      timestamp: new Date().toISOString(),
      device_type: deviceType(),
      page_url: currentPageUrl,
      traffic_source: trafficSource(),
      cart_value: cart.value,
      cart_items: cart.items,
      scroll_depth: avgScroll(),
      time_on_page: Math.round(performance.now() - pageEnteredAt),
      return_customer: localStorage.getItem("cartiq_returning") === "1",
      product_category: (window.CartIQStore && window.CartIQStore.getCategory()) || null,
      order_value: (extra && extra.order_value) || null,
      ...extra,
    });
  }

  async function flush() {
    if (!hasConsent() || queue.length === 0) return;
    const batch = queue;
    queue = [];
    try {
      await fetch(`${API_BASE}/v1/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CartIQ-Key": API_KEY },
        body: JSON.stringify({ events: batch }),
      });
    } catch (err) {
      // Non-blocking by design: a failed beacon never breaks the host page.
      console.warn("[CartIQ] event flush failed", err);
    }
  }

  async function requestScoreAndDiscount() {
    if (!hasConsent()) return;
    const session_id = getSessionId();
    try {
      const scoreRes = await fetch(`${API_BASE}/v1/score/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CartIQ-Key": API_KEY },
        body: JSON.stringify({ session_id }),
      });
      if (!scoreRes.ok) return;
      const score = await scoreRes.json();
      document.dispatchEvent(new CustomEvent("cartiq:score", { detail: score }));

      const discRes = await fetch(`${API_BASE}/v1/discounts/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CartIQ-Key": API_KEY },
        body: JSON.stringify({ session_id }),
      });
      if (!discRes.ok) return;
      const discount = await discRes.json();
      if (discount.issued) {
        document.dispatchEvent(new CustomEvent("cartiq:discount", { detail: discount }));
      }
    } catch (err) {
      console.warn("[CartIQ] scoring/discount request failed", err);
    }
  }

  function initConsentBanner() {
    if (hasConsent()) return;
    const banner = document.createElement("div");
    banner.id = "cartiq-consent-banner";
    banner.innerHTML = `
      <span>This site uses CartIQ to personalise your shopping experience. No personal data is collected.</span>
      <button id="cartiq-consent-accept">Accept</button>`;
    document.body.appendChild(banner);
    document.getElementById("cartiq-consent-accept").addEventListener("click", () => {
      localStorage.setItem(CONSENT_KEY, "1");
      banner.remove();
      enqueue("page_view");
    });
  }

  function initPassiveTracking() {
    window.addEventListener("scroll", () => {
      const scrolled = window.scrollY;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      scrollSamples.push(max > 0 ? Math.min(1, scrolled / max) : 0);
    });

    document.addEventListener("mouseleave", (e) => {
      if (e.clientY <= 0) enqueue("exit_intent");
    });

    window.addEventListener("beforeunload", () => {
      enqueue("page_view"); // final snapshot with accumulated time_on_page/scroll_depth
      navigator.sendBeacon &&
        navigator.sendBeacon(
          `${API_BASE}/v1/events`,
          new Blob([JSON.stringify({ events: queue })], { type: "application/json" })
        );
    });
  }

  window.CartIQ = {
    // Returns a promise so callers that fire multiple events in a row (e.g.
    // payment_attempt immediately followed by purchase) can await each one
    // and guarantee the requests land at the API in the order they happened
    // -- otherwise two near-simultaneous fetches can race and a later event's
    // stage-bump can overwrite an earlier one's result server-side.
    async track(eventType, extra) {
      enqueue(eventType, extra);
      if (["add_to_cart", "remove_from_cart", "checkout_start", "checkout_step", "payment_attempt", "purchase"].includes(eventType)) {
        await flush();
        await requestScoreAndDiscount();
      }
    },
  };

  enqueue("page_view");
  initConsentBanner();
  initPassiveTracking();
  setInterval(flush, FLUSH_INTERVAL_MS);
})();
