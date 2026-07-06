/**
 * Purely cosmetic: renders the live score badge and discount toast so the
 * end-to-end loop (browser -> CartIQ API -> score -> discount) is visible
 * while clicking through the demo, instead of only in the Network tab.
 */
(function () {
  function ensureBadge() {
    let badge = document.getElementById("cartiq-score-badge");
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "cartiq-score-badge";
      badge.className = "cartiq-score-badge hidden";
      document.body.appendChild(badge);
    }
    return badge;
  }

  document.addEventListener("cartiq:score", (e) => {
    const { score, segment } = e.detail;
    const badge = ensureBadge();
    badge.textContent = `CartIQ score: ${score.toFixed(0)} · ${segment}`;
    badge.classList.remove("hidden");
    badge.dataset.segment = segment.replace(/\s+/g, "-").toLowerCase();
  });

  document.addEventListener("cartiq:discount", (e) => {
    const { code, pct } = e.detail;
    const toast = document.createElement("div");
    toast.className = "cartiq-discount-toast";
    toast.innerHTML = `CartIQ noticed you might leave &mdash; here's <strong>${pct}% off</strong>: <code>${code}</code>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 9000);
  });
})();
