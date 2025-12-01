(function () {
  function fmtSec(s) {
    s = Math.max(0, Math.floor(s || 0));
    var h = Math.floor(s / 3600),
      m = Math.floor((s % 3600) / 60),
      sec = s % 60;
    var parts = [];
    if (h > 0) parts.push(h + \" h\");
    if (m > 0) parts.push(m + \" min\");
    if (h === 0 && m === 0) parts.push(sec + \" s\");
    return parts.join(\" \");
  }
  async function loadPred() {
    try {
      const r = await fetch(\"/api/prediction\", { cache: \"no-store\" });
      if (!r.ok) throw new Error(\"HTTP \" + r.status);
      const j = await r.json();
      document.getElementById(\"pred-now\").textContent = j.now_status || \"—\";
      document.getElementById(\"pred-current\").textContent = fmtSec(j.current_run_seconds);
      document.getElementById(\"pred-median\").textContent = fmtSec(j.typical_online_run_seconds_median);
      document.getElementById(\"pred-mean\").textContent = fmtSec(j.typical_online_run_seconds_mean);
      if (j.now_status === \"ONLINE\") {
        document.getElementById(\"pred-remaining\").textContent = \"≈ \" + fmtSec(j.predicted_remaining_online_seconds);
      } else {
        document.getElementById(\"pred-remaining\").textContent = \"—\";
      }
      document.getElementById(\"pred-updated\").textContent = new Date().toLocaleTimeString();
    } catch (e) {
      document.getElementById(\"pred-now\").textContent = \"błąd\";
    }
  }
  document.addEventListener(\"DOMContentLoaded\", function () {
    const btn = document.getElementById(\"pred-refresh\");
    if (btn) btn.addEventListener(\"click\", loadPred);
    loadPred();
    setInterval(loadPred, 30000);
  });
})();