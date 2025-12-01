import os, io, threading, time, platform, re, subprocess, shutil, json, sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, send_file, request, jsonify, Response
from zoneinfo import ZoneInfo

URL = "https://tera-europe-classic.de/stats.php?lang=en"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "monitor.db")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

APP_TZ = os.getenv("APP_TZ", "Europe/Warsaw")
def now_local(): return datetime.now(ZoneInfo(APP_TZ))
def now_utc():    return datetime.now(timezone.utc)

# Home Assistant
HA_BASE_URL = os.getenv("HA_BASE_URL", "").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
HA_VERIFY_SSL = os.getenv("HA_VERIFY_SSL", "1") == "1"
HA_NOTIFY_SERVICES = [s.strip() for s in os.getenv("HA_NOTIFY_SERVICES", "").split(",") if s.strip()]
ENABLE_STATUS_NOTIFY = os.getenv("ENABLE_STATUS_NOTIFY", "1") == "1"
ENABLE_PING_NOTIFY = os.getenv("ENABLE_PING_NOTIFY", "1") == "1"
PING_ALERT_MS = int(os.getenv("PING_ALERT_MS", "300"))
PING_ALERT_COOLDOWN = int(os.getenv("PING_ALERT_COOLDOWN", "600"))

SAMPLE_SECONDS = int(os.getenv("SAMPLE_SECONDS", "5"))
GLITCH_SECONDS = int(os.getenv("GLITCH_SECONDS", "15"))

ONLINE_GREEN = 70
ONLINE_YELLOW_MIN = 31
ONLINE_YELLOW_MAX = 69

app = Flask(__name__)

translations = {
    "PL": {"status":"Status serwera","players":"Graczy online",
           "ping":"Ping","min_ping":"Minimalny ping","max_ping":"Maksymalny ping","avg_ping":"Średni ping","plot_online":"ONLINE"},
    "EN": {"status":"Server status","players":"Players online",
           "ping":"Ping","min_ping":"Minimum ping","max_ping":"Maximum ping","avg_ping":"Average ping","plot_online":"ONLINE"},
}

# -------------- ustawienia --------------
DEFAULT_CFG = {
    "mute_enabled": False,
    "quiet_start": "01:00",
    "quiet_end": "06:00",
    "ping_alert_ms": PING_ALERT_MS,
    "glitch_seconds": GLITCH_SECONDS
}
def load_cfg():
    try:
        with open(CONFIG_PATH,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: return DEFAULT_CFG.copy()
def save_cfg(cfg:dict):
    with open(CONFIG_PATH,"w",encoding="utf-8") as f: json.dump(cfg, f, ensure_ascii=False, indent=2)

# -------------- baza --------------
def db():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
def init_db():
    conn = db(); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS samples_players(
      ts_utc TEXT NOT NULL, players INTEGER NOT NULL)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_players_ts ON samples_players(ts_utc)")
    c.execute("""CREATE TABLE IF NOT EXISTS samples_ping(
      ts_utc TEXT NOT NULL, ping_ms INTEGER NOT NULL)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ping_ts ON samples_ping(ts_utc)")
    c.execute("""CREATE TABLE IF NOT EXISTS events(
      ts_utc TEXT NOT NULL, kind TEXT NOT NULL, message TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc)")
    conn.commit(); conn.close()
def insert_player_sample(v:int):
    conn=db(); conn.execute("INSERT INTO samples_players VALUES(?,?)",(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),int(v))); conn.commit(); conn.close()
def insert_ping_sample(v:int):
    conn=db(); conn.execute("INSERT INTO samples_ping VALUES(?,?)",(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),int(v))); conn.commit(); conn.close()
def insert_event(kind,msg):
    conn=db(); conn.execute("INSERT INTO events VALUES(?,?,?)",(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),kind,msg)); conn.commit(); conn.close()

def read_df_players(hours=48):
    conn=db(); cutoff=(now_utc()-timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    df=pd.read_sql_query("SELECT ts_utc,players FROM samples_players WHERE ts_utc>=? ORDER BY ts_utc",conn,params=(cutoff,)); conn.close()
    if df.empty: return df
    df["Czas"]=pd.to_datetime(df["ts_utc"],utc=True).dt.tz_convert(APP_TZ); df["Graczy"]=pd.to_numeric(df["players"],errors="coerce")
    return df[["Czas","Graczy"]].dropna()
def read_df_ping(hours=48):
    conn=db(); cutoff=(now_utc()-timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    df=pd.read_sql_query("SELECT ts_utc,ping_ms FROM samples_ping WHERE ts_utc>=? ORDER BY ts_utc",conn,params=(cutoff,)); conn.close()
    if df.empty: return df
    df["Czas"]=pd.to_datetime(df["ts_utc"],utc=True).dt.tz_convert(APP_TZ); df["Ping"]=pd.to_numeric(df["ping_ms"],errors="coerce")
    return df[["Czas","Ping"]].dropna()
def read_events(limit=50):
    conn=db(); df=pd.read_sql_query("SELECT ts_utc,kind,message FROM events ORDER BY ts_utc DESC LIMIT ?",conn,params=(limit,)); conn.close()
    out=[]
    for _,r in df.iterrows():
        out.append({"ts": pd.to_datetime(r["ts_utc"],utc=True).tz_convert(APP_TZ).strftime("%Y-%m-%d %H:%M:%S"), "kind": r["kind"], "msg": r["message"]})
    return out

# -------------- scraping/ping --------------
def get_online_players():
    try:
        r = requests.get(URL, timeout=10, headers={"User-Agent":"Mozilla/5.0 (TeraMonitor)"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        node = soup.find("div", class_="stat-number")
        text = node.get_text(" ").strip() if node else soup.get_text(" ").strip()
        m = re.search(r"(\d[\d\s,\.]*)", text)
        if not m: return None
        num = re.sub(r"[^\d]", "", m.group(1))
        return int(num) if num else None
    except Exception as e:
        print(f"[BŁĄD] get_online_players(): {e}")
        return None
def measure_ping():
    host="103.249.70.12"
    try:
        is_win = platform.system().lower().startswith("win")
        count_flag="-n" if is_win else "-c"
        ping_bin = shutil.which("ping")
        if ping_bin:
            cp = subprocess.run([ping_bin, host, count_flag, "1"], capture_output=True, text=True, timeout=4)
            out = cp.stdout.replace("\r","")
            m = re.search(r"(Average|Średni)\s*=\s*(\d+)\s*ms", out, flags=re.IGNORECASE)
            if m: return int(m.group(2))
            m = re.search(r"time[=<]\s*([\d\.]+)\s*ms", out, flags=re.IGNORECASE)
            if m: return max(1, int(round(float(m.group(1)))))
    except Exception as e:
        print(f"[PING ICMP] {e}")
    try:
        import socket, time as _t
        t0=_t.perf_counter(); s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(3); s.connect((host,443)); s.close()
        return max(1, int(round((_t.perf_counter()-t0)*1000.0)))
    except Exception as e:
        print(f"[PING TCP] {e}")
        return None

# -------------- status/kolory --------------
def compute_status_color(players: int | None):
    if players is None or players == 0: return ("OFFLINE","red")
    if players >= ONLINE_GREEN: return ("ONLINE","limegreen")
    if ONLINE_YELLOW_MIN <= players <= ONLINE_YELLOW_MAX: return ("ONLINE","#ffd54f")
    return ("ONLINE","#f39c12")

# -------------- segmenty (glitch-resistant) --------------
def runs_from_series(times, status):
    if not status: return []
    runs=[]; cur=status[0]; st=times[0]
    for i in range(1,len(status)):
        if status[i]!=cur:
            runs.append([cur, st, times[i-1]])
            cur=status[i]; st=times[i]
    runs.append([cur, st, times[-1]])
    return runs
def merge_short_runs(runs, threshold_seconds):
    if not runs: return runs
    out=[]
    for i,(lab,st,en) in enumerate(runs):
        dur=(en-st).total_seconds()
        if dur < threshold_seconds:
            if out:
                out[-1][2]=en
            elif i+1 < len(runs):
                nxt_lab,nxt_st,nxt_en=runs[i+1]
                runs[i+1]=[nxt_lab, st, nxt_en]
            else:
                out.append([lab,st,en])
        else:
            out.append([lab,st,en])
    comp=[]
    for lab,st,en in out:
        if comp and comp[-1][0]==lab: comp[-1][2]=en
        else: comp.append([lab,st,en])
    return comp
def analyze_from_db():
    res={"players":None,"status_text":"OFFLINE","status_color":"red",
         "uptime_seconds":0,"uptime_counting":False,
         "downtime_seconds":0,"downtime_counting":False,
         "record_uptime_seconds":0,"record_min_uptime_seconds":0,
         "record_downtime_seconds":0,"record_min_downtime_seconds":0,
         "peak_players":None,"last_change_local":None}
    df=read_df_players(hours=720)
    if df.empty: return res
    players_now=int(df["Graczy"].iloc[-1])
    res["players"]=players_now
    res["peak_players"]=int(df["Graczy"].max())
    st_text,st_color=compute_status_color(players_now)
    res["status_text"],res["status_color"]=st_text,st_color

    cfg=load_cfg(); thr=int(cfg.get("glitch_seconds", GLITCH_SECONDS))
    status=(df["Graczy"]>0).astype(int).tolist(); times=df["Czas"].tolist()
    runs=merge_short_runs(runs_from_series(times,status), thr)
    if not runs:
        now=now_local(); lab=1 if players_now>0 else 0
        res["last_change_local"]=now.strftime("%d.%m.%Y %H:%M:%S")
        if lab: res["uptime_counting"]=True
        else: res["downtime_counting"]=True
        return res

    upt=[(en-st).total_seconds() for lab,st,en in runs if lab==1]
    dwn=[(en-st).total_seconds() for lab,st,en in runs if lab==0]
    if upt:
        res["record_uptime_seconds"]=int(max(upt))
        m=[x for x in upt if x>0]; res["record_min_uptime_seconds"]=int(min(m)) if m else 0
    if dwn:
        res["record_downtime_seconds"]=int(max(dwn))
        m=[x for x in dwn if x>0]; res["record_min_downtime_seconds"]=int(min(m)) if m else 0

    lab,st,en=runs[-1]
    now=now_local(); dur=int((now-st).total_seconds()); dur=max(0,dur)
    res["last_change_local"]=st.strftime("%d.%m.%Y %H:%M:%S")
    if lab==1:
        res["uptime_counting"]=True; res["uptime_seconds"]=dur
    else:
        res["downtime_counting"]=True; res["downtime_seconds"]=dur
    return res

# -------------- SLA / forecast --------------
def sla_between(start:datetime, end:datetime):
    df=read_df_players(hours=31*24)
    if df.empty: return (0,0,0.0)
    df=df[(df["Czas"]>=start)&(df["Czas"]<=end)]
    if df.empty: return (0,0,0.0)
    status=(df["Graczy"]>0).astype(int).to_numpy()
    times=df["Czas"].to_numpy()
    secs=np.diff(times.astype('datetime64[s]')).astype(int)
    secs=np.append(secs, secs[-1] if len(secs)>0 else SAMPLE_SECONDS)
    online=int(np.sum(secs*status)); total=int(np.sum(secs)); offline=max(0,total-online)
    sla=100.0*online/total if total>0 else 0.0
    return (online, offline, sla)

# -------------- powiadomienia / mute --------------
def is_quiet_now(cfg):
    if not cfg.get("mute_enabled"): return False
    try:
        qs=cfg.get("quiet_start","01:00"); qe=cfg.get("quiet_end","06:00")
        nowt=now_local().time()
        qs_h,qs_m=map(int,qs.split(":")); qe_h,qe_m=map(int,qe.split(":"))
        from datetime import time as T
        s=T(qs_h,qs_m); e=T(qe_h,qe_m)
        return (s<=nowt<e) if s<e else not (e<=nowt<s)
    except: return False
def ha_notify(title, message):
    if not (HA_BASE_URL and HA_TOKEN and HA_NOTIFY_SERVICES): return
    if is_quiet_now(load_cfg()):
        insert_event("MUTED: "+title, message); return
    headers={"Authorization":f"Bearer {HA_TOKEN}","Content-Type":"application/json"}
    payload={"title":title,"message":message,"data":{"ttl":0,"priority":"high"}}
    for svc in HA_NOTIFY_SERVICES:
        try:
            url=f"{HA_BASE_URL}/api/services/{svc.replace('.', '/')}"
            requests.post(url, headers=headers, json=payload, timeout=5, verify=HA_VERIFY_SSL)
        except Exception as e:
            print(f"[HA] notify failed: {e}")

# -------------- worker --------------
_last_status_lab=None
_last_ping_alert_ts=0.0
def background_worker():
    global _last_status_lab, _last_ping_alert_ts
    next_players=0; next_ping=0
    while True:
        now_ts=time.time()
        if now_ts>=next_players:
            p=get_online_players()
            if p is not None:
                insert_player_sample(p)
                # status zmiana
                summary=analyze_from_db()
                cur_lab = 1 if summary["status_text"]=="ONLINE" else 0
                if _last_status_lab is None:
                    _last_status_lab=cur_lab
                elif cur_lab!=_last_status_lab:
                    insert_event("STATUS", summary["status_text"])
                    if ENABLE_STATUS_NOTIFY:
                        ha_notify(f"TERA: status {summary['status_text']}", f"Gracze: {p}")
                    _last_status_lab=cur_lab
            next_players=now_ts+SAMPLE_SECONDS

        if now_ts>=next_ping:
            ping=measure_ping()
            if ping is not None:
                insert_ping_sample(ping)
                thr=int(load_cfg().get("ping_alert_ms", PING_ALERT_MS))
                if ENABLE_PING_NOTIFY and ping>=thr:
                    if (now_ts-_last_ping_alert_ts)>=PING_ALERT_COOLDOWN:
                        insert_event("PING_ALERT", f"{ping} ms")
                        ha_notify(f"TERA: wysoki ping {ping} ms", f"Próg: {thr} ms")
                        _last_ping_alert_ts=now_ts
            next_ping=now_ts+SAMPLE_SECONDS
        time.sleep(1)

# -------------- wykresy i sparki --------------
def make_time_axis(ax, minutes):
    tz = ZoneInfo(APP_TZ)
    step = 2 if minutes<=15 else 5 if minutes<=30 else 10 if minutes<=60 else 20 if minutes<=180 else 30 if minutes<=360 else 60
    now = now_local(); cutoff = now - timedelta(minutes=minutes)
    ax.set_xlim(cutoff, now)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=step, tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1, tz=tz))
    ax.tick_params(axis='x', labelrotation=45, colors="#e6e6e6")
    ax.tick_params(axis='y', colors="#e6e6e6")
def _render(fig):
    buf=io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0); plt.close(fig); return buf

@app.route("/")
def index():
    lang = request.args.get("lang","PL").upper()
    summary = analyze_from_db()
    ping_df = read_df_ping(hours=6)
    if ping_df.empty: min_ping=avg_ping=max_ping=rec_min=rec_max=None
    else:
        last = ping_df.tail(5)["Ping"]
        min_ping=int(last.min()); avg_ping=int(last.mean()); max_ping=int(last.max())
        allp=read_df_ping(hours=24*365); rec_min=int(allp["Ping"].min()); rec_max=int(allp["Ping"].max())
    cfg = load_cfg()
    class C: pass
    c=C(); c.mute_enabled=cfg.get("mute_enabled",False); c.quiet_start=cfg.get("quiet_start","01:00")
    c.quiet_end=cfg.get("quiet_end","06:00"); c.ping_alert_ms=cfg.get("ping_alert_ms",PING_ALERT_MS)
    c.glitch_seconds=cfg.get("glitch_seconds",GLITCH_SECONDS)
    t = translations.get(lang, translations["PL"])
    return render_template("index.html", t=t,
        status_text=summary["status_text"], status_color=summary["status_color"],
        players=summary["players"], peak_players=summary["peak_players"],
        min_ping=min_ping, avg_ping=avg_ping, max_ping=max_ping,
        record_min_ping=rec_min, record_max_ping=rec_max,
        lang=lang, cache_bust=int(time.time()),
        uptime_seconds=summary["uptime_seconds"], uptime_counting=summary["uptime_counting"],
        downtime_seconds=summary["downtime_seconds"], downtime_counting=summary["downtime_counting"],
        record_uptime_seconds=summary["record_uptime_seconds"],
        record_downtime_seconds=summary["record_downtime_seconds"],
        record_min_uptime_seconds=summary["record_min_uptime_seconds"],
        record_min_downtime_seconds=summary["record_min_downtime_seconds"],
        last_change_local=summary["last_change_local"],
        cfg=c)

@app.route("/api/summary")
def api_summary():
    summary = analyze_from_db()
    ping_df = read_df_ping(hours=6)
    if ping_df.empty: min_ping=avg_ping=max_ping=rec_min=rec_max=None
    else:
        last = ping_df.tail(5)["Ping"]
        min_ping=int(last.min()); avg_ping=int(last.mean()); max_ping=int(last.max())
        allp=read_df_ping(hours=24*365); rec_min=int(allp["Ping"].min()); rec_max=int(allp["Ping"].max())
    summary.update({"min_ping":min_ping,"avg_ping":avg_ping,"max_ping":max_ping,
                    "record_min_ping":rec_min,"record_max_ping":rec_max})
    return jsonify(summary)

@app.route("/api/events")
def api_events():
    limit=int(request.args.get("limit","50"))
    return jsonify(read_events(limit=limit))

@app.route("/plot_players/<int:minutes>")
def plot_players(minutes):
    df = read_df_players(hours=48)
    cutoff = now_local() - timedelta(minutes=minutes)
    df = df[df["Czas"] >= cutoff]
    plt.rcParams.update({"axes.facecolor":"#0f1115","figure.facecolor":"#0f1115"})
    fig, ax = plt.subplots(figsize=(8,3.6)); ax.set_facecolor("#0f1115")
    if df.empty:
        ax.text(0.5,0.5, "No data", color="#e6e6e6", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(df["Czas"], df["Graczy"], marker="o", linewidth=1.6, markersize=3.5, color="#5dade2")
    ax.axhline(70, linestyle="--", linewidth=1.0, color="#2ecc71", label="ONLINE (≥70)")
    ax.axhline(31, linestyle="--", linewidth=1.0, color="#ffd54f", label="ONLINE (31–69)")
    ax.axhline(1, linestyle="--", linewidth=1.0, color="#f39c12", label="ONLINE (1–30)")
    make_time_axis(ax, minutes)
    ax.set_xlabel("Time", color="#e6e6e6"); ax.set_ylabel("Players", color="#e6e6e6")
    ax.grid(True, alpha=0.25, color="#3a3f44")
    leg = ax.legend(); [tx.set_color("#e6e6e6") for tx in leg.get_texts()]
    return send_file(_render(fig), mimetype="image/png")

@app.route("/plot_ping/<int:minutes>")
def plot_ping(minutes):
    df = read_df_ping(hours=48)
    cutoff = now_local() - timedelta(minutes=minutes)
    df = df[df["Czas"] >= cutoff]
    plt.rcParams.update({"axes.facecolor":"#0f1115","figure.facecolor":"#0f1115"})
    fig, ax = plt.subplots(figsize=(8,3.6)); ax.set_facecolor("#0f1115")
    if df.empty:
        ax.text(0.5,0.5, "No data", color="#e6e6e6", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(df["Czas"], df["Ping"], linewidth=1.4, color="#a569bd")
    make_time_axis(ax, minutes)
    ax.set_xlabel("Time", color="#e6e6e6"); ax.set_ylabel("ms", color="#e6e6e6")
    ax.grid(True, alpha=0.25, color="#3a3f44")
    return send_file(_render(fig), mimetype="image/png")

# ---- Sparklines (ostatnie 60 min) ----
@app.route("/spark/players")
def spark_players():
    df = read_df_players(hours=2)
    cutoff = now_local()-timedelta(minutes=60)
    df=df[df["Czas"]>=cutoff]
    fig,ax=plt.subplots(figsize=(3.2,0.9)); ax.axis('off')
    if not df.empty: ax.plot(df["Czas"], df["Graczy"], linewidth=1.5)
    return send_file(_render(fig), mimetype="image/png")
@app.route("/spark/ping")
def spark_ping():
    df = read_df_ping(hours=2)
    cutoff = now_local()-timedelta(minutes=60)
    df=df[df["Czas"]>=cutoff]
    fig,ax=plt.subplots(figsize=(3.2,0.9)); ax.axis('off')
    if not df.empty: ax.plot(df["Czas"], df["Ping"], linewidth=1.5)
    return send_file(_render(fig), mimetype="image/png")

# ---- Ustawienia / Import / Eksport / Raport ----
@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    cfg = load_cfg()
    for k in ["mute_enabled","quiet_start","quiet_end","ping_alert_ms","glitch_seconds"]:
        if k in data: cfg[k]=data[k]
    save_cfg(cfg); return jsonify({"ok":True})

@app.route("/admin/import_csv")
def import_csv():
    pl_path=os.path.join(DATA_DIR,"log.csv"); pi_path=os.path.join(DATA_DIR,"ping_log.csv"); imp=0
    if os.path.exists(pl_path):
        df=pd.read_csv(pl_path); 
        if not df.empty:
            df["Czas"]=pd.to_datetime(df["Czas"], errors="coerce")
            conn=db(); c=conn.cursor()
            for _,r in df.dropna().iterrows():
                ts=pd.Timestamp(r["Czas"]).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO samples_players VALUES(?,?)",(ts,int(r["Graczy"]))); imp+=1
            conn.commit(); conn.close()
    if os.path.exists(pi_path):
        df=pd.read_csv(pi_path);
        if not df.empty:
            df["Czas"]=pd.to_datetime(df["Czas"], errors="coerce")
            conn=db(); c=conn.cursor()
            for _,r in df.dropna().iterrows():
                ts=pd.Timestamp(r["Czas"]).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO samples_ping VALUES(?,?)",(ts,int(r["Ping"]))); imp+=1
            conn.commit(); conn.close()
    insert_event("IMPORT", f"Zaimportowano: {imp}")
    return f"OK {imp}"

def _csv_between(table, col, date_from, date_to, header):
    conn=db()
    q="SELECT ts_utc,"+col+" FROM "+table+" WHERE 1=1"
    params=[]
    if date_from: q+=" AND ts_utc>=?"; params.append(date_from+" 00:00:00")
    if date_to: q+=" AND ts_utc<=?"; params.append(date_to+" 23:59:59")
    q+=" ORDER BY ts_utc"
    df=pd.read_sql_query(q,conn,params=params); conn.close()
    rows=[header]; rows+=df.values.tolist()
    csv="\n".join(",".join(map(str,r)) for r in rows)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition":f"attachment; filename={table}.csv"})

@app.route("/export/players.csv")
def export_players_csv():
    return _csv_between("samples_players", "players",
        request.args.get("from",""), request.args.get("to",""), ["ts_utc","players"])

@app.route("/export/ping.csv")
def export_ping_csv():
    return _csv_between("samples_ping", "ping_ms",
        request.args.get("from",""), request.args.get("to",""), ["ts_utc","ping_ms"])

@app.route("/export/db")
def export_db():
    with open(DB_PATH,"rb") as f:
        data=f.read()
    return Response(data, mimetype="application/octet-stream",
                    headers={"Content-Disposition":"attachment; filename=monitor.db"})

@app.route("/report.csv")
def report_csv():
    now = now_local()
    ranges = {
        "day": (now.replace(hour=0,minute=0,second=0,microsecond=0), now),
        "week": (now - timedelta(days=7), now),
        "month": (now - timedelta(days=30), now)
    }
    rows=[["range","online_sec","offline_sec","sla_percent"]]
    for k,(s,e) in ranges.items():
        on,off,sla=sla_between(s,e)
        rows.append([k,on,off,round(sla,3)])
    csv="\n".join(",".join(map(str,r)) for r in rows)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=sla_report.csv"})

@app.route("/notify/test")
def notify_test():
    ha_notify("TERA Monitor","Test push"); return "OK"

if __name__ == "__main__":
    init_db()
    if not os.path.exists(CONFIG_PATH):
        from json import dump; dump(DEFAULT_CFG, open(CONFIG_PATH,"w"), ensure_ascii=False)
    threading.Thread(target=background_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=9999)

# === PATCH: parameterized spark ping ===
@app.route("/spark/ping/<int:minutes>")
def spark_ping_minutes(minutes: int):
    df = read_df_ping(hours=max(2, (minutes//60)+1))
    cutoff = now_local()-timedelta(minutes=minutes)
    df = df[df["Czas"]>=cutoff]
    fig,ax=plt.subplots(figsize=(3.2,0.9)); ax.axis('off')
    if not df.empty: ax.plot(df["Czas"], df["Ping"], linewidth=1.5)
    return send_file(_render(fig), mimetype="image/png")
# === /PATCH ===
