# backtest/run_backtest.py
# Lance la simulation pour une ou plusieurs paires, écrit les résultats dans results/
# et envoie un résumé (et la courbe de capital) sur Telegram si BOT_TOKEN / CHAT_ID sont définis.
#
# Exemple :
#   python backtest/run_backtest.py --symbols BTCUSDT,ETHUSDT --interval 1h --start 2022-01-01

import argparse
import os
import pandas as pd
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import engine  # noqa: E402
import strategy  # noqa: E402


def apply_params(text):
    """'k=v;k2=v2' -> modifie strategy.PARAMS (chaque valeur est convertie dans le type de la valeur par défaut)."""
    changed = {}
    for part in [x.strip() for x in (text or "").replace("\n", ";").split(";") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"réglage invalide '{part}' (format attendu : nom=valeur)")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in strategy.PARAMS:
            raise ValueError(f"réglage inconnu '{k}'. Possibles : {', '.join(strategy.PARAMS)}")
        cur = strategy.PARAMS[k]
        strategy.PARAMS[k] = (float(v) if isinstance(cur, float) else int(float(v)) if isinstance(cur, int)
                              else v.replace(" ", ""))
        changed[k] = strategy.PARAMS[k]
    return changed


def setup_breakdown(trades):
    """Résultats par type de setup (ex. bubble5m|4h = bulle 5m, range H4) : n, % gagnants, espérance en R."""
    if not len(trades) or "setup" not in trades:
        return ""
    t = trades.copy()
    t["kind"] = t["setup"].astype(str).str.replace(r"[YR]$", "", regex=True)
    rows = []
    for k, g in t.groupby("kind"):
        rows.append((len(g), f"  {k}: {len(g)} trades | gagnants {(g['pnl'] > 0).mean() * 100:.0f} % | {g['r'].mean():+.2f} R"))
    rows.sort(reverse=True)
    return "Par setup :\n" + "\n".join(r for _, r in rows[:12])


def _fp(x):
    return f"{x:.2f}" if abs(x) >= 1 else f"{x:.5f}"


def trades_report(trades, symbol, interval):
    """Texte lisible : une fiche par trade (heure d'entrée, SL, TP, points du Fibonacci, confirmation)."""
    if not len(trades):
        return ""
    out = []
    for n, (_, t) in enumerate(trades.iterrows(), 1):
        g = lambda k: t[k] if k in t and t[k] == t[k] else None
        icon = "🔴 SHORT" if t["side"] == "short" else "🟢 LONG"
        res = f"{t['reason']} ({t['r']:+.2f} R, {t['pnl']:+.0f})"
        lines = [f"#{n} {symbol} {icon} | {res}",
                 f"Entrée : {str(t['entry_time'])[:16]} UTC à {_fp(t['entry'])} | sortie {str(t['exit_time'])[:16]} à {_fp(t['exit'])}",
                 f"SL {_fp(t['sl'])} | TP {_fp(t['tp'])} | {t['bars']} bougies {interval}"]
        if g("confirm"):
            lines.append(f"Confirmation : {t['confirm']} (clôturée {t['signal_time']} UTC) | range {t['range_tf']}")
        if g("trend_dir"):
            lines.append(f"Tendance {t['trend_tf']} {t['trend_dir']} ({t['trend_mode']}) → {t['side'].upper()} | sommets {_fp(t['trend_prev_high'])} → {_fp(t['trend_last_high'])}"
                         f" | creux {_fp(t['trend_prev_low'])} → {_fp(t['trend_last_low'])} | clôture {_fp(t['trend_close'])} vs EMA {_fp(t['trend_ema'])}")
        if g("fib_range_high") is not None:
            lines.append(f"Fibonacci du range ({t['range_tf']}) : haut {_fp(t['fib_range_high'])} ({t['fib_range_high_time']}) "
                         f"/ bas {_fp(t['fib_range_low'])} ({t['fib_range_low_time']})")
            lines.append(f"Zone OTE : {_fp(t['ote_zone_low'])} – {_fp(t['ote_zone_high'])}")
            lines.append(f"Fibonacci du TP (jambe) : haut {_fp(t['fib_leg_high'])} ({t['fib_leg_high_time']}) "
                         f"/ bas {_fp(t['fib_leg_low'])} ({t['fib_leg_low_time']}) | niveau {t['tp_fib']}")
        if g("bubble_band") is not None:
            edge_txt = "bas du corps" if t["bubble_kind"] == "HAUTE" else "haut du corps"
            band_txt = "bande haute" if t["bubble_kind"] == "HAUTE" else "bande basse"
            lines.append(f"Bulle {t['bubble_kind']} : bougie {t['bubble_time']} UTC | {edge_txt} {_fp(t['bubble_body_edge'])} "
                         f"vs {band_txt} {_fp(t['bubble_band'])} | écart {t['bubble_dist_pct']:.3f} %")
        if g("mss_second_extreme") is not None:
            lines.append(f"MSS : 1er extrême {_fp(t['mss_first_extreme'])} → 2e extrême {_fp(t['mss_second_extreme'])} "
                         f"| creux d'origine cassé {_fp(t['mss_origin_level'])}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def funnel_text():
    """Où les candidats (bulles / MSS) sont éliminés. Un même candidat est testé sur le range H4 et sur le range H1."""
    lines = []
    for kind, label in (("bubble", "Bulles"), ("mss", "MSS")):
        rows = sorted((r_, n) for (k, r_), n in strategy.DIAG.items() if k == kind)
        if not rows:
            continue
        parts = []
        for r_, n in rows:
            name = "détectés" if r_[0] == "0" else ("SIGNAUX" if r_[0] == "9" else r_[2:])
            parts.append(f"{name} {n}")
        lines.append(f"  {label} : " + " | ".join(parts))
    return "Entonnoir (candidats éliminés par chaque filtre) :\n" + "\n".join(lines) if lines else ""


def fmt_summary(symbol, interval, source, start, end, s):
    lines = [f"📊 Backtest {symbol} {interval} | {start} → {end or 'aujourd’hui'} | données: {source}",
             f"Capital : {s['capital_start']:.0f} → {s['capital_end']:.0f} ({s['return_pct']:+.2f} %)",
             f"Buy & hold : {s['buy_hold_pct']:+.2f} %",
             f"Max drawdown : {s['max_drawdown_pct']:.2f} %"]
    if s["trades"]:
        pf = "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        if s.get("skipped_leverage"):
            lines.append(f"⚠️ {s['skipped_leverage']} trades refusés : levier max insuffisant pour risquer exactement 1 R")
        lines += [f"Trades : {s['trades']} (long {s['longs']} / short {s['shorts']}) | gagnants {s['win_rate_pct']:.1f} %",
                  f"Espérance : {s['avg_r']:+.2f} R par trade | profit factor {pf}",
                  f"Durée moyenne : {s['avg_bars']:.1f} bougies | frais payés : {s['fees_paid']:.0f}"]
    else:
        lines.append("Aucun trade pris sur la période.")
    return "\n".join(lines)


def plot_equity(equity, df, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    base = equity.iloc[0]
    ax.plot(equity.index, equity / base * 100, label="Stratégie", linewidth=1.4)
    bh = df.set_index("time")["close"]
    ax.plot(bh.index, bh / bh.iloc[0] * 100, label="Buy & hold", linewidth=1, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Capital (base 100)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def telegram(text=None, photo=None, document=None):
    """Envoie un message / une image et AFFICHE la réponse de Telegram dans les logs (pour diagnostiquer)."""
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        print("TELEGRAM: BOT_TOKEN ou CHAT_ID manquant (secrets absents de ce dépôt ?) -> envoi ignoré.")
        return False
    ok = True
    try:
        if text:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text[:4000]}, timeout=20)
            print(f"TELEGRAM sendMessage -> {r.status_code} {r.text[:200]}")
            ok = ok and r.status_code == 200
        if photo:
            with open(photo, "rb") as f:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                                  data={"chat_id": chat}, files={"photo": f}, timeout=30)
            print(f"TELEGRAM sendPhoto -> {r.status_code} {r.text[:200]}")
            ok = ok and r.status_code == 200
        if document:
            with open(document, "rb") as f:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument",
                                  data={"chat_id": chat}, files={"document": f}, timeout=60)
            print(f"TELEGRAM sendDocument -> {r.status_code} {r.text[:200]}")
            ok = ok and r.status_code == 200
    except requests.RequestException as e:
        print("TELEGRAM erreur réseau:", e)
        return False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--source", default="auto", choices=["auto", "binance_vision", "kucoin", "bybit"])
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--risk", type=float, default=1.0, help="risque par trade en %% du capital")
    ap.add_argument("--fee", type=float, default=0.1, help="frais par ordre en %%")
    ap.add_argument("--slippage", type=float, default=0.0, help="glissement en %%")
    ap.add_argument("--leverage", type=float, default=20.0, help="levier maximum")
    ap.add_argument("--csv", default="", help="CSV local (time,open,high,low,close,volume) au lieu du téléchargement")
    ap.add_argument("--datadir", default="data", help="dossier des historiques locaux (SYMBOLE.csv.gz ou SYMBOLE.csv)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--params", default="", help="réglages de la stratégie : 'trend_mode=ma;tp_fib=0.62;confirms=bubble'")
    a = ap.parse_args()

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    os.makedirs(a.outdir, exist_ok=True)
    overrides = apply_params(a.params)
    ov_txt = ("Réglages : " + "; ".join(f"{k}={v}" for k, v in overrides.items())) if overrides else "Réglages : par défaut"
    print(ov_txt)
    # message de test immédiat : confirme que le bot Telegram et le CHAT_ID fonctionnent
    telegram(f"🚀 Backtest lancé : {', '.join(symbols)} {a.interval} depuis {a.start}\n{ov_txt}")
    recap = []
    for sym in symbols:
        try:
            local = next((os.path.join(a.datadir, sym + e) for e in (".csv.gz", ".csv") if os.path.exists(os.path.join(a.datadir, sym + e))), None)
            if local and not a.csv:
                df, source = data.load_csv(local), "fichier " + os.path.basename(local)
                t0 = pd.Timestamp(a.start, tz="UTC")
                df = df[df["time"] >= t0]
                if a.end:
                    df = df[df["time"] < pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)]
                df = df.reset_index(drop=True)
                if len(df) < 1000:
                    raise ValueError(f"trop peu de données après filtrage des dates ({len(df)} bougies)")
            elif a.csv:
                df, source = data.load_csv(a.csv), "csv"
            else:
                df, source = data.load(sym, a.interval, a.start, a.end or None, a.source)
            strategy.DIAG.clear()
            trades, equity, stats = engine.run_backtest(
                df, strategy, capital=a.capital, risk_pct=a.risk, fee_pct=a.fee, slippage_pct=a.slippage,
                max_leverage=a.leverage, be_rr=strategy.PARAMS.get("be_rr"), **strategy.ENGINE)
        except Exception as e:
            msg = f"❌ {sym} {a.interval} : {e}"
            print(msg)
            telegram(msg)
            recap.append(f"{sym}: erreur")
            continue
        summary = fmt_summary(sym, a.interval, source, a.start, a.end, stats)
        summary += "\n" + ov_txt
        ft = funnel_text()
        if ft:
            summary += "\n" + ft
        bd = setup_breakdown(trades)
        if bd:
            summary += "\n" + bd
        print(summary)
        tag = f"{sym}_{a.interval}"
        trades.to_csv(os.path.join(a.outdir, f"{tag}_trades.csv"), index=False)
        with open(os.path.join(a.outdir, f"{tag}_resume.txt"), "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        png = os.path.join(a.outdir, f"{tag}_equity.png")
        plot_equity(equity, df, png, f"{sym} {a.interval} — {stats['return_pct']:+.1f} % (buy & hold {stats['buy_hold_pct']:+.1f} %)")
        telegram(summary, png)
        rep_txt = trades_report(trades, sym, a.interval)
        if rep_txt:
            txt_path = os.path.join(a.outdir, f"{tag}_trades_detail.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(rep_txt + "\n")
            blocks = rep_txt.split("\n\n")
            telegram("🧾 Détail des trades (les plus récents) :\n\n" + "\n\n".join(blocks[-4:]))
            telegram(document=txt_path)
            telegram(document=os.path.join(a.outdir, f"{tag}_trades.csv"))
        recap.append(f"{sym}: {stats['return_pct']:+.1f} % | {stats['trades']} trades | DD {stats['max_drawdown_pct']:.1f} %")

    if len(symbols) > 1:
        text = f"📋 Récapitulatif {a.interval} \n" + "\n".join(recap)
        print(text)
        telegram(text)


if __name__ == "__main__":
    main()
