"""
grid_search_fast.py  —  Precomputed-IV parameter sweep, 720 combos in ~1 min.

Key optimisation: BS IV and smile fit are parameter-independent, so we compute
them ONCE upfront and cache per (day, ts, strike).  The parameter sweep then
only replays the cheap trading / hedging logic.

Run:
  python3 grid_search_fast.py
"""

import csv, itertools, math, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── Data paths ────────────────────────────────────────────────────────────────
DATA = {
    0: "../data/ROUND_3/prices_round_3_day_0.csv",
    1: "../data/ROUND_3/prices_round_3_day_1.csv",
    2: "../data/ROUND_3/prices_round_3_day_2.csv",
}
ATM     = [5000, 5100, 5200, 5300, 5400, 5500]
HG_SYM  = "HYDROGEL_PACK"
VF_SYM  = "VELVETFRUIT_EXTRACT"
VEV_SYM = {K: f"VEV_{K}" for K in ATM}
TTE_BASE = {0: 5.0, 1: 4.0, 2: 3.0}

# ── Minimal BS ────────────────────────────────────────────────────────────────

def _ncdf(x):
    t = 1/(1+0.2316419*abs(x))
    p = 1-(1/math.sqrt(2*math.pi))*math.exp(-0.5*x*x)*(
        t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429)))))
    return p if x >= 0 else 1-p

def _npdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def bs_call(S, K, Td, s):
    if Td<=1e-6 or s<=1e-8: return max(S-K,0)
    T=Td/252; sq=s*math.sqrt(T); d1=(math.log(S/K)+0.5*s*s*T)/sq
    return S*_ncdf(d1)-K*_ncdf(d1-sq)

def bs_delta(S, K, Td, s):
    if Td<=1e-6 or s<=1e-8: return 1.0 if S>K else 0.0
    T=Td/252; return _ncdf((math.log(S/K)+0.5*s*s*T)/(s*math.sqrt(T)))

def bs_vega(S, K, Td, s):
    if Td<=1e-6 or s<=1e-8: return 0.0
    T=Td/252; d1=(math.log(S/K)+0.5*s*s*T)/(s*math.sqrt(T))
    return S*math.sqrt(T)*_npdf(d1)

def bs_iv(C, S, K, Td):
    intr=max(S-K,0)
    if C<intr-0.5 or Td<=1e-6: return float("nan")
    C=max(C,intr+1e-6); T=Td/252
    s=math.sqrt(2*math.pi/T)*C/S; s=max(0.05,min(s,5.0))
    for _ in range(20):           # 20 iters is enough (was 40)
        p=bs_call(S,K,Td,s); v=bs_vega(S,K,Td,s)
        if abs(v)<1e-10: break
        s-=(p-C)/v; s=max(1e-6,min(s,10.0))
    return s

def fit_smile(pairs):
    if len(pairs)<3: return None
    sx4=sx3=sx2=sx1=n=syx2=syx1=sy=0.0
    for x,y in pairs:
        x2=x*x; sx4+=x2*x2;sx3+=x2*x;sx2+=x2;sx1+=x;n+=1;syx2+=y*x2;syx1+=y*x;sy+=y
    M=[[sx4,sx3,sx2],[sx3,sx2,sx1],[sx2,sx1,n]]; r=[syx2,syx1,sy]
    def d3(m): return(m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                     -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                     +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    def sub(m,v,c):
        o=[row[:] for row in m]; [o[i].__setitem__(c,v[i]) for i in range(3)]; return o
    D=d3(M)
    if abs(D)<1e-12: return None
    return d3(sub(M,r,0))/D,d3(sub(M,r,1))/D,d3(sub(M,r,2))/D

# ── Load raw price data ───────────────────────────────────────────────────────

def load_raw():
    by_ts = defaultdict(dict)
    for day, path in DATA.items():
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                by_ts[(day, int(row["timestamp"]))][row["product"]] = row
    return dict(by_ts)

def parse_bk(row):
    try:
        bb=float(row["bid_price_1"]); ba=float(row["ask_price_1"])
        bbv=float(row["bid_volume_1"]); bav=float(row["ask_volume_1"])
        if bb<=0 or ba<=0 or bbv<=0 or bav<=0 or bb>=ba: return None
        return bb,bbv,ba,bav
    except: return None

# ── Pre-compute per-tick derived values (expensive, done once) ────────────────

def precompute(by_ts, keys):
    """
    Returns list of TickData dicts (one per key), each containing:
      hg_bb, hg_bbv, hg_ba, hg_bav, hg_wmid   — HG book
      vf_bb, vf_ba, vf_bbv, vf_bav             — VF book
      tte                                        — time to expiry
      opt[K] = {bb, bbv, ba, bav, fair, delta}  — per-strike BS values
    Ticks where VF book is missing are excluded.
    """
    print("  Pre-computing IV/smile (one-time)...")
    t0 = time.time()
    result = []
    skipped = 0

    for key in keys:
        day, ts = key
        bks = by_ts.get(key, {})
        tte = max(0.01, TTE_BASE.get(day, 3.0) - ts/1_000_000)

        # HG
        hg_data = None
        hg_row = bks.get(HG_SYM)
        if hg_row:
            bk = parse_bk(hg_row)
            if bk:
                bb,bbv,ba,bav = bk
                hg_data = {"bb":bb,"bbv":bbv,"ba":ba,"bav":bav,
                            "wmid":(bb*bav+ba*bbv)/(bbv+bav)}

        # VF
        vf_row = bks.get(VF_SYM)
        if not vf_row: skipped+=1; continue
        vf_bk = parse_bk(vf_row)
        if not vf_bk: skipped+=1; continue
        vfbb,vfbbv,vfba,vfbav = vf_bk
        S = (vfbb+vfba)/2

        # Options: IV + smile fit
        pairs=[]; opt_raw={}
        for K in ATM:
            row=bks.get(VEV_SYM[K])
            if not row: continue
            bk2=parse_bk(row)
            if not bk2: continue
            bb2,bbv2,ba2,bav2=bk2
            mid=(bb2+ba2)/2; tv=mid-max(S-K,0)
            if tv<0.3: continue
            iv=bs_iv(mid,S,float(K),tte)
            if math.isnan(iv) or iv<0.05 or iv>3: continue
            pairs.append((math.log(K/S),iv))
            opt_raw[K]=(bb2,bbv2,ba2,bav2,iv)

        coeffs=fit_smile(pairs) if len(pairs)>=3 else None

        opt={}
        for K in ATM:
            if K not in opt_raw: continue
            bb2,bbv2,ba2,bav2,miv=opt_raw[K]
            m=math.log(K/S)
            if coeffs:
                a,b,c=coeffs; fiv=max(0.05,a*m*m+b*m+c)
            else:
                fiv=miv
            fair=bs_call(S,float(K),tte,fiv)
            delt=bs_delta(S,float(K),tte,fiv)
            opt[K]={"bb":bb2,"bbv":bbv2,"ba":ba2,"bav":bav2,"fair":fair,"delta":delt}

        result.append({
            "key":key,"tte":tte,"S":S,
            "hg":hg_data,
            "vf":{"bb":vfbb,"bbv":vfbbv,"ba":vfba,"bav":vfbav},
            "opt":opt,
        })

    print(f"  Done in {time.time()-t0:.1f}s  ({len(result)} ticks, {skipped} skipped)")
    return result

# ── Fast backtest using precomputed data ──────────────────────────────────────

def backtest(ticks,
             OPT_TAKER_EDGE=1.5, OPT_POS_CAP=35, HEDGE_DEADBAND=5,
             HG_EMA_ALPHA=0.010, HG_TAKE_EDGE=5):

    pos  = defaultdict(int)
    cash = 0.0
    hg_ema = None
    VF_LIM = HG_LIM = 200

    def fill(sym, price, qty):
        nonlocal cash
        pos[sym]+=qty; cash-=price*qty

    for td in ticks:
        # ── HG ────────────────────────────────────────────────────────────────
        hg = td["hg"]
        if hg:
            hg_ema = hg["wmid"] if hg_ema is None else (
                (1-HG_EMA_ALPHA)*hg_ema + HG_EMA_ALPHA*hg["wmid"])
            hp=pos[HG_SYM]; bb=hg["bb"]; bbv=hg["bbv"]; ba=hg["ba"]; bav=hg["bav"]
            if hp>=190:
                q=min(30,int(bbv),HG_LIM+hp);
                if q>0: fill(HG_SYM,bb,-q)
            elif hp<=-190:
                q=min(30,int(bav),HG_LIM-hp);
                if q>0: fill(HG_SYM,ba,q)
            elif bb>=hg_ema+HG_TAKE_EDGE and hp>-HG_LIM:
                q=min(10,int(bbv),HG_LIM+hp);
                if q>0: fill(HG_SYM,bb,-q)
            elif ba<=hg_ema-HG_TAKE_EDGE and hp<HG_LIM:
                q=min(10,int(bav),HG_LIM-hp);
                if q>0: fill(HG_SYM,ba,q)

        # ── Options + hedge ───────────────────────────────────────────────────
        vf=td["vf"]; S=td["S"]; tte=td["tte"]
        if tte<0.3: continue

        total_delta=0.0
        for K,o in td["opt"].items():
            hp=pos[VEV_SYM[K]]
            room_b=max(0,OPT_POS_CAP-hp); room_s=max(0,OPT_POS_CAP+hp)
            fair=o["fair"]
            if o["ba"]<fair-OPT_TAKER_EDGE and room_b>0:
                q=min(10,int(o["bav"]),room_b);
                if q>0: fill(VEV_SYM[K],o["ba"],q)
            elif o["bb"]>fair+OPT_TAKER_EDGE and room_s>0:
                q=min(10,int(o["bbv"]),room_s);
                if q>0: fill(VEV_SYM[K],o["bb"],-q)
            total_delta+=pos[VEV_SYM[K]]*o["delta"]

        tgt=max(-VF_LIM,min(VF_LIM,-round(total_delta)))
        hdg=tgt-pos[VF_SYM]
        if abs(hdg)>=HEDGE_DEADBAND:
            if hdg>0:
                q=min(int(hdg),int(vf["bav"]),VF_LIM-pos[VF_SYM]);
                if q>0: fill(VF_SYM,vf["ba"],q)
            else:
                q=min(int(-hdg),int(vf["bbv"]),VF_LIM+pos[VF_SYM]);
                if q>0: fill(VF_SYM,vf["bb"],-q)

    # MTM at last tick
    last_opt=ticks[-1]["opt"] if ticks else {}
    mtm=0.0
    for K,o in last_opt.items():
        qty=pos.get(VEV_SYM[K],0)
        if qty!=0: mtm+=qty*(o["bb"]+o["ba"])/2
    last_hg=ticks[-1]["hg"] if ticks else None
    if last_hg: mtm+=pos.get(HG_SYM,0)*(last_hg["bb"]+last_hg["ba"])/2
    last_vf=ticks[-1]["vf"] if ticks else None
    if last_vf: mtm+=pos.get(VF_SYM,0)*(last_vf["bb"]+last_vf["ba"])/2

    return cash+mtm, cash, mtm, dict(pos)

# ── Grid ──────────────────────────────────────────────────────────────────────

GRID = {
    "OPT_TAKER_EDGE": [0.5, 1.0, 1.5, 2.0, 3.0],
    "OPT_POS_CAP":    [10, 20, 35, 50],
    "HEDGE_DEADBAND": [0, 5, 10, 20],
    "HG_EMA_ALPHA":   [0.005, 0.010, 0.020],
    "HG_TAKE_EDGE":   [3, 5, 8],
}
SWEEP_KEYS = list(GRID.keys())

def main():
    print("Loading raw data...")
    by_ts = load_raw()
    keys  = sorted(by_ts.keys())
    print(f"  {len(keys)} (day,ts) keys")

    ticks = precompute(by_ts, keys)

    combos = list(itertools.product(*[GRID[k] for k in SWEEP_KEYS]))
    print(f"\nSweeping {len(combos)} combos...")
    t0=time.time()

    results=[]
    for combo in combos:
        p=dict(zip(SWEEP_KEYS,combo))
        pnl,cash,mtm,pos=backtest(ticks,**p)
        results.append((pnl,p,cash,mtm,pos))

    results.sort(key=lambda x:-x[0])
    print(f"Done in {time.time()-t0:.1f}s\n")

    hdr="Rank  PnL          Cash         MtM         Edge   Cap  Hdge  EMA    TkEdge"
    print(hdr); print("-"*len(hdr))
    for rank,(pnl,p,cash,mtm,pos) in enumerate(results[:25],1):
        print(f"{rank:4d}  {pnl:+11.2f}  {cash:+11.2f}  {mtm:+10.2f}"
              f"  {p['OPT_TAKER_EDGE']:5.1f}  {p['OPT_POS_CAP']:3d}"
              f"  {p['HEDGE_DEADBAND']:4d}  {p['HG_EMA_ALPHA']:6.3f}"
              f"  {p['HG_TAKE_EDGE']:6d}")

    bp=results[0][1]
    print("\n=== BEST PARAMS ===")
    for k in SWEEP_KEYS: print(f"  {k:22s}: {bp[k]}")
    print(f"  {'PnL':22s}: {results[0][0]:+.2f}")
    print(f"  {'Positions':22s}: {results[0][4]}")

    print("\n=== SENSITIVITY (best params, vary one at a time) ===")
    for key in SWEEP_KEYS:
        vals=GRID[key]
        if len(vals)==1: continue
        print(f"\n  {key}:")
        for v in vals:
            p=dict(bp); p[key]=v
            pnl,_,_,_=backtest(ticks,**p)
            m="  <- best" if v==bp[key] else ""
            print(f"    {str(v):8s}  {pnl:+10.2f}{m}")

if __name__ == "__main__":
    main()
