import re
import numpy as np
import pandas as pd
import gender_guesser.detector as gd

pd.set_option("display.width", 160)
BASE = "/home/casa/projects/multi-dashboard/data/secihti"
DET = gd.Detector(case_sensitive=False)

def first_token(name):
    if not isinstance(name, str):
        return None
    t = name.strip().split()
    return t[0] if t else None

_map = {"male": "M", "mostly_male": "M", "female": "F", "mostly_female": "F"}
def guess(tok):
    if not tok:
        return "U"
    return _map.get(DET.get_gender(tok), "U")

def gender_stats(series_firsttoken, label):
    g = series_firsttoken.map(guess)
    vc = g.value_counts()
    n = len(g)
    classified = vc.get("M", 0) + vc.get("F", 0)
    f = vc.get("F", 0); m = vc.get("M", 0); u = vc.get("U", 0)
    print(f"[GENDER] {label}: N={n}  M={m}  F={f}  Unknown={u} ({u/n*100:.1f}% unknown)")
    if classified:
        print(f"         Among classified (N={classified}): F={f/classified*100:.1f}%  M={m/classified*100:.1f}%")
    return g

# ---------------- BECAS ----------------
def load_becas(path):
    df = pd.read_excel(path, sheet_name=0, header=2, engine="openpyxl")
    df = df.dropna(axis=1, how="all")
    df = df[pd.to_numeric(df["CONSEC."], errors="coerce").notna()].copy()
    df["CONSEC."] = pd.to_numeric(df["CONSEC."])
    df["ENTIDAD"] = df["ENTIDAD"].astype(str).str.strip().str.upper()
    df["_first"] = df["NOMBRE BECARIO"].map(first_token)
    return df

b25 = load_becas(f"{BASE}/becas/S190_Becas_de_Posgrado_y_Apoyos_a_la_Calidad_de_Enero_a_Diciembre_2025.xlsx")
b26 = load_becas(f"{BASE}/becas/S190_Becas_de_Posgrado_y_Apoyos_a_la_Calidad_de_Enero_a_Marzo_2026.xlsx")
print("BECAS rows:", len(b25), len(b26))

print("\n===== BECAS 1. GENDER =====")
b25["_g"] = gender_stats(b25["_first"], "BECAS 2025 full year")
b26["_g"] = gender_stats(b26["_first"], "BECAS 2026 Q1")

print("\n===== BECAS 2. TOP 10 STATES (2025) =====")
print((b25["ENTIDAD"].value_counts().head(10) / len(b25) * 100).round(2))
print(b25["ENTIDAD"].value_counts().head(10))

print("\n===== BECAS 3. NIVEL DE ESTUDIOS (2025) =====")
print(b25["NIVEL DE ESTUDIOS"].value_counts())
print((b25["NIVEL DE ESTUDIOS"].value_counts()/len(b25)*100).round(2))

print("\n===== BECAS 4. AREA DEL CONOCIMIENTO (2025) =====")
print(b25["ÁREA DEL CONOCIMIENTO"].value_counts())

print("\n===== BECAS 5. AVG MONTO_TOTAL by level & gender (2025) =====")
mt = "IMPORTE TOTAL PAGADO ENERO-DICIEMBRE"
print("Overall mean/median:", round(b25[mt].mean(),1), round(b25[mt].median(),1))
print("-- by level (mean, median) --")
print(b25.groupby("NIVEL DE ESTUDIOS")[mt].agg(["count","mean","median"]).round(1).sort_values("count", ascending=False))
print("-- by inferred gender (mean, median) --")
print(b25[b25["_g"].isin(["M","F"])].groupby("_g")[mt].agg(["count","mean","median"]).round(1))

print("\n===== BECAS 6. YoY 2025 annual vs 2026 Q1 =====")
print("2025 total scholars:", len(b25), "| 2026 Q1 scholars:", len(b26))
print("2025 Q1 payment mean:", round(b25["IMPORTE PAGADO ENERO-MARZO"].mean(),1),
      "| 2026 Q1 payment mean:", round(b26["IMPORTE PAGADO ENERO-MARZO"].mean(),1))
q1_25 = b25["IMPORTE PAGADO ENERO-MARZO"].sum()
q1_26 = b26["IMPORTE PAGADO ENERO-MARZO"].sum()
print(f"2025 Q1 total spend: {q1_25:,.0f} | 2026 Q1 total spend: {q1_26:,.0f} | change {(q1_26/q1_25-1)*100:.1f}%")
print("2025 full-year total spend:", f"{b25[mt].sum():,.0f}")

# gender by level for becas (gap wider at higher levels?)
print("\n===== BECAS gender share by NIVEL (2025, classified only) =====")
sub = b25[b25["_g"].isin(["M","F"])]
tab = sub.groupby("NIVEL DE ESTUDIOS")["_g"].value_counts().unstack(fill_value=0)
tab["F_pct"] = (tab.get("F",0)/(tab.get("F",0)+tab.get("M",0))*100).round(1)
print(tab.sort_values("F_pct"))

# ---------------- SNII ----------------
print("\n\n########## SNII ##########")
snii25 = pd.read_csv(f"{BASE}/snii/s191_snii_1s_2025.csv", encoding="utf-8-sig")
snii25 = snii25.drop(columns=["comentario"])
snii25["_first"] = snii25["nombre"].map(first_token)

snii26 = pd.read_excel(f"{BASE}/snii/Padron-SNII-2026-1T.xlsx", sheet_name=0, header=0, engine="openpyxl")
snii26 = snii26.dropna(axis=1, how="all")
snii26 = snii26[snii26["CVU"].notna()].copy()
if "NOTAS" in snii26.columns:
    snii26 = snii26.drop(columns=["NOTAS"])
# name format APELLIDO, NOMBRE -> after comma
def snii26_first(x):
    if not isinstance(x,str): return None
    part = x.split(",")
    nm = part[1] if len(part)>1 else part[0]
    return first_token(nm)
snii26["_first"] = snii26["NOMBRE DEL INVESTIGADOR"].map(snii26_first)
print("SNII rows:", len(snii25), len(snii26))

print("\n===== SNII 1. GENDER =====")
snii25["_g"] = gender_stats(snii25["_first"], "SNII H1 2025")
snii26["_g"] = gender_stats(snii26["_first"], "SNII Q1 2026")

NIVEL_LBL = {"C":"Candidato","1":"Nivel I","2":"Nivel II","3":"Nivel III","E":"Emerito"}
print("\n===== SNII 2. NIVEL distribution =====")
print("-- 2025 CSV --")
print(snii25["nivel"].astype(str).value_counts())
print("-- 2026 XLSX --")
print(snii26["NIVEL"].astype(str).value_counts())

print("\n===== SNII 3. TOP 10 STATES (2026 ENTIDAD FINAL) =====")
print(snii26["ENTIDAD FINAL"].value_counts().head(10))
print((snii26["ENTIDAD FINAL"].value_counts().head(10)/len(snii26)*100).round(2))

print("\n===== SNII 4. AREA DE CONOCIMIENTO =====")
print("-- 2026 XLSX --")
print(snii26["AREA DE CONOCIMIENTO"].value_counts())
print("-- 2025 CSV --")
print(snii25["area_conocimiento"].value_counts())

print("\n===== SNII 5. GROWTH 2025->2026 =====")
print(f"H1 2025: {len(snii25)} | Q1 2026: {len(snii26)} | growth {(len(snii26)/len(snii25)-1)*100:.2f}%")
merged = snii25.merge(snii26[["CVU","NIVEL"]], left_on="cvu", right_on="CVU", how="outer", indicator=True)
print(merged["_merge"].value_counts())

print("\n===== SNII 6. GENDER GAP BY LEVEL (2026, classified only) =====")
s = snii26[snii26["_g"].isin(["M","F"])].copy()
s["lvl"] = s["NIVEL"].astype(str).map(NIVEL_LBL)
tab = s.groupby("lvl")["_g"].value_counts().unstack(fill_value=0)
tab["F_pct"] = (tab.get("F",0)/(tab.get("F",0)+tab.get("M",0))*100).round(1)
order = ["Candidato","Nivel I","Nivel II","Nivel III","Emerito"]
print(tab.reindex(order))
# chi-square trend test: is female share associated with level?
from scipy.stats import chi2_contingency
ct = tab.reindex(order)[["F","M"]].values
chi2,p,dof,_ = chi2_contingency(ct)
print(f"Chi-square (F/M x level): chi2={chi2:.1f}, dof={dof}, p={p:.2e}")

print("\n===== SNII 6b. same for H1 2025 CSV =====")
s2 = snii25[snii25["_g"].isin(["M","F"])].copy()
s2["lvl"] = s2["nivel"].astype(str).map(NIVEL_LBL)
tab2 = s2.groupby("lvl")["_g"].value_counts().unstack(fill_value=0)
tab2["F_pct"] = (tab2.get("F",0)/(tab2.get("F",0)+tab2.get("M",0))*100).round(1)
print(tab2.reindex(order))

# ---------------- CROSS ----------------
print("\n\n########## CROSS-PROGRAM ##########")
print("\n===== 7. STATE RANKING: BECAS(2025) vs SNII(2026) =====")
def norm_state(s):
    if not isinstance(s,str): return s
    s = s.strip().upper()
    repl = {"CIUDAD DE MEXICO":"CDMX","DISTRITO FEDERAL":"CDMX",
            "VERACRUZ DE IGNACIO DE LA LLAVE":"VERACRUZ","MICHOACAN DE OCAMPO":"MICHOACAN",
            "MEXICO":"MEXICO","COAHUILA DE ZARAGOZA":"COAHUILA","COAHUILA":"COAHUILA"}
    return repl.get(s,s)
becas_rank = b25["ENTIDAD"].map(norm_state).value_counts()
snii_rank = snii26["ENTIDAD FINAL"].map(norm_state).value_counts()
br = becas_rank.head(10).index.tolist()
sr = snii_rank.head(10).index.tolist()
print("BECAS top10:", br)
print("SNII  top10:", sr)
print("Overlap:", sorted(set(br)&set(sr)), "| count:", len(set(br)&set(sr)))
# spearman on shared states
common = becas_rank.index.intersection(snii_rank.index)
from scipy.stats import spearmanr
br_r = becas_rank[common].rank(ascending=False)
sr_r = snii_rank[common].rank(ascending=False)
rho,pp = spearmanr(becas_rank[common], snii_rank[common])
print(f"Spearman rho (counts across {len(common)} shared states): {rho:.3f}, p={pp:.2e}")

print("\n===== 8. AREA: BECAS vs SNII (normalized text) =====")
def norm_area(a):
    if not isinstance(a,str): return a
    a = re.sub(r"^[IVX]+\.?\s*","",a.strip().upper())
    a = a.replace("-"," ").replace("CS.","CIENCIAS").replace("  "," ").strip()
    return a
ba = b25["ÁREA DEL CONOCIMIENTO"].map(norm_area).value_counts(normalize=True).mul(100).round(1)
sa = snii26["AREA DE CONOCIMIENTO"].map(norm_area).value_counts(normalize=True).mul(100).round(1)
print("-- BECAS area % --"); print(ba)
print("-- SNII area % --"); print(sa)
