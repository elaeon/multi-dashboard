import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
np.random.seed(42)

BASE = "/home/casa/projects/multi-dashboard/data/sener"

# ============ 1. PRODUCCION ============
print("="*70)
print("1. PRODUCCION HIDROCARBUROS")
print("="*70)
p1 = pd.read_csv(f"{BASE}/produccion_hidrocarburos/01_DGMPB_produccion_nacional_mar26.csv")
p2 = pd.read_csv(f"{BASE}/produccion_hidrocarburos/DGMPB_produccion_nacional_ene_feb.csv")
prod = pd.concat([p1, p2], ignore_index=True)
prod['fecha'] = pd.to_datetime(prod['fecha'])
prod['mes'] = prod['fecha'].dt.strftime('%Y-%m')
print(f"Total rows concatenated: {len(prod)}")
print(f"Months present: {sorted(prod['mes'].unique())}")

prodcols = ['produccion_petroleo_bls','produccion_condensado_bls','produccion_agua_bls','gas_asociado_mmpc','gas_no_asociado_mmpc']

print("\n--- Total national production by month (each product) ---")
by_month = prod.groupby('mes')[prodcols].sum()
print(by_month.round(1).to_string())

print("\n--- Top-10 fields by cumulative OIL+CONDENSATE (bls) Jan-Mar ---")
prod['oil_cond'] = prod['produccion_petroleo_bls'] + prod['produccion_condensado_bls']
top_oil = prod.groupby('nombre_campo')['oil_cond'].sum().sort_values(ascending=False).head(10)
print(top_oil.round(1).to_string())
print(f"National oil+cond total: {prod['oil_cond'].sum():,.1f}")

print("\n--- Top-10 fields by cumulative GAS (assoc+non-assoc mmpc) ---")
prod['gas_tot'] = prod['gas_asociado_mmpc'] + prod['gas_no_asociado_mmpc']
top_gas = prod.groupby('nombre_campo')['gas_tot'].sum().sort_values(ascending=False).head(10)
print(top_gas.round(1).to_string())
print(f"National gas total: {prod['gas_tot'].sum():,.1f}")

print("\n--- Contract-type (tipo_contractual) share of oil & gas ---")
ct = prod.groupby('tipo_contractual').agg(oil_cond=('oil_cond','sum'), gas=('gas_tot','sum'))
ct['oil_share_%'] = 100*ct['oil_cond']/ct['oil_cond'].sum()
ct['gas_share_%'] = 100*ct['gas']/ct['gas'].sum()
print(ct.round(2).to_string())

print("\n--- Well activity: inactive rate per month ---")
# a well is active in a month if any hydrocarbon production > 0 (oil, cond, gas). Water excluded.
prod['any_hc'] = (prod['produccion_petroleo_bls']+prod['produccion_condensado_bls']+prod['gas_asociado_mmpc']+prod['gas_no_asociado_mmpc']) > 0
act = prod.groupby('mes').agg(rows=('any_hc','size'), active=('any_hc','sum'))
act['inactive'] = act['rows']-act['active']
act['inactive_rate_%'] = 100*act['inactive']/act['rows']
print(act.to_string())

# ============ 2. RESERVAS ============
print("\n"+"="*70)
print("2. RESERVAS")
print("="*70)
r5 = pd.read_csv(f"{BASE}/reservas/5_DGRH_reservas_produccion.csv")
print("\n--- R/P ratios (years remaining) ---")
print(r5.to_string(index=False))

r3 = pd.read_csv(f"{BASE}/reservas/3_DGRH_reservas_campo.csv")
r3_1p = r3[r3['categoria_reserva']=='1P'].copy()
print(f"\n3_campo 1P rows: {len(r3_1p)}, unique fields: {r3_1p['nombre_campo'].nunique()}")
print("\n--- Top-10 fields by 1P PCE reserves (mmbpce) ---")
pce = r3_1p.groupby('nombre_campo')['reserva_petroleo_crudo_equivalente_mmbpce'].sum().sort_values(ascending=False)
print(pce.head(10).round(2).to_string())
print(f"Total 1P PCE across fields: {pce.sum():,.2f}")

# Gini
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    n = len(x)
    if n == 0 or x.sum()==0: return np.nan
    cum = np.cumsum(x)
    return (n + 1 - 2*np.sum(cum)/cum[-1]) / n
g = gini(pce.values)
print(f"Gini of 1P PCE across {len(pce)} fields: {g:.4f}")
top5share = 100*pce.head(5).sum()/pce.sum()
top10share = 100*pce.head(10).sum()/pce.sum()
print(f"Top-5 fields share: {top5share:.1f}%  Top-10 share: {top10share:.1f}%")

r4 = pd.read_csv(f"{BASE}/reservas/4_DGRH_reservas_totalidad.csv")
r4_1p = r4[r4['categoria_reserva']=='1P']
print("\n--- Asignacion vs Contrato share of 1P reserves ---")
mod = r4_1p.groupby('tipo_modalidad')[['reserva_aceite_mmb','reserva_gas_natural_mmmpc','reserva_petroleo_crudo_equivalente_mmbpce']].sum()
modsh = 100*mod/mod.sum()
print("Absolute:"); print(mod.round(2).to_string())
print("Share %:"); print(modsh.round(2).to_string())

# ============ 3. PERMISOS ============
print("\n"+"="*70)
print("3. PERMISOS (HPP_FEB)")
print("="*70)
pm = pd.read_csv(f"{BASE}/permisos/permisos_importacion_exportacion_HPP_FEB.csv")
print(f"Rows: {len(pm)}")
print("\n--- Count by tipo_permiso ---")
print(pm['tipo_permiso'].value_counts(dropna=False).to_string())
print("\n--- Count by regimen ---")
print(pm['regimen'].value_counts(dropna=False).to_string())
print("\n--- Top-10 permisionario by permit count ---")
print(pm['permisionario'].value_counts().head(10).to_string())

print("\n--- unidad distribution ---")
print(pm['unidad'].value_counts(dropna=False).to_string())
top_unidad = pm['unidad'].value_counts().index[0]
print(f"\nMost common unidad: {top_unidad}")
print(f"\n--- Top-10 permisionario by total cantidad ({top_unidad} only) ---")
sub = pm[pm['unidad']==top_unidad]
qty = sub.groupby('permisionario')['cantidad'].sum().sort_values(ascending=False).head(10)
print(qty.round(1).to_string())
print(f"Total cantidad in {top_unidad}: {sub['cantidad'].sum():,.1f}")

print("\n--- vigencia parsing & duration from today 2026-07-05 ---")
vig = pd.to_datetime(pm['vigencia'], errors='coerce')
print(f"Parsed as date: {vig.notna().sum()}/{len(pm)} ({100*vig.notna().mean():.1f}%)")
today = pd.Timestamp('2026-07-05')
months_left = (vig - today).dt.days / 30.44
ml = months_left.dropna()
print(f"Duration (months remaining) stats over {len(ml)} parsed rows:")
print(f"  min={ml.min():.1f} p25={ml.quantile(.25):.1f} median={ml.median():.1f} mean={ml.mean():.1f} p75={ml.quantile(.75):.1f} max={ml.max():.1f}")
print(f"  Already expired (<0 months): {(ml<0).sum()} ({100*(ml<0).mean():.1f}%)")
print(f"  Expiring within 6 months: {((ml>=0)&(ml<6)).sum()}")
print(f"  Valid >12 months: {(ml>12).sum()}")
print(f"vigencia min date={vig.min()} max date={vig.max()}")
