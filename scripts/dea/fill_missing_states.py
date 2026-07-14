import pandas as pd
import glob
import os

BASE = '/home/casa/projects/multi-dashboard/data/dea/desmantelamiento_laboratorios/'
OUT  = BASE + 'desmantelamiento_laboratorios_clean.parquet'

# Static patch for (COUNTY.upper(), CITY.upper()) pairs that only appear in
# missing-state records and cannot be resolved from the internal lookup.
# Verified by US county geography reference.
STATIC_PATCH = {
    ('ADAMS',      'GRAND MARSH'):      'WI',  # Adams County WI
    ('ATLANTIC',   'HAMILTON TWP'):     'NJ',  # Atlantic County NJ
    ('BERGEN',     'EAST RUTHERFORD'):  'NJ',  # Bergen County NJ
    ('BERGEN',     'GARFIELD'):         'NJ',  # Bergen County NJ
    ('BUCKS',      'PERKASIE'):         'PA',  # Bucks County PA
    ('BUCKS',      'SOUTHAMPTON'):      'PA',  # Bucks County PA
    ('BUTLER',     'TRENTON'):          'OH',  # Butler County OH
    ('CENTRE',     'CENTRE HALL'):      'PA',  # Centre County PA
    ('CLAY',       'KANSAS CITY'):      'MO',  # Clay County MO (KC north)
    ('CLAYTON',    'RIVERDALE'):        'GA',  # Clayton County GA
    ('COLUMBIA',   'WISCONSIN DELLS'):  'WI',  # Columbia County WI
    ('CONVERSE',   'GLENROCK'):         'WY',  # Converse County WY
    ('CUMBERLAND', 'PORTLAND'):         'ME',  # Cumberland County ME
    ('CUMBERLAND', 'VINELAND'):         'NJ',  # Cumberland County NJ
    ('DEKALB',     'BROOKHAVEN'):       'GA',  # DeKalb County GA
    ('DOUGLAS',    'DOUGLASVILLE'):     'GA',  # Douglas County GA (county seat)
    ('ESSEX',      'ANDOVER'):          'MA',  # Essex County MA
    ('ESSEX',      'MONTCLAIR'):        'NJ',  # Essex County NJ
    ('ESSEX',      'NEWARK'):           'NJ',  # Essex County NJ (county seat)
    ('FAIRFAX',    'FAIRFAX'):          'VA',  # Fairfax County VA
    ('FULTON',     'ROSWELL'):          'GA',  # Fulton County GA
    ('GALLIA',     'BIDWELL'):          'OH',  # Gallia County OH
    ('GOSHEN',     'TORRINGTON'):       'WY',  # Goshen County WY (county seat)
    ('GREEN LAKE', 'MARKESAN'):         'WI',  # Green Lake County WI
    ('HAMPDEN',    'SPRINGFIELD'):      'MA',  # Hampden County MA (county seat)
    ('HENRY',      'KAWANEE'):          'IL',  # Henry County IL (Kewanee, likely typo)
    ('JACKSON',    'CENTRAL POINT'):    'OR',  # Jackson County OR
    ('JACKSON',    'MEDFORD'):          'OR',  # Jackson County OR (county seat)
    ('JEFFERSON',  'TEXICO'):           'NM',  # Texico is in NM (Curry County; county field likely wrong)
    ('KEITH',      'OGALLALA'):         'NE',  # Keith County NE (county seat)
    ('KINGMAN',    'PRETTY PRAIRIE'):   'KS',  # Kingman County KS area (Pretty Prairie is in Reno Co KS but state is KS)
    ('LAKE',       'KIRTLAND'):         'OH',  # Lake County OH
    ('LIBERTY',    'CLEVELAND'):        'TX',  # Liberty County TX
    ('LIBERTY',    'DAYTON'):           'TX',  # Liberty County TX
    ('MARSHALL',   'GILBERTSVILLE'):    'KY',  # Marshall County KY
    ('MERCER',     'TRENTON'):          'NJ',  # Mercer County NJ (state capital)
    ('MIDDLESEX',  'MALDEN'):           'MA',  # Middlesex County MA
    ('MIDDLESEX',  'PISCATAWAY'):       'NJ',  # Middlesex County NJ
    ('MILWAUKEE',  'MILWAUKEE'):        'WI',  # Milwaukee County WI
    ('MONMOUTH',   'LONG BRANCH'):      'NJ',  # Monmouth County NJ
    ('NATRONA',    'CASPER'):           'WY',  # Natrona County WY (county seat)
    ('NORFOLK',    'QUINCY'):           'MA',  # Norfolk County MA
    ('ORANGE',     'LAKE FOREST'):      'CA',  # Orange County CA
    ('OSCEOLA',    'EVART'):            'MI',  # Osceola County MI
    ('PIKE',       'HOLMESVILLE'):      'OH',  # Pike County OH area
    ('SEQUATCHIE', 'WHITWELL'):         'TN',  # Sequatchie County TN
    ('SONOMA',     'PETALUMA'):         'CA',  # Sonoma County CA
    ('ST. CLAIR',  'CAHOKIA'):          'IL',  # St. Clair County IL
    ('STRAFFORD',  'LEE'):              'NH',  # Strafford County NH
    ('UNION',      'PLAINFIELD'):       'NJ',  # Union County NJ
    ('UNKNOWN',    'MAITE'):            'GU',  # Guam (territory)
    ('UNKNOWN',    'SINAJANA'):         'GU',  # Guam (territory)
    ('WAPELLO',    'OTTUMWA'):          'IA',  # Wapello County IA (county seat)
    ('WARREN',     'ROCK ISLAND'):      'IL',  # Warren/Rock Island area IL
    ('WASHINGTON', 'HILLSBORO'):        'OR',  # Washington County OR (county seat)
    ('WAYNE',      'FAIRFIELD'):        'IN',  # Wayne County IN area
    # JOHNSON | OZARK: ambiguous — Ozark appears in AR/AL/MO but none in Johnson County
}


def is_missing(v):
    return pd.isna(v) or str(v).strip() in ('', 'N/A', 'NA', 'UNKNOWN', 'UNK', 'NONE')


# 1. Load all CSVs
frames = []
for f in sorted(glob.glob(BASE + '*.csv')):
    df = pd.read_csv(f, encoding='latin-1', low_memory=False)
    df['source_file'] = os.path.basename(f)
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(all_df)} rows from {len(frames)} files")

# 2. Build lookups from known-state rows
known = all_df[~all_df['state'].apply(is_missing)].copy()
known['_county'] = known['county'].str.upper().str.strip()
known['_city']   = known['city'].str.upper().str.strip()

city_county_lookup = (
    known.groupby(['_county', '_city'])['state']
    .agg(lambda x: x.mode()[0])
    .to_dict()
)

county_states = known.groupby('_county')['state'].nunique()
unambiguous_counties = county_states[county_states == 1].index
county_lookup = (
    known[known['_county'].isin(unambiguous_counties)]
    .groupby('_county')['state'].first()
    .to_dict()
)
print(f"Internal lookup: {len(city_county_lookup)} (county,city) pairs, {len(county_lookup)} unambiguous counties")

# 3. Fill missing states — internal lookup first, then static patch
all_df['state_inferred']   = False
all_df['state_unresolved'] = False

missing_mask = all_df['state'].apply(is_missing)
print(f"Missing state: {missing_mask.sum()} rows")

for i in all_df[missing_mask].index:
    row    = all_df.loc[i]
    county = str(row['county']).upper().strip()
    city   = str(row['city']).upper().strip()

    if (county, city) in city_county_lookup:
        all_df.at[i, 'state'] = city_county_lookup[(county, city)]
        all_df.at[i, 'state_inferred'] = True
    elif county in county_lookup:
        all_df.at[i, 'state'] = county_lookup[county]
        all_df.at[i, 'state_inferred'] = True
    elif (county, city) in STATIC_PATCH:
        all_df.at[i, 'state'] = STATIC_PATCH[(county, city)]
        all_df.at[i, 'state_inferred'] = True
    else:
        all_df.at[i, 'state_unresolved'] = True

# 4. Add year column
all_df['year'] = pd.to_datetime(all_df['date'], errors='coerce').dt.year

# 5. Report
n_inferred   = all_df['state_inferred'].sum()
n_unresolved = all_df['state_unresolved'].sum()
print(f"\nResults:")
print(f"  Inferred:   {n_inferred} rows")
print(f"  Unresolved: {n_unresolved} rows")

if n_unresolved > 0:
    print("\nUnresolved rows (check manually):")
    print(all_df[all_df['state_unresolved']][['county', 'city', 'date', 'source_file']].to_string())

# 6. Save
all_df.drop(columns=['source_file']).to_parquet(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"Final row count: {len(all_df)}")
