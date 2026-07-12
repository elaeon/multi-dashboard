"""
Convierte el tar.gz de becas_CNBBBJ de un año a un único archivo Parquet.

Los subdirectorios dentro del tar (basica, media_superior, superior) se agregan
como columna de partición `nivel_educativo`. Lee en memoria sin descomprimir al disco.

Uso:
    uv run python scripts/becas_CNBBBJ_to_parquet.py <year>

Ejemplo:
    uv run python scripts/becas_CNBBBJ_to_parquet.py 2025
    # Lee:    data/datos_gob/becas_CNBBBJ/2025.tar.gz
    # Genera: data/datos_gob/becas_CNBBBJ/becas_2025.parquet
"""

import io
import sys
import tarfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


BASE = Path("data/datos_gob/becas_CNBBBJ")

schema = {
    "TRIMESTRE": pl.String,
    "CVE_EDO": pl.String,
    "NOM_EDO": pl.String,
    "CVE_MUN": pl.String,
    "NOM_MUN": pl.String,
    "CVE_LOC": pl.String,
    "NOM_LOC": pl.String,
    "BECA": pl.Float64,
    "FECHA_ALTA": pl.Date,
    "nivel_educativo": pl.String,
}


def main():
    if len(sys.argv) != 2:
        print("Uso: uv run python scripts/becas_CNBBBJ_to_parquet.py <year>")
        sys.exit(1)

    year = sys.argv[1]
    input_path = BASE / f"{year}.tar.gz"
    output_path = BASE / f"becas_{year}.parquet"

    if not input_path.exists():
        print(f"Error: {input_path} no encontrado")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total = 0

    with tarfile.open(input_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".csv") and m.isfile()]
        print(f"Columnas de partición: ['nivel_educativo']  ({len(members)} archivos)")

        for member in sorted(members, key=lambda m: m.name):
            # Strip leading "{year}/" prefix, then parts[:-1] = partition values
            rel_parts = Path(member.name.removeprefix(f"{year}/")).parts
            nivel_educativo = rel_parts[0] if len(rel_parts) > 1 else ""

            buf = io.BytesIO(tar.extractfile(member).read())
            df = pl.read_csv(buf, infer_schema_length=0)
            df = df.with_columns(pl.lit(nivel_educativo).alias("nivel_educativo"))
            df = df.with_columns(pl.col("FECHA_ALTA").str.to_date(format="%d/%m/%Y"))
            df = df.cast(schema)

            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)

            total += len(df)
            print(f"  {member.name}  ({len(df):,} filas)")

            del df, table, buf

    if writer:
        writer.close()

    print(f"\nListo: {total:,} filas → {output_path}")


if __name__ == "__main__":
    main()
