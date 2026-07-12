"""
Convierte un árbol de CSVs a un único archivo Parquet.

Las carpetas intermedias entre el directorio raíz y cada CSV se agregan
como columnas de partición en el orden en que aparecen en la ruta.
Escribe de forma incremental: un archivo a la vez en memoria.

Uso:
    uv run python scripts/csv_to_parquet.py <input_dir> <output.parquet>

Ejemplo:
    uv run python scripts/csv_to_parquet.py data/datos_gob/becas_CNBBBJ/{YEAR} data/datos_gob/becas_CNBBBJ/{YEAR}
    # Genera columnas: particion_1 (nivel educativo)
"""

import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


partition_names_map = {
    "partition_1": "nivel_educativo"
}

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
    "nivel_educativo": pl.String
}


def partition_names(root: Path, csv_paths: list[Path]) -> list[str]:
    max_depth = max(len(p.relative_to(root).parts) - 1 for p in csv_paths)
    return [partition_names_map[f"partition_{i + 1}"] for i in range(max_depth)]


def main():
    if len(sys.argv) != 3:
        print("Uso: uv run python scripts/csv_to_parquet.py <input_dir> <output.parquet>")
        sys.exit(1)

    root = Path(sys.argv[1])
    output = Path(sys.argv[2])

    if not root.is_dir():
        print(f"Error: {root} no es un directorio válido")
        sys.exit(1)

    csv_paths = sorted(root.rglob("*.csv"))
    if not csv_paths:
        print(f"Error: no se encontraron archivos CSV en {root}")
        sys.exit(1)

    cols = partition_names(root, csv_paths)
    print(f"Columnas de partición: {cols}  ({len(csv_paths)} archivos)")

    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total = 0

    for path in csv_paths:
        parts = path.relative_to(root).parts[:-1]
        df = pl.read_csv(path, infer_schema_length=0)
        for col_name, value in zip(cols, parts):
            df = df.with_columns(pl.lit(value).alias(col_name))
            df = df.with_columns(pl.col("FECHA_ALTA").str.to_date(format="%d/%m/%Y"))
            df = df.cast(schema)

        table = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(output), table.schema)
        writer.write_table(table)

        total += len(df)
        print(f"  {path.relative_to(root)}  ({len(df):,} filas)")

        del df, table  # libera memoria antes del siguiente archivo

    if writer:
        writer.close()

    print(f"\nListo: {total:,} filas → {output}")


if __name__ == "__main__":
    main()
