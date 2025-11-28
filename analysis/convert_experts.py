from pathlib import Path
import pandas as pd

def convert_dir(xlsx_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(xlsx_dir.glob("*.xlsx")):
        try:
            df = pd.read_excel(f, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed reading {f}: {e}")
        csv_path = out_dir / (f.stem + ".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[ok] {f.name} -> {csv_path.name}")

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    # experts: xlsx -> data/experts/*.csv
    convert_dir(repo / "data" / "experts", repo / "data" / "experts")
    # model inference: xlsx -> model/inference/*.csv
    convert_dir(repo / "model" / "inference", repo / "model" / "inference")
