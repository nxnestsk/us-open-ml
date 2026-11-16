"""Corre o projeto inteiro, por ordem, com um só comando (Windows, Linux e macOS).

    python scripts/run_all.py                  # descarrega os dados, corre tudo e os testes
    python scripts/run_all.py --skip-download  # se já tens data/raw preenchido
    python scripts/run_all.py --release        # dados do plano B (ver README)

Depois: streamlit run app/streamlit_app.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A ordem importa: o 05 precisa dos parâmetros do 04 e dos modelos v1 do 02; o 03 precisa dos modelos do 05.
STEPS = ["01_eda_elo.py", "02_models.py", "04_tune_elo.py", "05_walkforward.py", "03_simulate.py"]


def run(args: list[str]) -> None:
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"    ok ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="não descarregar os dados")
    ap.add_argument("--release", action="store_true", help="descarregar os dados da Release do GitHub (plano B)")
    args = ap.parse_args()
    if not args.skip_download:
        run(["scripts/00_download_data.py", *(["--source", "release"] if args.release else [])])
    for s in STEPS:
        run([f"scripts/{s}"])
    run(["-m", "pytest", "-q", "tests"])
    print("\nTudo concluído. Para abrir a aplicação: streamlit run app/streamlit_app.py")
