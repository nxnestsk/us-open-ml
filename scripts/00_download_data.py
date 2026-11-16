"""Passo 0: descarrega os dados abertos usados pelo projeto para data/raw/.

Fontes (ver o README para licenças e atribuição):
  1. Arquivo de Jeff Sackmann (espelho comunitário no GitHub): jogos ATP e WTA até Roland Garros 2026.
     Descarrega-se só os ficheiros de jogos (cerca de 77 MB), sem rankings nem ponto a ponto.
  2. tennis-data.co.uk: resultados e odds de 2025 e 2026 (usados depois de Roland Garros 2026).

Uso:
    python scripts/00_download_data.py            # descarrega o que falta
    python scripts/00_download_data.py --dest X   # outra pasta de destino
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR_URL = "https://github.com/Aneeshers/tennis-sackmann-archive.git"
RELEASE_URL = "https://github.com/nxnestsk/us-open-ml/releases/download/data-v1/sackmann-matches.zip"
# Padrões "sparse": só ficheiros de jogos e a documentação dos campos.
SPARSE = [
    "/atp/atp_matches_[12]*.csv", "/wta/wta_matches_[12]*.csv",
    "/atp/matches_data_dictionary.txt", "/atp/atp_players.csv", "/wta/wta_players.csv",
]
TD_BASE = "http://www.tennis-data.co.uk/hrjk-85HytOjkhth76j_ygh4jf7"
TD_FILES = {  # nome local -> caminho no site
    "atp_2025.xlsx": "2025/2025.xlsx", "atp_2026.xlsx": "2026/2026.xlsx",
    "wta_2025.xlsx": "2025w/2025.xlsx", "wta_2026.xlsx": "2026w/2026.xlsx",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _sackmann_ready(dest: Path) -> bool:
    return bool(list((dest / "atp").glob("atp_matches_*.csv")) and list((dest / "wta").glob("wta_matches_*.csv")))


def download_sackmann_release(dest: Path) -> None:
    """Plano B: o mesmo subconjunto de ficheiros, guardado numa Release deste repositório."""
    print("Sackmann: a descarregar da Release do GitHub...")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_sackmann.zip"
    req = urllib.request.Request(RELEASE_URL, headers={"User-Agent": "us-open-ml-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest)
    tmp.unlink()


def download_sackmann(raw: Path, source: str = "auto") -> None:
    dest = raw / "sackmann_mirror"
    if _sackmann_ready(dest):
        print("Sackmann: já existe, nada a fazer.")
        return
    if dest.exists():
        shutil.rmtree(dest)  # descarga incompleta de uma tentativa anterior
    if source in ("auto", "git"):
        try:
            print("Sackmann: a clonar só os ficheiros de jogos (cerca de 77 MB)...")
            run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", "-q", MIRROR_URL, str(dest)])
            run(["git", "sparse-checkout", "set", "--no-cone", *SPARSE], cwd=dest)
            run(["git", "checkout", "-q"], cwd=dest)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if source == "git":
                raise
            print(f"Sackmann: o espelho falhou ({type(exc).__name__}); a usar a Release do GitHub.")
            if dest.exists():
                shutil.rmtree(dest)
            download_sackmann_release(dest)
    else:
        download_sackmann_release(dest)
    if not _sackmann_ready(dest):
        sys.exit("Sackmann: os ficheiros de jogos não ficaram completos. Vê o README (secção Dados).")
    n_atp = len(list((dest / "atp").glob("atp_matches_*.csv")))
    n_wta = len(list((dest / "wta").glob("wta_matches_*.csv")))
    print(f"Sackmann: {n_atp} ficheiros ATP, {n_wta} ficheiros WTA.")


def download_tennis_data(raw: Path) -> None:
    dest = raw / "tennis_data_uk"
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in TD_FILES.items():
        target = dest / name
        if target.exists() and target.stat().st_size > 10_000:
            print(f"tennis-data: {name} já existe.")
            continue
        print(f"tennis-data: a descarregar {name}...")
        # identificador honesto do cliente. Se o site responder 403, descarrega o ficheiro à mão
        # em http://www.tennis-data.co.uk/alldata.php e guarda-o em data/raw/tennis_data_uk/ com o nome acima.
        req = urllib.request.Request(f"{TD_BASE}/{path}", headers={"User-Agent": "us-open-ml-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if target.stat().st_size < 10_000:  # página de erro em vez de Excel
            target.unlink()
            sys.exit(f"Falha em {name}: o site devolveu uma resposta inesperada. "
                     "Confirma o endereço atual em http://www.tennis-data.co.uk/alldata.php")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "raw", help="pasta de destino")
    ap.add_argument("--source", choices=["auto", "git", "release"], default="auto",
                    help="origem dos dados Sackmann: git (espelho), release (plano B) ou auto (git e, se falhar, release)")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for d in ("data/processed", "models", "reports"):  # pastas de saída dos restantes scripts
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    download_sackmann(args.dest, args.source)
    download_tennis_data(args.dest)
    print("Concluído. Próximo passo: python scripts/01_eda_elo.py")
