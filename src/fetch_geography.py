"""Download CEPII GeoDist (dist_cepii.xls) into data/external/cepii/ (run once; needs internet).
Source: https://www.cepii.fr/distance/dist_cepii.zip (free for research use; not redistributed in this repository).
"""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

OUT = Path(__file__).resolve().parent.parent / "data" / "external" / "cepii"
URL = "https://www.cepii.fr/distance/dist_cepii.zip"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=300, context=ssl.create_default_context(cafile=certifi.where())).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(OUT)
    print("saved", [p.name for p in OUT.iterdir()])
