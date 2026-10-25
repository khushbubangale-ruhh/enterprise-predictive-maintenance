"""Download official UCI AI4I 2020 CSV; needs internet."""
import io
import zipfile
import requests
from src.common import DATA
URL = 'https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip'

def main():
    r = requests.get(URL, timeout=90)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith('ai4i2020.csv')]
        if not candidates:
            raise RuntimeError('Official ZIP did not contain ai4i2020.csv')
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(archive.read(candidates[0]))
    print(f'Downloaded dataset to {DATA}')

if __name__ == '__main__':
    main()
