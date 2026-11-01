import glob
import os
import shutil
import sys
import tarfile

from huggingface_hub import hf_hub_download

REPO, REVISION = 'SandboxAQ/aqcat25-dataset', '2ec3a36c138ef0150ac9b1d783a7acf0aadfeda9'
SPLITS = ('test_ood_both', 'test_ood_slabs', 'val_ood_both', 'val_ood_slabs')
MAP_URL = 'https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl'


def get(filename, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    path = hf_hub_download(REPO, filename, repo_type='dataset', revision=REVISION, cache_dir='data/hf_cache')
    shutil.copy(path, dest)
    print(dest, flush=True)


if __name__ == '__main__':
    extra = sys.argv[1:]
    for s in SPLITS:
        get(f'parquet/{s}.parquet', f'data/parquet/{s}.parquet')
        get(f'{s}.tar.gz', f'data/tar/{s}.tar.gz')
        if not glob.glob(f'data/raw/{s}/**/*.aselmdb', recursive=True):
            os.makedirs(f'data/raw/{s}', exist_ok=True)
            with tarfile.open(f'data/tar/{s}.tar.gz') as t:
                t.extractall(f'data/raw/{s}', filter='data')
    for s in extra:
        get(f'parquet/{s}.parquet', f'data/parquet/{s}.parquet')
    if not os.path.exists('data/oc20_data_mapping.pkl'):
        import urllib.request

        urllib.request.urlretrieve(MAP_URL, 'data/oc20_data_mapping.pkl')
        print('data/oc20_data_mapping.pkl')
