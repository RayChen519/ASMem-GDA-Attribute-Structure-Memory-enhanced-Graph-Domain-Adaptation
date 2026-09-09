"""Install pinned conda-forge CPython 3.13 Windows binaries into this .venv.

PyPI's source distribution needs Unix headers with the available MinGW toolchain.
This bootstrap uses the binary distribution linked by the PyMetis maintainers.
No conda environment or second Python interpreter is created.
"""
import hashlib
import io
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
import zipfile

import zstandard

PACKAGES = [
    ('pymetis', '2025.2.2', 'pymetis-2025.2.2-py313hb20f1cf_0.conda',
     'd86879837afcceb6b6fb3d604c692f051e13874387a5d28344e0003a253d8e8c'),
    ('metis', '5.1.0', 'metis-5.1.0-h17e2fc9_1007.conda',
     '4c1dff710c59bb42a7a5d3e77f1772585c56df9fd62744b53b554bbdb682e2a8')]


def main():
    project = Path(__file__).resolve().parents[3]
    venv = project / '.venv'
    if Path(sys.prefix).resolve() != venv.resolve() or sys.version_info[:2] != (3, 13) or sys.platform != 'win32':
        raise RuntimeError('Run with project .venv CPython 3.13 on Windows')
    cache = project / 'code/artifacts/cache/dependencies'
    cache.mkdir(parents=True, exist_ok=True)
    site = venv / 'Lib/site-packages'
    for package, version, filename, sha in PACKAGES:
        archive = cache / filename
        if not archive.exists():
            url = f'https://conda.anaconda.org/conda-forge/win-64/{filename}'
            with urllib.request.urlopen(url, timeout=60) as response:
                archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Dependency archive checksum mismatch')
        with zipfile.ZipFile(archive) as zip_:
            entry = next(n for n in zip_.namelist() if n.startswith('pkg-'))
            with zstandard.ZstdDecompressor().stream_reader(zip_.open(entry)) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        name = member.name.replace('\\', '/')
                        if name.startswith('Lib/site-packages/'):
                            dest = site / name.removeprefix('Lib/site-packages/')
                        elif name.startswith('Library/bin/') and name.endswith('.dll'):
                            # DLL colocated with _internal.pyd, no global PATH edits.
                            dest = site / 'pymetis' / Path(name).name
                        elif name.startswith('info/licenses/'):
                            dest = cache / f'{package}-licenses' / Path(name).name
                        else:
                            continue
                        if not dest.resolve().is_relative_to(project.resolve()):
                            raise RuntimeError('Unsafe archive path')
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, dest.open('wb') as out:
                            shutil.copyfileobj(src, out)
        print(f'Installed {package} {version}; sha256={sha}')
    import pymetis
    print(pymetis.part_graph(2, adjacency=[[1], [0, 2], [1, 3], [2]], options=pymetis.Options(seed=0)))


if __name__ == '__main__':
    main()
