"""Fetch and verify the real Prudential underwriting-decision benchmark.

Unlike freMTPL2 (which only carries the *observed outcome* "did the policy file
a claim?"), this dataset carries genuine underwriting decisions: the ``Response``
column is the ordinal risk class (1-8) assigned to each of the 59,381 real life
insurance applications by Prudential's human underwriters.

The dataset is distributed through Kaggle and is therefore fetched through the
official Kaggle API with the caller's own credentials. Two prerequisites hold for
the caller (both are documented clearly in the error messages):

    * a Kaggle API access token (prefix ``KGAT_``) in the ``KAGGLE_API_TOKEN``
      environment variable or at ``~/.kaggle/access_token`` -- or, for the legacy
      scheme, a ``username``/``key`` pair at ``~/.kaggle/kaggle.json``;
    * having accepted the competition rules once on the Kaggle website.

The download is fail-closed: without credentials, or if the schema does not match
the documented competition layout, nothing is written and the run aborts.

Outputs:
    data/raw/prudential/train.csv         the real applications and their decisions
    data/raw/prudential/manifest.json     provenance: url, sha256, records, terms
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_DIR = RAW_DIR / 'prudential'

COMPETITION = 'prudential-life-insurance-assessment'
FILE_NAME = 'train.csv'
COMPETITION_URL = f'https://www.kaggle.com/c/{COMPETITION}'
RULES_URL = f'https://www.kaggle.com/c/{COMPETITION}/rules'
DOWNLOAD_URL = (
    f'https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/{FILE_NAME}'
)

EXPECTED_COLUMNS = 129  # Id + 127 real features + Response
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024

# The Competition Terms are non-commercial research use only: recorded honestly.
TERMS = 'Kaggle Competition Terms (research / non-commercial use of the competition data)'


def _config_dir() -> Path:
    """Return the Kaggle configuration directory."""
    return Path(os.environ.get('KAGGLE_CONFIG_DIR', Path.home() / '.kaggle'))


def _authorization() -> str:
    """Build the Kaggle Authorization header, or fail closed.

    The preferred scheme is the new-style API access token (prefix ``KGAT_``),
    read from the ``KAGGLE_API_TOKEN`` environment variable or from
    ``~/.kaggle/access_token`` and sent as a bearer token. The legacy
    ``username``/``key`` pair (``~/.kaggle/kaggle.json``, HTTP basic auth) is
    still supported for older accounts.
    """
    access_token = os.environ.get('KAGGLE_API_TOKEN')
    if not access_token:
        token_file = _config_dir() / 'access_token'
        if token_file.is_file():
            access_token = token_file.read_text(encoding='utf-8').strip()
    if access_token:
        return f'Bearer {access_token.strip()}'

    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    if not (username and key):
        legacy_file = _config_dir() / 'kaggle.json'
        if legacy_file.is_file():
            payload = json.loads(legacy_file.read_text(encoding='utf-8'))
            username = payload.get('username')
            key = payload.get('key')
    if username and key:
        encoded = base64.b64encode(f'{username}:{key}'.encode('utf-8')).decode('ascii')
        return f'Basic {encoded}'

    raise FileNotFoundError(
        'No Kaggle credential found. Create an API token on Kaggle (Account > Create '
        'New API Token), then save it to '
        f'{_config_dir() / "access_token"} or export it as KAGGLE_API_TOKEN, and accept '
        f'the competition rules at {RULES_URL}. A legacy ~/.kaggle/kaggle.json is also '
        'accepted.'
    )


def _download(authorization: str, destination: Path) -> tuple[int, str]:
    """Download the competition file with the given Authorization header and hash it."""
    request = Request(
        DOWNLOAD_URL,
        headers={'Authorization': authorization, 'User-Agent': 'underwriting-research/1.0'},
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with urlopen(request, timeout=60) as response:
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                raise PermissionError(
                    'Kaggle returned an HTML page instead of the CSV. This usually means '
                    f'the competition rules have not been accepted yet: open {RULES_URL}, '
                    'accept them, then re-run.'
                )
            with destination.open('wb') as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f'{FILE_NAME} exceeds the download limit.')
                    digest.update(chunk)
                    output.write(chunk)
    except HTTPError as error:
        if error.code in (401, 403):
            raise PermissionError(
                'Kaggle rejected the request (HTTP %d). Check the API token and make sure '
                'the competition rules have been accepted: %s' % (error.code, RULES_URL)
            ) from error
        raise
    return byte_count, digest.hexdigest()


def inspect_csv(path: Path) -> dict:
    """Validate the documented layout and summarise the real records."""
    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f'{path.name} is empty.') from error
        if len(header) != EXPECTED_COLUMNS:
            raise ValueError(
                f'{path.name}: expected {EXPECTED_COLUMNS} columns '
                f'(Id + 127 features + Response), found {len(header)}.'
            )
        if header[0] != 'Id' or header[-1] != 'Response':
            raise ValueError(
                f'{path.name}: expected the first column "Id" and last "Response"; '
                f'found "{header[0]}" ... "{header[-1]}".'
            )

        rows = 0
        response_index = len(header) - 1
        seen_classes = set()
        for row_number, row in enumerate(reader, start=2):
            if len(row) != EXPECTED_COLUMNS:
                raise ValueError(
                    f'{path.name}:{row_number} has {len(row)} values; '
                    f'expected {EXPECTED_COLUMNS}.'
                )
            value = row[response_index].strip()
            if value:
                seen_classes.add(int(float(value)))
            rows += 1

    if rows == 0:
        raise ValueError(f'{path.name} contains no records.')
    if not seen_classes:
        raise ValueError(f'{path.name} carries no Response value.')
    if min(seen_classes) < 1 or max(seen_classes) > 8:
        raise ValueError(f'{path.name}: Response must be in 1..8; found {sorted(seen_classes)}.')
    return {
        'records': rows,
        'columns': len(header),
        'features': len(header) - 2,
        'response_classes': sorted(seen_classes),
    }


def fetch(output_dir: Path = OUTPUT_DIR) -> dict:
    """Download, verify and record the real underwriting-decision benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / FILE_NAME

    authorization = _authorization()
    print(f'Downloading {FILE_NAME} from the Kaggle competition {COMPETITION} ...')

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{FILE_NAME}-', suffix='.part', dir=output_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        byte_count, sha256 = _download(authorization, temporary_path)
        inspected = inspect_csv(temporary_path)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest = {
        'source': f'Kaggle competition "{COMPETITION}" (real human underwriting decisions)',
        'competition_url': COMPETITION_URL,
        'rules_url': RULES_URL,
        'license': TERMS,
        'citation': (
            'Prudential Life Insurance Assessment, Kaggle (2015). Response is the '
            'ordinal risk class (1-8) assigned by the insurer\'s underwriters.'
        ),
        'file': FILE_NAME,
        'bytes': byte_count,
        'sha256': sha256,
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        **inspected,
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(
        f"{FILE_NAME}: {inspected['records']:,} real applications, "
        f"{inspected['features']} features, classes {inspected['response_classes']}"
    )
    print(f"sha256 {sha256}")
    print(f'Manifest: {output_dir / "manifest.json"}')
    return manifest


if __name__ == '__main__':
    fetch()
