"""Fetch and verify the public French freMTPL2 actuarial benchmark."""

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

DATASETS = {
    'freMTPL2freq': {
        'url': 'https://openml.org/data/v1/download/20649148/freMTPL2freq.arff',
        'md5': 'f8875568bf0ca622929105197e2db613',
        'expected_attributes': [
            'IDpol', 'ClaimNb', 'Exposure', 'Area', 'VehPower', 'VehAge',
            'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region',
        ],
        'catalogue_description': '677991 motor third-party liability policies',
    },
    'freMTPL2sev': {
        'url': 'https://openml.org/data/v1/download/20649149/freMTPL2sev.arff',
        'md5': '24cc74449e3931cb1aad0d43a12e7a6e',
        'expected_attributes': ['IDpol', 'ClaimAmount'],
        'catalogue_description': '26639 motor third-party liability claim amounts',
    },
}


def _download_verified(dataset_name, metadata, destination):
    digest = hashlib.md5()
    byte_count = 0
    with urlopen(metadata['url'], timeout=30) as response:
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f'{dataset_name} exceeds the 100 MiB download limit.')

        with destination.open('wb') as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f'{dataset_name} exceeds the 100 MiB download limit.'
                    )
                digest.update(chunk)
                output.write(chunk)

    actual_digest = digest.hexdigest()
    if actual_digest != metadata['md5']:
        raise ValueError(
            f'{dataset_name} checksum mismatch: expected {metadata["md5"]}, '
            f'received {actual_digest}. The file was rejected.'
        )
    return byte_count


def inspect_arff(path, expected_attributes):
    """Validate an ARFF header and count records without loading the dataset into RAM."""
    attributes = []
    row_count = 0
    missing_value_count = 0
    data_section_found = False

    with path.open(encoding='utf-8-sig', newline='') as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('%'):
                continue
            if not data_section_found:
                if stripped.lower() == '@data':
                    data_section_found = True
                    continue
                attribute = re.match(
                    r"""@attribute\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+""",
                    stripped,
                    re.IGNORECASE,
                )
                if attribute:
                    attributes.append(next(value for value in attribute.groups() if value))
                continue

            row = next(csv.reader([line], skipinitialspace=True))
            if len(row) != len(expected_attributes):
                raise ValueError(
                    f'{path.name}:{line_number} has {len(row)} values; '
                    f'expected {len(expected_attributes)}.'
                )
            row_count += 1
            missing_value_count += sum(value.strip() == '?' for value in row)

    if not data_section_found:
        raise ValueError(f'{path.name} has no ARFF @data section.')
    if attributes != expected_attributes:
        raise ValueError(
            f'{path.name} schema mismatch: expected {expected_attributes}, '
            f'received {attributes}.'
        )
    if row_count == 0:
        raise ValueError(f'{path.name} contains no records.')
    if missing_value_count:
        raise ValueError(
            f'{path.name} contains {missing_value_count} missing values; '
            'review before using this source.'
        )

    return {
        'attributes': attributes,
        'records': row_count,
        'missing_values': missing_value_count,
    }


def fetch_datasets(output_dir=RAW_DATA_DIR):
    """Download both public datasets, verify checksums/schema, and write provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'source': 'OpenML freMTPL2 datasets',
        'license': 'CC0 (as stated in OpenML dataset metadata)',
        'citation': 'Christophe Dutang and Arthur Charpentier (2018), CASdatasets: Insurance datasets.',
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'datasets': {},
    }

    for dataset_name, metadata in DATASETS.items():
        file_name = f'{dataset_name}.arff'
        final_path = output_dir / file_name
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f'.{dataset_name}-',
            suffix='.part',
            dir=output_dir,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            byte_count = _download_verified(
                dataset_name,
                metadata,
                temporary_path,
            )
            inspected = inspect_arff(
                temporary_path,
                metadata['expected_attributes'],
            )
            os.replace(temporary_path, final_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        manifest['datasets'][dataset_name] = {
            'file': file_name,
            'url': metadata['url'],
            'md5': metadata['md5'],
            'bytes': byte_count,
            'catalogue_description': metadata['catalogue_description'],
            **inspected,
        }
        print(
            f"{dataset_name}: {inspected['records']:,} records, "
            f"{byte_count:,} bytes, schema and checksum verified"
        )

    manifest_path = output_dir / 'fremtpl2_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'Provenance manifest: {manifest_path}')
    return manifest


if __name__ == '__main__':
    fetch_datasets()
