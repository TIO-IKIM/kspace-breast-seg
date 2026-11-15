"""Resolve execution locations without editing scientific manifests."""
from __future__ import annotations
import json
import os
from pathlib import Path

DATASETS = ('mama_mia', 'fastmri_breast')


def configure(site, run_root, dataset, cache_base=None):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    os.environ['NATIVE_DATASET'] = dataset
    os.environ['NATIVE_SITE'] = site
    if cache_base:
        os.environ['NATIVE_CACHE_BASE'] = str(Path(cache_base).resolve())
    mapping = Path(run_root) / 'path_map.json'
    if mapping.exists():
        os.environ['NATIVE_PATH_MAP'] = str(mapping)


def resolve(path):
    value = Path(path)
    if not value.is_absolute():
        return value
    location = os.environ.get('NATIVE_PATH_MAP')
    mappings = json.loads(Path(location).read_text()) if location else {}
    for source, target in sorted(mappings.items(), key=lambda item: -len(item[0])):
        try:
            return Path(target) / value.relative_to(source)
        except ValueError:
            continue
    return value


def cache_base(manifest):
    override = os.environ.get('NATIVE_CACHE_BASE')
    return Path(override) if override else resolve(manifest['cache_base'])


def cache_root(manifest):
    return cache_base(manifest) / 'f100'


def prediction_path(run_root, value):
    path = Path(value)
    return resolve(path) if path.is_absolute() else Path(run_root) / path
