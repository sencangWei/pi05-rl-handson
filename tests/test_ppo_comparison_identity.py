"""Reject simulation comparisons with different policy/runtime contracts."""
import copy
import json
from pathlib import Path

import pytest

from scripts.package_ppo_comparison import validate_identities

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/20261026-ppo-simulation/first-saved-15-updates'


def identities():
    return [json.loads((ARTIFACTS / name).read_text()) for name in
            ('native-sft-identity.json', 'ppo-identity.json')]


def test_recorded_matched_identity():
    assert validate_identities(*identities())['status'] == 'PASS'


@pytest.mark.parametrize('field', ['base_weights_sha256', 'norm_stats_sha256',
                                  'num_steps', 'precision', 'seed_protocol'])
def test_reject_changed_policy_contract(field):
    before, after = identities()
    after = copy.deepcopy(after)
    after['metadata'][field] = 'changed'
    with pytest.raises(ValueError, match=field):
        validate_identities(before, after)


def test_reject_runtime_source_change():
    before, after = identities()
    key = next(k for k in after['source_sha256'] if '/rlinf/models/' in k)
    after['source_sha256'][key] = 'changed'
    with pytest.raises(ValueError, match='runtime source'):
        validate_identities(before, after)
