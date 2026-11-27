import json

import yaml

from mccs_data_qa.reports import report_ui


def test_folder_set_row(tmp_path):
    """ The folder table row comes from the set's yaml (LST from the Set ID) and its QA mark. """
    d = tmp_path / 'set_2026-11-17_110'
    (d / 'reports').mkdir(parents=True)
    details = [{'Station ID': s, 'Day/Night (AWST)': 'Day'} for s in ('s8-1', 's8-6')]
    (d / report_ui.SET_YAML).write_text(yaml.safe_dump({'set_id': 20261117110, 'set_details': details}))
    (d / 'reports' / 'set_qa.json').write_text(json.dumps({'status': 'degraded'}))

    row = report_ui.folder_set_row(d)
    assert row['name'] == d.name
    assert row['lst'] == 11.0
    assert row['day'] == 'Day'
    assert row['n_obs'] == 2
    assert row['qa'] == 'degraded'
    assert row['reports'] == []
