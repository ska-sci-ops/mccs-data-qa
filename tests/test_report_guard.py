"""guarded(): a failing report section becomes an error card instead of losing the report."""

import healpy as hp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mccs_data_qa.reports.report import guarded, guarded_tab  # noqa: E402


def test_ok_passes_through():
    assert guarded('x', lambda: '<p>ok</p>') == ('<p>ok</p>', True)
    assert guarded_tab('t', 'Tab', lambda: 'body') == ('t', 'Tab', 'body')


def test_failure_becomes_error_card_and_closes_figures():
    def build():
        plt.figure()
        raise ValueError('bad <data>')

    body, ok = guarded('Plot', build)
    assert not ok
    assert 'Plot failed' in body and 'bad &lt;data&gt;' in body
    assert plt.get_fignums() == []
    assert guarded_tab('t', 'Tab', build)[1] == 'Tab (failed)'


def test_all_nan_mollview_is_caught():
    # The quicklook crash: an all-NaN healpix with max=NaN breaks the colorbar
    body, ok = guarded('Combined image', lambda: hp.mollview(np.full(hp.nside2npix(8), np.nan), max=np.nan))
    assert not ok and 'Combined image failed' in body
