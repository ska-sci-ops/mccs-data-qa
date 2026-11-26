from mccs_data_qa.reports.report import AA1_GRID, station_grid


def test_station_grid():
    assert station_grid(['s9-3']) == ({'s9-3': (1, 2, 1)}, 1, 2)
    # Compact grids are filled in station order, not AA1 position
    grid, rows, cols = station_grid({'s10-1', 's8-2', 's9-4'})
    assert (rows, cols) == (2, 2)
    assert list(grid.items()) == [('s8-2', (2, 2, 1)), ('s9-4', (2, 2, 2)), ('s10-1', (2, 2, 3))]
    # 5+ stations keep the fixed AA1 layout
    names = list(AA1_GRID)[:16]
    assert station_grid(names) == ({n: AA1_GRID[n] for n in names}, 3, 6)
