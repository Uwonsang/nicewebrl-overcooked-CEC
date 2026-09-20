"""Maps the raw Overcooked-AI human-human trial layouts (2019/2020 hh_trials.csv) onto
jaxmarl's fixed 9x9 Overcooked grid.

`Overcooked` hardcodes a 9x9 board (see `overcooked.py`), so a layout dict must encode
wall/agent/pot/... indices relative to width=9. We build that by placing the raw grid
(taken verbatim from the CSV's own `layout` column) in the top-left corner of a 9x9
canvas and filling the rest with walls -- the same placement `make_9x9_layout` uses for
`rotate=False`. Leading rows that are entirely wall (e.g. asymmetric_advantages' row 0)
are trimmed first, so the layout sits flush against the top instead of leaving a
redundant all-wall row. `build_padded_layout` returns the row offset this trims, which
callers must subtract from any raw (x, y) position (agent/object positions in CSV state)
before indexing into the padded grid -- a raw x is unaffected (columns aren't trimmed).
"""

from jaxmarl.environments.overcooked import layout_grid_to_dict

# overcooked_ai_py grid symbol -> jaxmarl `layout_grid_to_dict` symbol
# raw: X=wall, O=onion pile, P=pot, D=dish(plate) pile, S=serving/goal, ' '=floor, 1/2=agent
# jaxmarl: W=wall, A=agent, X=goal, B=plate pile, O=onion pile, P=pot, ' '=floor
RAW_TO_JAX_CHAR = {
    "X": "W",
    "O": "O",
    "P": "P",
    "D": "B",
    "S": "X",
    " ": " ",
    "1": "A",
    "2": "A",
}

# raw CSV `layout_name` -> jaxmarl layout name.
# random0/random3 have no jaxmarl grid string of their own; per project convention they
# are treated as the forced_coord / counter_circuit families.
LAYOUT_NAME_MAP = {
    "cramped_room": "cramped_room",
    "coordination_ring": "coord_ring",
    "asymmetric_advantages": "asymm_advantages",
    "random0": "forced_coord",
    "random3": "counter_circuit",
}

GRID_SIZE = 9


def build_padded_layout(raw_grid_rows, jax_layout_name):
    """Embed `raw_grid_rows` (list[str] in overcooked_ai symbols) top-left into a 9x9
    canvas (unused cells become walls) and return `(layout, row_offset)`, where
    `row_offset` is the number of leading all-wall rows trimmed (see module docstring).
    """
    row_offset = 0
    while len(raw_grid_rows) > 1 and all(ch == "X" for ch in raw_grid_rows[0]):
        raw_grid_rows = raw_grid_rows[1:]
        row_offset += 1

    height = len(raw_grid_rows)
    width = len(raw_grid_rows[0])
    if height > GRID_SIZE or width > GRID_SIZE:
        raise ValueError(
            f"layout {jax_layout_name} is {height}x{width}, too large for the {GRID_SIZE}x{GRID_SIZE} env"
        )

    padded_rows = []
    for r in range(GRID_SIZE):
        if r < height:
            row = raw_grid_rows[r]
            chars = []
            for c in range(GRID_SIZE):
                if c < width:
                    ch = row[c]
                    if ch not in RAW_TO_JAX_CHAR:
                        raise ValueError(f"unsupported tile '{ch}' in layout {jax_layout_name}")
                    chars.append(RAW_TO_JAX_CHAR[ch])
                else:
                    chars.append("W")
            padded_rows.append("".join(chars))
        else:
            padded_rows.append("W" * GRID_SIZE)

    grid_str = "\n" + "\n".join(padded_rows) + "\n"
    return layout_grid_to_dict(grid_str, layout_name=jax_layout_name), row_offset
