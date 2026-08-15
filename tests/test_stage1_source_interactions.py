import ast
import json

import pandas as pd
import pytest

from src.stage1_source_interactions import (
    canonicalize_source_columns,
    generate_source_interactions,
    records_to_columns,
    verify_source_interactions,
)


def test_source_uses_steam_id_sorts_and_collapses_by_fieldwise_maximum():
    records = [
        {
            "user_id": "display_name",
            "steam_id": "20",
            "items": [
                {"item_id": "3", "playtime_forever": 1, "playtime_2weeks": 4},
                {"item_id": "2", "playtime_forever": 5, "playtime_2weeks": 0},
            ],
        },
        {
            "user_id": "display_name",
            "steam_id": "20",
            "items": [
                {"item_id": "3", "playtime_forever": 7, "playtime_2weeks": 2},
            ],
        },
        {
            "user_id": "10",
            "steam_id": "10",
            "items": [{"item_id": "9", "playtime_forever": 0}],
        },
    ]
    columns, diagnostics = records_to_columns(records)

    assert columns["user_id"].tolist() == [10, 20, 20]
    assert columns["item_id"].tolist() == [9, 2, 3]
    assert columns["playtime_forever"].tolist() == [0.0, 5.0, 7.0]
    assert columns["playtime_2weeks"].tolist() == [0.0, 0.0, 4.0]
    assert diagnostics["raw_interaction_row_count"] == 4
    assert diagnostics["canonical_edge_count"] == 3
    assert diagnostics["duplicate_excess_rows"] == 1
    assert diagnostics["active_user_count"] == 2


def test_source_rejects_invalid_numeric_and_playtime_contracts():
    with pytest.raises(ValueError, match="steam_id"):
        records_to_columns([{"steam_id": "name", "items": []}])
    with pytest.raises(ValueError, match="nonnegative"):
        records_to_columns(
            [
                {
                    "steam_id": "1",
                    "items": [{"item_id": "2", "playtime_forever": -1}],
                }
            ]
        )
    with pytest.raises(ValueError, match="inconsistent"):
        canonicalize_source_columns([1], [2, 3], [0], [0])


def test_source_publication_is_no_clobber_and_hash_verified(tmp_path):
    raw = tmp_path / "raw.json"
    records = [
        {
            "user_id": "alias",
            "steam_id": "7",
            "items": [{"item_id": "11", "playtime_forever": 3}],
        },
        {"user_id": "8", "steam_id": "8", "items": []},
    ]
    raw.write_text("".join(repr(x) + "\n" for x in records), encoding="utf-8")
    output = tmp_path / "protected" / "source.csv"
    manifest = tmp_path / "public" / "manifest.json"

    first = generate_source_interactions(
        raw_path=raw,
        output_path=output,
        manifest_path=manifest,
        cycle_id="test-cycle",
        project_root=tmp_path,
    )
    second = verify_source_interactions(
        raw_path=raw,
        output_path=output,
        manifest_path=manifest,
        cycle_id="test-cycle",
        project_root=tmp_path,
    )
    assert first == second
    assert pd.read_csv(output).to_dict("records") == [
        {
            "user_id": 7,
            "item_id": 11,
            "playtime_forever": 3,
            "playtime_2weeks": 0,
        }
    ]
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["diagnostics"]["zero_item_record_count"] == 1

    output.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_source_interactions(
            raw_path=raw,
            output_path=output,
            manifest_path=manifest,
            cycle_id="test-cycle",
            project_root=tmp_path,
        )
