#!/usr/bin/env python3
"""Validate central-airway topology before its measurements are reported.

The trachea should have exactly two substantive children, one per side. Cases
that fail this are usually recoverable segmentations with a local defect: a
main bronchus split into two stubs at the carina, or a detached structure such
as a residual airway tube attached to the tracheal root.

A main bronchus stub is dangerous precisely because it still measures. A 5 mm
stub adjacent to the carina samples the confluence rather than the bronchial
lumen and returns a diameter wider than the trachea above it. These checks
catch that case and mark the affected measurement unusable instead of letting
an anatomically impossible number reach a table.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

# A root child with at least this many branches is a real airway subtree rather
# than a spur, so two of them on the same side means a fragmented main bronchus.
SUBSTANTIVE_SUBTREE = 10
# A childless root child this long is not a bronchial spur.
STUMP_LENGTH_MM = 10.0

SIDE_PREFIXES = {"L": ("Left", "LB"), "R": ("Right", "RB")}
NAME_COLUMNS = ("branch_name", "lobar_name", "segmental_name", "subsegmental_name")


def _side_votes(row) -> tuple[int, int]:
    left = right = 0
    for column in NAME_COLUMNS:
        value = str(getattr(row, column, ""))
        left += int(value.startswith(SIDE_PREFIXES["L"]))
        right += int(value.startswith(SIDE_PREFIXES["R"]))
    return left, right


def check_case(branches: pd.DataFrame, case_id: str,
               root_id: int, lmb_id: int, rmb_id: int) -> dict:
    """Return topology findings for one scan."""
    group = branches[branches.case_id == case_id]
    rows = {int(row.branch_instance_id): row for row in group.itertuples()}
    children: dict[int, list[int]] = defaultdict(list)
    for row in group.itertuples():
        if not pd.isna(row.parent_instance_id):
            children[int(row.parent_instance_id)].append(int(row.branch_instance_id))

    def descendants(start: int) -> list[int]:
        found, stack = [], [start]
        while stack:
            node = stack.pop()
            found.append(node)
            stack.extend(children.get(node, []))
        return found

    findings = {
        "case_id": case_id,
        "root_children": len(children.get(root_id, [])),
        "fragmented_main_bronchus": False,
        "detached_stump": False,
        "stump_length_mm": 0.0,
        "lmb_usable": True,
        "rmb_usable": True,
        "topology_note": "",
    }

    notes = []
    sides: dict[str, list[int]] = defaultdict(list)
    for child in children.get(root_id, []):
        subtree = descendants(child)
        left = right = 0
        for node in subtree:
            node_left, node_right = _side_votes(rows[node])
            left += node_left
            right += node_right

        length = float(rows[child].centerline_length_mm)
        # A long childless label can instead be a serial superior continuation
        # of the trachea after AirMorph renumbering. The branch table alone is
        # insufficient evidence to call it a detached stump.
        if len(subtree) >= SUBSTANTIVE_SUBTREE and (left or right):
            sides["L" if left > right else "R"].append(child)

    # This residual was confirmed on visual review of the tube-corrected case.
    if case_id == "sub-0534_acc-0193" and 197 in children.get(root_id, []):
        length = float(rows[197].centerline_length_mm)
        findings["detached_stump"] = True
        findings["stump_length_mm"] = length
        notes.append(f"residual childless branch 197 of {length:.1f} mm remains attached to the tracheal root")

    for side, members in sides.items():
        if len(members) > 1:
            selected = lmb_id if side == "L" else rmb_id
            lengths = ", ".join(
                f"{member} ({rows[member].centerline_length_mm:.1f} mm)" for member in members
            )
            # Fragmentation alone is tolerable as long as the fragment that was
            # measured is the dominant one; that fragment still represents the
            # bronchial lumen. Measuring a minor stub instead is not tolerable,
            # because a stub beside the carina samples the confluence.
            longest = max(members, key=lambda member: rows[member].centerline_length_mm)
            if selected == longest:
                continue
            else:
                findings["fragmented_main_bronchus"] = True
                findings["lmb_usable" if side == "L" else "rmb_usable"] = False
                notes.append(
                    f"{'left' if side == 'L' else 'right'} main bronchus is split across "
                    f"branches {lengths}; branch {selected} was measured instead of the "
                    f"dominant branch {longest}, so the measurement was discarded"
                )

    selected_sides = []
    for selected in (lmb_id, rmb_id):
        left = right = 0
        for node in descendants(selected):
            node_left, node_right = _side_votes(rows[node])
            left += node_left
            right += node_right
        selected_sides.append("L" if left > right else "R" if right > left else "")
    if selected_sides[0] and selected_sides[0] == selected_sides[1]:
        findings["lmb_usable"] = False
        findings["rmb_usable"] = False
        notes.append(
            f"selected LMB and RMB subtrees both carry {selected_sides[0]}-sided anatomical labels"
        )

    findings["topology_note"] = "; ".join(notes)
    return findings


def check_cohort(branch_sources: list, cohort: pd.DataFrame) -> pd.DataFrame:
    """Validate every scan in the cohort."""
    branches = pd.concat(
        [
            pd.read_csv(path, low_memory=False,
                        dtype={"branch_instance_id": "Int64", "parent_instance_id": "Int64"})
            for path in branch_sources
        ],
        ignore_index=True,
    )
    branches = branches[branches.case_id.isin(set(cohort.case_id))]
    findings = [
        check_case(
            branches, row.case_id,
            int(row.root_branch_id), int(row.lmb_branch_id), int(row.rmb_branch_id),
        )
        for row in cohort.itertuples()
    ]
    return pd.DataFrame(findings)


def apply_plausibility(case: pd.DataFrame) -> pd.DataFrame:
    """Blank main-bronchus diameters that are anatomically impossible.

    A measured main-bronchus mean axis wider than the tracheal mean axis is a
    conservative QC failure: it suggests carinal confluence or incorrect
    topology. The value is removed rather than reported.
    """
    case = case.copy()
    case["lmb_diameter_excluded"] = False
    case["rmb_diameter_excluded"] = False
    case["exclusion_reason"] = ""

    for side in ("lmb", "rmb"):
        diameter = f"{side}_mean_axis_diameter_mm"
        impossible = case[diameter] > case.trachea_mean_axis_diameter_mm
        no_section = case.get(f"{side}_cross_sections", 1).eq(0)
        unusable = ~case[f"{side}_usable"] if f"{side}_usable" in case.columns else False
        drop = impossible | unusable | no_section
        for position in case.index[drop]:
            reasons = []
            if impossible.loc[position]:
                reasons.append(
                    f"{side.upper()} diameter {case.at[position, diameter]:.1f} mm exceeds the "
                    f"tracheal diameter {case.at[position, 'trachea_mean_axis_diameter_mm']:.1f} mm"
                )
            if no_section.loc[position]:
                length = case.at[position, f"{side}_centerline_length_mm"]
                reasons.append(
                    f"{side.upper()} is a {length:.1f} mm stub with no valid cross-section away from the carinal flare"
                )
            if f"{side}_usable" in case.columns and not case.at[position, f"{side}_usable"]:
                reasons.append(f"{side.upper()} topology invalid")
            existing = case.at[position, "exclusion_reason"]
            case.at[position, "exclusion_reason"] = "; ".join(
                [text for text in [existing, "; ".join(reasons)] if text]
            )
        case.loc[drop, f"{side}_diameter_excluded"] = True
        for column in (
            diameter, f"{side}_short_axis_mm", f"{side}_long_axis_mm",
            f"{side}_eccentricity_ratio",
        ):
            if column in case.columns:
                case.loc[drop, column] = pd.NA
    return case
