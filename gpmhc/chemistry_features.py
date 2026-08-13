# Minimal residue-chemistry annotation helper for the Graph-pMHC edge-feature
# ablation experiment. Deliberately does NOT define interaction weights or
# distances - binary pair annotations only, as specified.
#
# These are appended to EXISTING edges (see gnn_parts.py: lookup_graph,
# lookup_single_graph). This module has no knowledge of graph topology and
# does not construct, count, or filter edges - it only classifies a pair of
# residue letters.

HYDROPHOBIC = set("AVILMFWY")
POLAR = set("STNQC")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWY")

# Order is fixed and must match the order chemistry features are appended
# to the edge tensor in gnn_parts.py.
CHEMISTRY_FEATURE_NAMES = [
    "hydrophobic_pair",
    "polar_pair",
    "positive_negative_pair",
    "positive_positive_pair",
    "negative_negative_pair",
    "aromatic_pair",
]


def get_chemistry_features(residue_1, residue_2):
    """Binary chemistry-compatibility annotations for a pair of residues.

    Parameters
    ----------
    residue_1, residue_2 : str
        Single-character amino acid codes for the two residues connected by
        an existing graph edge. Non-standard tokens (pad '*', flank '$',
        special tokens) simply fall outside every property set below and
        yield all-zero features - this is intentional, not an error case.

    Returns
    -------
    list of int (length 6)
        [hydrophobic_pair, polar_pair, positive_negative_pair,
         positive_positive_pair, negative_negative_pair, aromatic_pair]
        Each is 1 if both residues satisfy the corresponding relationship,
        else 0. No weighting or scaling - binary annotations only.
    """
    r1 = residue_1.upper() if isinstance(residue_1, str) else residue_1
    r2 = residue_2.upper() if isinstance(residue_2, str) else residue_2

    hydrophobic_pair = int(r1 in HYDROPHOBIC and r2 in HYDROPHOBIC)
    polar_pair = int(r1 in POLAR and r2 in POLAR)
    positive_negative_pair = int(
        (r1 in POSITIVE and r2 in NEGATIVE) or (r1 in NEGATIVE and r2 in POSITIVE)
    )
    positive_positive_pair = int(r1 in POSITIVE and r2 in POSITIVE)
    negative_negative_pair = int(r1 in NEGATIVE and r2 in NEGATIVE)
    aromatic_pair = int(r1 in AROMATIC and r2 in AROMATIC)

    return [
        hydrophobic_pair,
        polar_pair,
        positive_negative_pair,
        positive_positive_pair,
        negative_negative_pair,
        aromatic_pair,
    ]
