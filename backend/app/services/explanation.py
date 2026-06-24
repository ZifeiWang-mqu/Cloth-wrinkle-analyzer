"""Japanese explanation generation.

Turns a requirement type + score + supporting feature values into a short,
human-readable Japanese message for the artist. Kept separate from the rule
logic so wording can be iterated without touching scoring.
"""

from __future__ import annotations

from app.services.feature_extraction import Features

# 日本語の表示名（type -> label）
LABELS: dict[str, str] = {
    "gravity_inconsistency": "重力と皺の矛盾",
    "joint_inconsistency": "関節と皺の矛盾",
    "tension_ambiguity": "張力点が不明瞭",
    "body_volume_inconsistency": "体の立体と皺の矛盾",
    "density_inconsistency": "皺の密度が不自然",
    "shadow_wrinkle_mismatch": "陰影と皺の不一致",
}


def label_for(issue_type: str) -> str:
    return LABELS.get(issue_type, issue_type)


def _intensity_word(score: float) -> str:
    if score >= 0.75:
        return "強く"
    if score >= 0.55:
        return "ややはっきりと"
    return "わずかに"


def generate_explanation(issue_type: str, score: float, features: Features) -> str:
    """Return a Japanese explanation for one detected issue."""
    deg = _intensity_word(score)

    if issue_type == "gravity_inconsistency":
        return (
            f"皺の主方向が水平寄り（基準軸から約{features.angle_diff_from_gravity:.0f}度）で、"
            f"重力で垂れる布の流れと{deg}矛盾しています。"
            "縦方向に落ちる皺になっているか確認してください。"
        )

    if issue_type == "joint_inconsistency":
        joint = features.nearest_joint or "関節"
        if features.joint_angle is not None:
            return (
                f"{joint}（約{features.joint_angle:.0f}度に屈曲）の圧縮側に皺が少なく、"
                "伸び側との分布が皺の出方と{deg}矛盾している可能性があります。"
                "曲げ内側（圧縮側）に皺が集まり、外側が滑らかになっているか確認してください。"
            ).replace("{deg}", deg)
        return (
            f"{joint}付近に皺が{deg}不自然に分布しています。"
            "曲げた側（圧縮側）に皺が集まっているか、伸びる側に偏っていないか確認してください。"
        )

    if issue_type == "tension_ambiguity":
        return (
            f"皺が一点へ{deg}収束していますが、近くに肩・袖口・腰・ベルトなどの"
            "張力点が見当たりません。引っ張られる起点が妥当か確認してください。"
        )

    if issue_type == "body_volume_inconsistency":
        return (
            f"皺が{deg}直線的・平行的で、胸や腰・腕脚の円柱的な立体に沿っていない可能性があります。"
            "体のボリュームに巻き込まれる曲がりがあるか確認してください。"
        )

    if issue_type == "density_inconsistency":
        return (
            f"皺の密度が基準から{deg}外れています（過多または過少、もしくは一部への集中）。"
            "肘・膝・袖口など皺が出やすい場所と、平面的な場所の差を確認してください。"
        )

    if issue_type == "shadow_wrinkle_mismatch":
        return (
            f"皺線と明暗勾配の向きが{deg}噛み合っていません。"
            "皺の片側に出る影の向きが、全体の光源方向と整合しているか確認してください。"
        )

    return f"{label_for(issue_type)}の可能性があります。"
