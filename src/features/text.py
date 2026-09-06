"""Mô-đun trích xuất cờ tiện ích (Binary Text Flags) từ tiêu đề và mô tả tin rao."""

import re
import pandas as pd

KEYWORDS: dict[str, list[str]] = {
    "has_furniture": ["nội thất", "full nội thất"],
    "car_alley": ["hẻm xe hơi", "hẻm ô tô", "hẻm oto", "ô tô vào", "oto vào", "hxh"],
    "near_market": ["gần chợ", "sát chợ"],
    "near_school": ["gần trường", "đại học", "trường học"],
    "is_urgent_sale": ["cần bán gấp", "bán gấp", "chính chủ"],
}

# Tiền tố phủ định tiếng Việt thường gặp
NEGATION_PATTERN: str = r"(?:không|chưa|chẳng|ko|chua|khong)\s+(?:có\s+|được\s+)?"


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất các cờ nhị phân (0 hoặc 1) có xử lý từ phủ định.

    Nếu người dùng đã chủ động truyền cờ trong dữ liệu đầu vào (ví dụ form API),
    sẽ ưu tiên giữ giá trị đó.
    """
    out = df.copy()
    title = out["Title"] if "Title" in out else pd.Series("", index=out.index)
    description = (
        out["Description"]
        if "Description" in out
        else pd.Series("", index=out.index)
    )

    raw_text = (
        title.fillna("").astype(str)
        + " "
        + description.fillna("").astype(str)
    ).str.lower()

    for flag, keywords in KEYWORDS.items():
        kw_pattern = "|".join(re.escape(k) for k in keywords)
        neg_kw_pattern = rf"{NEGATION_PATTERN}(?:{kw_pattern})"

        # Loại bỏ các cụm bị phủ định trước khi quét từ khóa khẳng định
        sanitized_text = raw_text.str.replace(neg_kw_pattern, "", regex=True)
        extracted = sanitized_text.str.contains(kw_pattern, regex=True).astype(int)

        if flag in out:
            supplied = pd.to_numeric(out[flag], errors="coerce")
            out[flag] = supplied.where(supplied.notna(), extracted).clip(0, 1).astype(int)
        else:
            out[flag] = extracted

    return out
