"""Build company-level metadata (industry, market-cap tier) for VeriPromiseESG4K.

Sources
-------
TWSE OpenAPI (public, no auth required):
  * opendata/t187ap03_L           -> industry code, paid-in capital, par value
  * exchangeReport/STOCK_DAY_ALL  -> closing price (snapshot date recorded in output)

Market capitalisation is approximated as

    shares_outstanding = paid_in_capital / par_value
    market_cap         = shares_outstanding * closing_price

which is the standard TWSE approximation for common shares. Note this is a
*current* snapshot, not the value at the time the 2024 ESG reports were
published; the snapshot date is written to the output so tiering is
reproducible and auditable.
"""
import csv
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "research", "data", "raw")
OUT = os.path.join(ROOT, "research", "data")

# TWSE listed-company industry codes. Codes appearing in this dataset were
# spot-validated against known classifications: 1101->01 cement, 1216->02 food,
# 2317->31 other-electronics, 2330->24 semiconductor, 2412->27 telecom,
# 2603->15 shipping, 2881->17 finance, 3231->25 computer-peripherals.
TWSE_INDUSTRY = {
    "01": ("水泥工業", "Cement"),
    "02": ("食品工業", "Food"),
    "03": ("塑膠工業", "Plastics"),
    "04": ("紡織纖維", "Textiles"),
    "05": ("電機機械", "Electric Machinery"),
    "06": ("電器電纜", "Electrical and Cable"),
    "08": ("玻璃陶瓷", "Glass and Ceramics"),
    "09": ("造紙工業", "Paper and Pulp"),
    "10": ("鋼鐵工業", "Iron and Steel"),
    "11": ("橡膠工業", "Rubber"),
    "12": ("汽車工業", "Automobile"),
    "14": ("建材營造", "Building Material and Construction"),
    "15": ("航運業", "Shipping and Transportation"),
    "16": ("觀光餐旅", "Tourism and Hospitality"),
    "17": ("金融保險", "Finance and Insurance"),
    "18": ("貿易百貨", "Trading and Consumer Goods"),
    "20": ("其他", "Other"),
    "21": ("化學工業", "Chemical"),
    "22": ("生技醫療業", "Biotechnology and Medical Care"),
    "23": ("油電燃氣業", "Oil, Gas and Electricity"),
    "24": ("半導體業", "Semiconductor"),
    "25": ("電腦及週邊設備業", "Computer and Peripheral Equipment"),
    "26": ("光電業", "Optoelectronic"),
    "27": ("通信網路業", "Communications and Internet"),
    "28": ("電子零組件業", "Electronic Parts and Components"),
    "29": ("電子通路業", "Electronic Products Distribution"),
    "30": ("資訊服務業", "Information Service"),
    "31": ("其他電子業", "Other Electronic"),
    "32": ("文化創意業", "Cultural and Creative"),
    "33": ("農業科技業", "Agricultural Technology"),
    "34": ("電子商務", "E-Commerce"),
    "35": ("綠能環保", "Green Energy and Environmental Services"),
    "36": ("數位雲端", "Digital and Cloud Services"),
    "37": ("運動休閒", "Sports and Leisure"),
    "38": ("居家生活", "Home and Living"),
}

# Coarse grouping used for the ML-Promise comparison. The ML-Promise Chinese
# (Taiwan) subset covers exactly three industries -- Semiconductor, Energy and
# Biomedical -- per Seki et al. (2025), Table 3.
MLPROMISE_GROUP = {
    "24": "Semiconductor",
    "23": "Energy",
    "22": "Biomedical",
}

DATA_FILES = [
    ("vpesg_4k_train_1000.json", "train"),
    ("vpesg4k_val_1000.json", "val"),
    ("vpesg4k_test_2000.json", "test"),
]


def load_rows():
    rows = []
    for fn, split in DATA_FILES:
        path = os.path.join(ROOT, "data", fn)
        d = json.load(open(path, encoding="utf-8"))
        if isinstance(d, dict):
            d = d.get("data", list(d.values())[0])
        for r in d:
            r["_split"] = split
        rows += d
    return rows


def main():
    rows = load_rows()

    comp = {}
    for r in rows:
        ticker = str(r.get("ticker")).strip()
        name = (r.get("company") or "").strip()
        entry = comp.setdefault(ticker, {
            "names": set(), "n": 0, "pdf_url": set(), "splits": set(),
        })
        entry["names"].add(name)
        entry["n"] += 1
        entry["pdf_url"].add(r.get("pdf_url"))
        entry["splits"].add(r["_split"])

    twse = {r["公司代號"]: r for r in json.load(
        open(os.path.join(RAW, "twse_listed_companies.json"), encoding="utf-8"))}
    price_rows = json.load(
        open(os.path.join(RAW, "twse_stock_day_all.json"), encoding="utf-8"))
    price = {r["Code"]: r for r in price_rows}
    snapshot = price_rows[0]["Date"]

    out = []
    for ticker, entry in comp.items():
        rec = twse.get(ticker)
        row = {
            "ticker": ticker,
            "company_key": sorted(entry["names"])[0],
            "name_variants": "|".join(sorted(entry["names"])),
            "n_instances": entry["n"],
            "n_pdf_urls": len(entry["pdf_url"]),
            "splits": "|".join(sorted(entry["splits"])),
            "twse_name": rec["公司簡稱"] if rec else "",
            "industry_code": rec["產業別"] if rec else "",
            "industry_zh": "",
            "industry_en": "",
            "mlpromise_group": "",
            "shares_outstanding": "",
            "close_price": "",
            "market_cap_ntd": "",
            "price_snapshot_roc_date": snapshot,
        }
        if rec:
            code = rec["產業別"]
            zh, en = TWSE_INDUSTRY.get(code, ("UNKNOWN", "UNKNOWN"))
            row["industry_zh"], row["industry_en"] = zh, en
            row["mlpromise_group"] = MLPROMISE_GROUP.get(code, "Other")
            cap = rec.get("實收資本額")
            par_raw = rec.get("普通股每股面額") or ""
            m = re.search(r"([\d.]+)", par_raw.replace(",", ""))
            if cap and m and float(m.group(1)) > 0:
                shares = float(cap) / float(m.group(1))
                row["shares_outstanding"] = int(shares)
                try:
                    p = float(str(price.get(ticker, {}).get("ClosingPrice", "")).replace(",", ""))
                    row["close_price"] = p
                    row["market_cap_ntd"] = int(shares * p)
                except (TypeError, ValueError):
                    pass
        out.append(row)

    # Market-cap tiers: global tertiles, and tertiles within each industry.
    capped = sorted([r for r in out if r["market_cap_ntd"] != ""],
                    key=lambda r: -r["market_cap_ntd"])
    n = len(capped)
    for i, r in enumerate(capped):
        r["mktcap_tier_global"] = ("High" if i < n / 3 else
                                   "Medium" if i < 2 * n / 3 else "Low")

    by_ind = {}
    for r in capped:
        by_ind.setdefault(r["industry_en"], []).append(r)
    for group in by_ind.values():
        g = len(group)
        for i, r in enumerate(group):
            r["mktcap_tier_within_industry"] = ("High" if i < g / 3 else
                                                "Medium" if i < 2 * g / 3 else "Low")

    for r in out:
        r.setdefault("mktcap_tier_global", "")
        r.setdefault("mktcap_tier_within_industry", "")

    out.sort(key=lambda r: (r["industry_en"], -(r["market_cap_ntd"] or 0)))
    cols = ["ticker", "company_key", "name_variants", "n_instances", "n_pdf_urls",
            "splits", "twse_name", "industry_code", "industry_zh", "industry_en",
            "mlpromise_group", "shares_outstanding", "close_price", "market_cap_ntd",
            "mktcap_tier_global", "mktcap_tier_within_industry",
            "price_snapshot_roc_date"]

    dest = os.path.join(OUT, "company_metadata.csv")
    with open(dest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    print("wrote %s  (%d tickers, TWSE price snapshot ROC %s)" % (dest, len(out), snapshot))

    missing = [r["ticker"] for r in out if not r["industry_code"]]
    if missing:
        print("  tickers absent from TWSE listed set:", missing)
    nocap = [r["ticker"] for r in out if r["market_cap_ntd"] == ""]
    if nocap:
        print("  tickers without market cap:", nocap)
    for r in out:
        if len(r["name_variants"].split("|")) > 1:
            print("  NOTE name-variant collision: %s -> %s" % (r["ticker"], r["name_variants"]))

    counts = {}
    for r in out:
        counts[r["industry_en"]] = counts.get(r["industry_en"], 0) + 1
    print("\n  industry distribution (companies):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print("    %-40s %d" % (k, v))


if __name__ == "__main__":
    main()
