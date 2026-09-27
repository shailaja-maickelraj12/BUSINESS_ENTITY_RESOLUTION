"""Generate professional Word Document (.docx) report for Amazon ML Challenge 2026."""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set internal padding for table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def create_document():
    doc = docx.Document()

    # Page Margins (1 inch all around)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Palette
    PRIMARY = RGBColor(16, 44, 87)       # Deep Navy
    SECONDARY = RGBColor(53, 89, 142)    # Slate Blue
    TEXT_COLOR = RGBColor(33, 37, 41)    # Dark Charcoal
    ACCENT = RGBColor(218, 98, 23)       # Burnt Orange
    LIGHT_BG = "F4F6F9"
    HEADER_BG = "1F3A60"

    # Base Styles
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = TEXT_COLOR

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("Amazon ML Challenge 2026")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(26)
    r_title.font.bold = True
    r_title.font.color.rgb = PRIMARY

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    r_sub = p_sub.add_run("Business Entity Resolution — Technical Architecture & Methodology Report")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(15)
    r_sub.font.color.rgb = SECONDARY

    # Metadata Box Table
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Team Name", "Team Antigravity"),
        ("Challenge Track", "Business Entity Resolution (Multi-Source Matching)"),
        ("Evaluation Metric", "Macro-average F_0.5 (Precision-Weighted, Singletons Included)"),
        ("Supported Geographies", "United States, India, and France (Open String Label)"),
        ("Date", "September 2026"),
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        cell_k = meta_table.cell(row_idx, 0)
        cell_v = meta_table.cell(row_idx, 1)
        cell_k.text = k
        cell_v.text = v
        cell_k.paragraphs[0].runs[0].font.bold = True
        cell_k.paragraphs[0].runs[0].font.color.rgb = PRIMARY
        set_cell_background(cell_k, "EAEEF3")
        set_cell_background(cell_v, "FAFBFC")
        set_cell_margins(cell_k, top=80, bottom=80, left=120, right=120)
        set_cell_margins(cell_v, top=80, bottom=80, left=120, right=120)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = PRIMARY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = SECONDARY
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.bold = True
            r_bold.font.color.rgb = PRIMARY
        p.add_run(text)
        return p

    def format_table(table, headers, rows_data):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        # Header Row
        hdr_cells = table.rows[0].cells
        for col_idx, header_text in enumerate(headers):
            hdr_cells[col_idx].text = header_text
            set_cell_background(hdr_cells[col_idx], HEADER_BG)
            set_cell_margins(hdr_cells[col_idx], top=100, bottom=100, left=120, right=120)
            p = hdr_cells[col_idx].paragraphs[0]
            if p.runs:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        # Data Rows
        for row_idx, r_data in enumerate(rows_data, start=1):
            row_cells = table.rows[row_idx].cells
            bg = LIGHT_BG if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(r_data):
                row_cells[col_idx].text = str(val)
                set_cell_background(row_cells[col_idx], bg)
                set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)

    # Section 1: Executive Summary
    add_h1("1. Executive Summary")
    p = doc.add_paragraph()
    p.add_run(
        "We designed and implemented a production-grade, offline, multi-stage machine learning system "
        "for enterprise-scale Business Entity Resolution (ER) across heterogeneous data sources. "
        "The architecture integrates jurisdiction-aware normalization, 5-channel inverted index candidate blocking, "
        "and a gradient boosted decision tree (LightGBM) trained on 34 discriminative pairwise and group-ranking features. "
        "By optimizing the decision threshold specifically for the competition's precision-heavy Macro-F0.5 metric "
        "with singleton protection (tau = 0.69), the system achieves a 96.72% blocking recall ceiling with a 99.966% reduction ratio "
        "(~34 candidates per reference entity), and delivers 97.27% Macro-F0.5 (99.18% precision, 97.37% singleton identification accuracy) "
        "on the held-out validation benchmark."
    )

    # Section 2: Problem Analysis
    add_h1("2. Problem Analysis & Core Challenges")
    add_bullet(
        " Reference Source 1 is deduplicated; Candidate Sources 2 and 3 contain noisy, unstandardized business records.",
        "Deduplication vs Multiplicity:"
    )
    add_bullet(
        " Heavy variations in entity designations across US (Inc vs Incorporated, LLC vs L.L.C.), India (Pvt Ltd vs Private Limited), and France (SARL, SAS, SA, EURL).",
        "Legal Suffix Noise:"
    )
    add_bullet(
        " Severe permutations of address components (street before city, pin code before state), omitted postal codes, and landmark references.",
        "Address Distortion:"
    )
    add_bullet(
        " The test set introduces France, which does not exist in training data. The entire pipeline treats country as an open string label without hardcoded filtering.",
        "Unseen Geographies (France):"
    )
    add_bullet(
        " Approximately 65% of Source 1 entities have zero matching records in S2/S3. Under the Macro-F0.5 scoring rule, correctly predicting an empty list yields 1.0, while false merges yield 0.0.",
        "Singleton Preponderance:"
    )

    # Section 3: Metric Formulation
    add_h1("3. Evaluation Metric Formulation (Macro-F0.5)")
    doc.add_paragraph(
        "Submissions are evaluated using Macro-average F_0.5 (beta = 0.5), which penalizes false merges (false positives) twice as severely as missed matches:"
    )
    p_formula = doc.add_paragraph()
    p_formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f = p_formula.add_run("F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)")
    r_f.font.bold = True
    r_f.font.color.rgb = PRIMARY
    r_f.font.size = Pt(12)

    doc.add_paragraph(
        "For each Source 1 entity: if no true matches exist (singleton), predicting an empty list scores 1.0; predicting any match scores 0.0. "
        "The overall score is the unweighted arithmetic mean over all Source 1 entities in the evaluation set."
    )

    # Section 4: Multi-Channel Blocking
    add_h1("4. Candidate Generation (Multi-Channel Blocking)")
    doc.add_paragraph(
        "Comparing 1.73M reference entities against 10M candidate records requires evaluating over 1.7 x 10^13 pairs. "
        "Our candidate generator cuts this search space by 99.966% using 5 complementary retrieval channels:"
    )
    add_bullet(" Exact match on core business name (legal suffixes stripped) and alphabetically sorted token signatures.", "Channel 1 (Exact Core Name):")
    add_bullet(" Inverted index on rare name tokens occurring in <= 500 records, returning up to 10 candidates per rare token.", "Channel 2 (Rare Token Index):")
    add_bullet(" Inverted index on postal codes (ZIP/PIN) and building/house numbers, capped at 500 records.", "Channel 3 (Structured Address):")
    add_bullet(" Sub-word character 3-gram TF-IDF cosine retrieval (threshold >= 0.50, top-k=5) to catch typos and transpositions.", "Channel 4 (Char 3-Gram TF-IDF):")
    add_bullet(" Full-text TF-IDF cosine retrieval (threshold >= 0.40, top-k=3) for blended name and address records.", "Channel 5 (Combined TF-IDF):")

    add_h2("Blocking Evaluation Benchmarks")
    t_block = doc.add_table(rows=4, cols=6)
    b_headers = ["Configuration", "Pair Recall", "Entity Recall", "Mean Cands/S1", "Reduction Ratio", "Runtime"]
    b_rows = [
        ["Iteration A (Baseline)", "88.22%", "70.30%", "30.89", "99.9696%", "2.0s"],
        ["Iteration B (+Char & Combined)", "96.72%", "91.24%", "34.71", "99.9659%", "12.2s"],
        ["Iteration C (Tuned Lean)", "93.73%", "83.76%", "24.36", "99.9760%", "13.2s"],
    ]
    format_table(t_block, b_headers, b_rows)

    p_geo = doc.add_paragraph()
    p_geo.paragraph_format.space_before = Pt(8)
    p_geo.add_run("Geographic Breakdown: ").font.bold = True
    p_geo.add_run("US Pair Recall Ceiling: 99.23% | India Pair Recall Ceiling: 92.78% | Source 2 Recall: 97.98% | Source 3 Recall: 95.49%.")

    # Section 5: Feature Engineering
    add_h1("5. Pairwise Feature Engineering (34 Dimensions)")
    doc.add_paragraph("For each generated candidate pair, our feature engine computes 34 discriminative metrics:")
    add_bullet(
        " Levenshtein ratio, partial ratio, token sort ratio, token set ratio, core name sort, character 3-gram and 4-gram Jaccard, prefix 3/5 matches, first token exact match, legal suffix match compatibility, length difference and ratio.",
        "Name Features (14):"
    )
    add_bullet(
        " Address Levenshtein, token sort, token set, token Jaccard, postal code exact match (-1 missing, 0 mismatch, 1 match), house number exact match, numeric token Jaccard overlap, address missingness indicators, length ratio.",
        "Address Features (11):"
    )
    add_bullet(
        " Full combined token set ratio, candidate source flags (is_s2, is_s3), country indicators (country_is_us, country_is_in, country_is_fr).",
        "Combined & Source Features (6):"
    )
    add_bullet(
        " candidate_rank (ordinal rank of candidate for this S1 entity sorted by name similarity), name_sort_diff_to_best (similarity margin against the #1 candidate), total_candidates_for_s1 (cluster density proxy).",
        "Entity Group Ranking Features (3):"
    )

    # Section 6: Model Training & Threshold Tuning
    add_h1("6. Matching Model & Macro-F0.5 Optimization")
    doc.add_paragraph(
        "We train a LightGBM Gradient Boosted Decision Tree (LGBMClassifier) with 350 trees, max depth 6, and 31 leaves. "
        "The model is evaluated using strict entity-grouped validation to prevent data leakage. "
        "A threshold grid search tau in [0.35, 0.90] was conducted directly against the competition Macro-F0.5 metric:"
    )

    t_thresh = doc.add_table(rows=8, cols=5)
    th_headers = ["Threshold (tau)", "Macro-F0.5", "Macro-Precision", "Macro-Recall", "Singleton Accuracy"]
    th_rows = [
        ["0.35", "96.94%", "98.54%", "95.09%", "95.61%"],
        ["0.45", "97.13%", "98.81%", "94.93%", "96.49%"],
        ["0.55", "97.11%", "98.88%", "94.78%", "96.49%"],
        ["0.65", "97.26%", "99.10%", "94.65%", "97.37%"],
        ["0.69 (Optimal)", "97.27%", "99.18%", "94.47%", "97.37%"],
        ["0.75", "97.12%", "99.21%", "94.10%", "97.37%"],
        ["0.85", "96.98%", "99.35%", "93.52%", "98.25%"],
    ]
    format_table(t_thresh, th_headers, th_rows)

    add_h2("Top 10 Feature Importances (Split Count)")
    t_feat = doc.add_table(rows=11, cols=3)
    f_headers = ["Rank", "Feature Name", "Importance (Splits)"]
    f_rows = [
        ["1", "full_token_set", "685"],
        ["2", "name_core_sort", "605"],
        ["3", "name_lev", "531"],
        ["4", "total_candidates_for_s1", "529"],
        ["5", "name_partial", "501"],
        ["6", "name_jaccard", "490"],
        ["7", "addr_len_ratio", "476"],
        ["8", "addr_token_set", "453"],
        ["9", "candidate_rank", "451"],
        ["10", "name_len_ratio", "437"],
    ]
    format_table(t_feat, f_headers, f_rows)

    # Section 7: Inference Engine
    add_h1("7. Streaming Test Inference Engine")
    doc.add_paragraph(
        "To process 1.73M test entities against 10M candidate records within desktop memory constraints, "
        "the inference engine utilizes two architectural mechanisms:"
    )
    add_bullet(
        " Records are partitioned by France, US, and India. Since matches never cross international borders, candidate indexes are constructed and deallocated per country, capping RAM usage at < 3.5 GB.",
        "Country-Partitioned Isolation:"
    )
    add_bullet(
        " Test Source 1 records are evaluated in 50,000-entity chunks and appended directly to matching_results.tsv and candidate_pairs.tsv.",
        "Chunked Streaming Output:"
    )
    add_bullet(
        " Predictions are strictly filtered to guarantee matched_entity_ids is a subset of candidate_entity_ids on 100% of rows.",
        "Subset Invariant Enforcement:"
    )

    # Section 8: Deliverables & Verification
    add_h1("8. Verification & Submission Package Structure")
    doc.add_paragraph("The submission package satisfies all formatting, legal, and directory structure requirements:")
    add_bullet(" Verified clean with zero delimiter, quoting, or duplicate ID issues (Exit Code 0).", "Official Validator Check:")
    add_bullet(" matching_results.tsv (leaderboard) and candidate_pairs.tsv (candidate audit) in output/.", "Output Files:")
    add_bullet(" Runnable pipeline in code/business_entity_resolution/src/ with README.md and requirements.txt.", "Reproducible Code:")
    add_bullet(" Complete methodology report filled in Documentation_template.md.", "Documentation:")

    # Save
    output_docx_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "PROJECT_DOCUMENTATION.docx")
    output_docx_path = os.path.abspath(output_docx_path)
    doc.save(output_docx_path)
    print(f"Generated DOCX document successfully at: {output_docx_path}")
    return output_docx_path


if __name__ == "__main__":
    create_document()
