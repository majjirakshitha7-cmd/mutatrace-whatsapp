"""Generate a standalone interactive HTML dashboard to explore WhatsApp message variant families,
visual diffs, and misinformation propagation results."""
import json, re, difflib
from pathlib import Path
import pandas as pd
from src import config as C

pairs = pd.read_csv(C.PROC / "mutation_pairs_en.csv").dropna(subset=["root_en", "var_en"])
BAD = re.compile(r"fuck|shit|bitch|asshole|idiot|stupid|porn|whore", re.I)
pairs = pairs[~(pairs["root_en"].str.contains(BAD) | pairs["var_en"].str.contains(BAD))]
pairs = pairs[pairs["root_len"] <= 120].copy()

# Load propagation stats
prop_df = pd.read_csv(C.OUT / "misinformation_propagation.csv")
mis_tests = pd.read_csv(C.OUT / "misinformation_tests.csv")

samples = []
for idx, r in pairs.iterrows():
    ta, tb = r["root_en"].split(), r["var_en"].split()
    sm = difflib.SequenceMatcher(None, ta, tb)
    diff_html = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            diff_html.append(" ".join(ta[i1:i2]))
        elif tag == "delete":
            diff_html.append(f'<del class="diff-del">{" ".join(ta[i1:i2])}</del>')
        elif tag == "insert":
            diff_html.append(f'<ins class="diff-ins">{" ".join(tb[j1:j2])}</ins>')
        elif tag == "replace":
            diff_html.append(f'<span class="diff-rep"><del>{" ".join(ta[i1:i2])}</del> &rarr; <ins>{" ".join(tb[j1:j2])}</ins></span>')

    samples.append({
        "cluster": int(r["cluster"]),
        "family_size": int(r["family_size"]),
        "order": int(r["order_in_family"]),
        "root_text": r["root_en"],
        "var_text": r["var_en"],
        "diff_html": " ".join(diff_html),
        "header_added": bool(r["prefix_added"]),
        "footer_added": bool(r["suffix_added"]),
        "numbers_changed": bool(r["numbers_changed"]),
        "words_added": int(r["words_added"]),
        "words_removed": int(r["words_removed"]),
        "real_rewrites": int(r["real_rewrites"]),
        "sharings": int(r["max_sharings"]),
        "is_covid": bool(r["covid_family"])
    })

data_json = json.dumps(samples[:150])  # Top 150 cleanest examples
prop_json = prop_df.to_dict(orient="records")[0]
mis_tests_json = mis_tests.to_dict(orient="records")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>WhatsApp Message Mutation & Lineage Explorer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --primary: #128c7e;
      --primary-dark: #075e54;
      --bg: #f8f9fa;
      --card-bg: #ffffff;
      --text: #212529;
      --text-muted: #6c757d;
      --border: #e9ecef;
      --danger: #dc3545;
      --success: #28a745;
      --warning: #ffc107;
      --info: #17a2b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .header {{
      background: linear-gradient(135deg, var(--primary-dark), var(--primary));
      color: white;
      padding: 28px 32px;
      border-radius: 12px;
      margin-bottom: 24px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    .header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }}
    .header p {{ opacity: 0.9; font-size: 15px; max-width: 900px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      padding: 20px;
      border-radius: 10px;
      border: 1px solid var(--border);
      box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}
    .kpi-val {{ font-size: 28px; font-weight: 700; color: var(--primary-dark); }}
    .kpi-label {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-top: 4px; }}
    .main-grid {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 24px;
    }}
    @media (max-width: 960px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}
    .controls {{
      background: var(--card-bg);
      padding: 20px;
      border-radius: 10px;
      border: 1px solid var(--border);
      height: fit-content;
      position: sticky;
      top: 24px;
    }}
    .controls h3 {{ font-size: 16px; margin-bottom: 14px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }}
    .form-group {{ margin-bottom: 14px; }}
    .form-group label {{ display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }}
    .form-control {{
      width: 100%;
      padding: 8px 12px;
      border: 1px solid #ced4da;
      border-radius: 6px;
      font-size: 14px;
    }}
    .checkbox-group label {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      margin-bottom: 8px;
      cursor: pointer;
    }}
    .card-list {{ display: flex; flex-direction: column; gap: 16px; }}
    .pair-card {{
      background: var(--card-bg);
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .pair-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 600;
      border-radius: 12px;
      text-transform: uppercase;
    }}
    .badge-primary {{ background: #e0f2f1; color: var(--primary-dark); }}
    .badge-danger {{ background: #ffebee; color: var(--danger); }}
    .badge-warning {{ background: #fff8e1; color: #b78103; }}
    .badge-info {{ background: #e1f5fe; color: #0277bd; }}
    .section-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}
    .diff-box {{
      background: #fafbfc;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      font-size: 14px;
      line-height: 1.6;
      margin-top: 8px;
      word-break: break-word;
    }}
    .diff-del {{
      background: #ffeef0;
      color: #b31d28;
      text-decoration: line-through;
      padding: 2px 4px;
      border-radius: 3px;
    }}
    .diff-ins {{
      background: #e6ffed;
      color: #22863a;
      text-decoration: none;
      font-weight: 600;
      padding: 2px 4px;
      border-radius: 3px;
    }}
    .diff-rep {{
      background: #fffbdd;
      padding: 2px 4px;
      border-radius: 3px;
    }}
    .diff-rep del {{ color: #cb2431; }}
    .diff-rep ins {{ color: #22863a; text-decoration: none; font-weight: 600; }}
    .stat-row {{
      display: flex;
      gap: 16px;
      margin-top: 12px;
      font-size: 12px;
      color: var(--text-muted);
    }}
    .stat-item b {{ color: var(--text); }}
  </style>
</head>
<body>

  <div class="header">
    <h1>WhatsApp Message Mutation & Lineage Explorer</h1>
    <p>Interactive dashboard for inspecting variant families, mutation operators (headers, footers, number changes, and word rewrites), and fact-checking propagation in WhatsApp forwarding cascades.</p>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-val">17,672</div>
      <div class="kpi-label">Unique Messages Cleaned</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">604</div>
      <div class="kpi-label">Variant Families (984 Pairs)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">96.5%</div>
      <div class="kpi-label">Benchmark Recall (1.0 Prec)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">+22.1%</div>
      <div class="kpi-label">Fact-Check Coverage Boost</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val">AUC 0.50</div>
      <div class="kpi-label">Virality Predictability (Null Result)</div>
    </div>
  </div>

  <div class="main-grid">
    <div class="controls">
      <h3>Filter Variant Pairs</h3>
      <div class="form-group">
        <label for="search">Keyword Search</label>
        <input type="text" id="search" class="form-control" placeholder="Search text...">
      </div>
      <div class="form-group">
        <label>Mutation Types</label>
        <div class="checkbox-group">
          <label><input type="checkbox" id="chk-header"> Header Added (Prefix)</label>
          <label><input type="checkbox" id="chk-footer"> Footer Added (Suffix)</label>
          <label><input type="checkbox" id="chk-numbers"> Numbers Changed</label>
          <label><input type="checkbox" id="chk-rewrites"> Words Rewritten (>3)</label>
          <label><input type="checkbox" id="chk-covid"> COVID-19 Related</label>
        </div>
      </div>
      <div style="font-size: 12px; color: var(--text-muted); margin-top: 16px;">
        Showing <b id="count-display">0</b> variant pairs
      </div>
    </div>

    <div class="card-list" id="card-list">
      <!-- Injected by JavaScript -->
    </div>
  </div>

  <script>
    const data = {data_json};

    function renderCards() {{
      const q = document.getElementById("search").value.toLowerCase();
      const fHeader = document.getElementById("chk-header").checked;
      const fFooter = document.getElementById("chk-footer").checked;
      const fNumbers = document.getElementById("chk-numbers").checked;
      const fRewrites = document.getElementById("chk-rewrites").checked;
      const fCovid = document.getElementById("chk-covid").checked;

      const filtered = data.filter(d => {{
        if (q && !(d.root_text.toLowerCase().includes(q) || d.var_text.toLowerCase().includes(q))) return false;
        if (fHeader && !d.header_added) return false;
        if (fFooter && !d.footer_added) return false;
        if (fNumbers && !d.numbers_changed) return false;
        if (fRewrites && d.real_rewrites <= 3) return false;
        if (fCovid && !d.is_covid) return false;
        return true;
      }});

      document.getElementById("count-display").innerText = filtered.length;
      const container = document.getElementById("card-list");
      container.innerHTML = "";

      filtered.forEach(d => {{
        const card = document.createElement("div");
        card.className = "pair-card";

        let badges = `<span class="badge badge-primary">Family #` + d.cluster + ` (Size ` + d.family_size + `)</span>`;
        if (d.is_covid) badges += ` <span class="badge badge-danger">COVID-19</span>`;
        if (d.header_added) badges += ` <span class="badge badge-warning">Header Added</span>`;
        if (d.footer_added) badges += ` <span class="badge badge-info">Footer Added</span>`;
        if (d.numbers_changed) badges += ` <span class="badge badge-danger">Numbers Changed</span>`;

        card.innerHTML = `
          <div class="card-header">
            <div>` + badges + `</div>
            <div style="font-size:12px;color:var(--text-muted);">Variant #` + d.order + ` &bull; Max Sharings: <b>` + d.sharings + `</b></div>
          </div>
          <div class="section-title">Word-Level Mutation Diff (Original &rarr; Variant)</div>
          <div class="diff-box">` + d.diff_html + `</div>
          <div class="stat-row">
            <span class="stat-item">Words Added: <b>` + d.words_added + `</b></span>
            <span class="stat-item">Words Removed: <b>` + d.words_removed + `</b></span>
            <span class="stat-item">Rewrites: <b>` + d.real_rewrites + `</b></span>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    document.getElementById("search").addEventListener("input", renderCards);
    document.querySelectorAll(".checkbox-group input").forEach(el => el.addEventListener("change", renderCards));
    renderCards();
  </script>
</body>
</html>
"""

out_path = C.OUT / "variant_explorer.html"
out_path.write_text(html_content, encoding="utf-8")
print(f"Generated standalone interactive explorer: {out_path} ({len(html_content)} bytes)")
