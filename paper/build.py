"""Assemble the Intelligent Oncology submission package (Step 5.2). CPU only.

    python paper/build.py            # writes paper/submission/ and compiles the previews

Steps
  1. references.tex in order of first citation (AMA, superscript numerals via natbib)
  2. paper/submission/ with main.tex, references.tex, figures/Figure1-4 (PDF and PNG), highlights.txt
  3. preview PDF of main.tex with the stand-in class in paper/preview (needs pdflatex)
  4. manuscript.docx through pandoc, single column, left aligned, double spaced, line numbers
  5. supplement tables (scripts/make_supplement.py) and supplement.pdf
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "submission")
sys.path.insert(0, HERE)
from references import REFS  # noqa: E402

FIGS = {"Figure1": "fig1_privacy_utility", "Figure2": "fig4_h5_noise", "Figure3": "fig2_lira_roc",
        "Figure4": "fig3_explanations"}


def cite_order(tex: str) -> list[str]:
    order = []
    for grp in re.findall(r"\\cite\{([^}]*)\}", tex):
        for k in (x.strip() for x in grp.split(",")):
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in REFS]
    unused = [k for k in REFS if k not in order]
    if missing or unused:
        sys.exit(f"citation mismatch, missing {missing}, unused {unused}")
    return order


def references_tex(order: list[str]) -> str:
    items = "\n".join(f"\\bibitem{{{k}}} {REFS[k]}" for k in order)
    return f"\\begin{{thebibliography}}{{{len(order)}}}\n{items}\n\\end{{thebibliography}}\n"


def compress(nums: list[int]) -> str:
    nums, out, i = sorted(nums), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i:j + 1])))
        i = j + 1
    return ",".join(out)


def env_body(tex: str, env: str) -> str:
    m = re.search(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", tex, re.S)
    return m.group(1).strip() if m else ""


def docx_tex(tex: str, order: list[str]) -> str:
    """Plain LaTeX that pandoc can read, with numbered superscript citations and a numbered reference list."""
    num = {k: i + 1 for i, k in enumerate(order)}
    title = re.search(r"\\title\[[^]]*\]\{(.*?)\}\n\n", tex, re.S).group(1)
    abstract = env_body(tex, "abstract").replace("\\noindent", "")
    keywords = re.sub(r"\s*\\sep\s*", "; ", env_body(tex, "keywords"))
    body = tex.split("\\maketitle", 1)[1].split("\\end{document}")[0]
    body = re.sub(r"\\(linenumbers|RaggedRight)\b", "", body)
    body = re.sub(r"\\setlength\{\\parindent\}\{[^}]*\}", "", body)
    body = re.sub(r"\\cite\{([^}]*)\}",
                  lambda m: "\\textsuperscript{" + compress([num[k.strip()] for k in m.group(1).split(",")]) + "}",
                  body)
    # resolve cross-references and number captions, which pandoc leaves as [label] in Word
    labels, counts = {}, {"tab": 0, "fig": 0, "eq": 0, "sec": 0}
    for kind, name in re.findall(r"\\label\{(tab|fig|eq|sec):([^}]*)\}", tex):
        counts[kind] += 1
        labels[f"{kind}:{name}"] = str(counts[kind])
    sections = [m.start() for m in re.finditer(r"\\section\{", body)]
    for name in re.findall(r"\\label\{(sec:[^}]*)\}", body):
        pos = body.index(f"\\label{{{name}}}")
        labels[name] = str(sum(1 for p in sections if p < pos))
    body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    n = {"table": 0, "figure": 0}

    def cap(m):
        env = m.group(1)
        n[env] += 1
        return m.group(0) + ("Table " if env == "table" else "Fig. ") + f"{n[env]}. "
    body = re.sub(r"\\begin\{(table|figure)\}(?:\[[^]]*\])?(?:(?!\\caption\{).)*\\caption\{", cap, body, flags=re.S)
    body = re.sub(r"\\begin\{(table|figure)\}\[[^]]*\]", r"\\begin{\1}", body)
    body = re.sub(r"\\begin\{tabular\*\}\{\\tblwidth\}\{@\{\}([LRC]+)@\{\}\}",
                  lambda m: "\\begin{tabular}{" + m.group(1).lower() + "}", body)
    body = body.replace("\\end{tabular*}", "\\end{tabular}")
    body = re.sub(r"Figure(\d)\.pdf", r"Figure\1.png", body)
    refs = "\\section*{References}\n\\begin{enumerate}\n" + "\n".join(f"\\item {REFS[k]}" for k in order) + \
           "\n\\end{enumerate}\n"
    body = body.replace("\\input{references.tex}", refs)
    return ("\\documentclass{article}\n\\usepackage{amsmath,graphicx,booktabs}\n\\graphicspath{{figures/}}\n"
            f"\\title{{{title}}}\n\\begin{{document}}\n\\maketitle\n"
            "\\textbf{Authors.} Names, affiliations and the corresponding author's full contact details to be completed.\n\n"
            f"\\section*{{Abstract}}\n{abstract}\n\n\\textbf{{Keywords.}} {keywords}\n\n{body}\n\\end{{document}}\n")


def format_docx(path: str) -> None:
    """Journal Word layout. Times New Roman 12 pt, left aligned, double spacing, continuous line numbers."""
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    d = docx.Document(path)
    for st in d.styles:
        if st.type == 1 and st.font is not None:
            st.font.name = "Times New Roman"
            rpr = st.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                fonts.set(qn(a), "Times New Roman")
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if fonts.get(qn(a)) is not None:
                    del fonts.attrib[qn(a)]
            st.font.color.rgb = None
            if st.name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract"):
                st.font.size = Pt(12)
    for p in d.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if p.style.name not in ("Image Caption", "Table Caption", "Captioned Figure"):
            p.paragraph_format.line_spacing = 2.0
    for sec in d.sections:
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        sec._sectPr.append(ln)
    d.save(path)


def run(cmd, cwd, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-2000:])
        sys.exit(f"failed: {' '.join(cmd)}")


def main():
    tex = open(os.path.join(HERE, "main.tex")).read()
    order = cite_order(tex)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figures"))
    open(os.path.join(HERE, "references.tex"), "w").write(references_tex(order))
    for f in ("main.tex", "references.tex"):
        shutil.copy(os.path.join(HERE, f), OUT)
    for new, old in FIGS.items():
        for ext in ("pdf", "png"):
            shutil.copy(os.path.join(ROOT, "figures", f"{old}.{ext}"), os.path.join(OUT, "figures", f"{new}.{ext}"))
    hl = [re.sub(r"\$\\varepsilon\$", "ε", h.strip()) for h in env_body(tex, "highlights").split("\\item") if h.strip()]
    too_long = [h for h in hl if len(h) > 85]
    if too_long or not 3 <= len(hl) <= 5:
        sys.exit(f"highlights must be 3 to 5 items of at most 85 characters, got {too_long}")
    open(os.path.join(OUT, "highlights.txt"), "w").write("Highlights\n\n" + "\n".join(f"• {h}" for h in hl) + "\n")

    env = dict(os.environ, TEXINPUTS=os.path.join(HERE, "preview") + "//:")
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], OUT, env)
    shutil.copy(os.path.join(OUT, "main.pdf"), os.path.join(OUT, "manuscript_preview.pdf"))

    open(os.path.join(OUT, "_docx.tex"), "w").write(docx_tex(tex, order))
    run(["pandoc", "_docx.tex", "-f", "latex", "-t", "docx", "-o", "manuscript.docx", "--resource-path=.:figures"],
        OUT)
    format_docx(os.path.join(OUT, "manuscript.docx"))

    run([sys.executable, os.path.join(ROOT, "scripts", "make_supplement.py")], ROOT)
    sup = os.path.join(HERE, "supplement")
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "supplement.tex"], sup)
    shutil.copy(os.path.join(sup, "supplement.pdf"), os.path.join(OUT, "supplementary_material.pdf"))

    for f in os.listdir(OUT):
        if f.startswith("main.") and f not in ("main.tex",) or f.startswith("_docx"):
            os.remove(os.path.join(OUT, f))
    print(f"{len(order)} references, {len(hl)} highlights")
    print("\n".join(sorted(os.path.relpath(os.path.join(dp, f), OUT) for dp, _, fs in os.walk(OUT) for f in fs)))


if __name__ == "__main__":
    main()
