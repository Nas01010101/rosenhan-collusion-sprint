"""Builds report.docx from report.md with the Apart template's look, for import into Google Docs.

The Apart Google Doc sets Old Standard TT as direct formatting while its docDefaults are Arial, so
using it as a pandoc reference-doc gives Arial. This edits pandoc's default reference styles instead,
then gives each table rules above, below and under its header, since Docs ignores table-style borders.
Usage: python build_docx.py [out.docx]  (needs pandoc on PATH; default out is build/report.docx)
"""

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "template/reference.docx"
FONT = '<w:rFonts w:ascii="{0}" w:hAnsi="{0}" w:eastAsia="{0}" w:cs="{0}" />'
RULE = '<w:{0} w:val="single" w:sz="{1}" w:space="{2}" w:color="000000" />'
# COMPACT=1: 10 pt, single spacing, 0.75-inch margins, for a body that must fit the 8-page cap
COMPACT = os.environ.get("COMPACT") == "1"
MARGIN, BODY_SZ, LINE = (1080, 20, 240) if COMPACT else (1440, 22, 276)
SECT = (f'<w:sectPr><w:pgSz w:w="12240" w:h="15840" /><w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" '
        f'w:bottom="{MARGIN}" w:left="{MARGIN}" w:header="720" w:footer="720" w:gutter="0" /></w:sectPr>')
TABLE_TEXT = ('<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableText"><w:name w:val="Table Text" />'
              '<w:basedOn w:val="Normal" /><w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto" />'
              '</w:pPr><w:rPr><w:sz w:val="18" /><w:szCs w:val="18" /></w:rPr></w:style>')
TEXT_WIDTH = 12240 - 2 * MARGIN  # twips between the margins on letter paper
SPACER = '<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="160" w:lineRule="exact" /></w:pPr></w:p>'

# styleId -> (pPr, rPr) that replace the style's own; None keeps what pandoc has
STYLES = {
    "BodyText": ('<w:spacing w:before="0" w:after="120" /><w:jc w:val="both" />', None),
    # left-aligned so bullets with long URLs don't stretch into gappy justified lines
    "Compact": ('<w:spacing w:before="20" w:after="20" /><w:jc w:val="left" />', None),
    "Title": ("<w:pBdr>" + RULE.format("top", 6, 8) + RULE.format("bottom", 6, 8) + "</w:pBdr>"
              '<w:spacing w:before="0" w:after="240" w:line="240" w:lineRule="auto" /><w:jc w:val="center" />',
              '<w:b /><w:sz w:val="40" /><w:szCs w:val="40" />'),
    "Author": ('<w:spacing w:before="0" w:after="120" /><w:jc w:val="center" />', '<w:sz w:val="22" /><w:szCs w:val="22" />'),
    "AbstractTitle": ('<w:keepNext /><w:spacing w:before="120" w:after="60" /><w:jc w:val="center" />',
                      '<w:b /><w:sz w:val="28" /><w:szCs w:val="28" />'),
    "Abstract": ('<w:spacing w:before="0" w:after="200" /><w:jc w:val="both" />', '<w:sz w:val="22" /><w:szCs w:val="22" />'),
    "Heading2": ('<w:keepNext /><w:keepLines /><w:spacing w:before="240" w:after="80" /><w:outlineLvl w:val="1" />',
                 '<w:b /><w:sz w:val="28" /><w:szCs w:val="28" />'),
    "Heading3": ('<w:keepNext /><w:keepLines /><w:spacing w:before="160" w:after="60" /><w:outlineLvl w:val="2" />',
                 '<w:b /><w:sz w:val="24" /><w:szCs w:val="24" />'),
    "Caption": ('<w:spacing w:before="60" w:after="160" /><w:jc w:val="both" />', '<w:sz w:val="19" /><w:szCs w:val="19" />'),
    "TableCaption": ('<w:keepNext /><w:spacing w:before="120" w:after="60" /><w:jc w:val="both" />', None),
    "CaptionedFigure": ('<w:keepNext /><w:spacing w:before="120" w:after="0" /><w:jc w:val="center" />', None),
    "FootnoteText": (None, '<w:sz w:val="18" /><w:szCs w:val="18" />'),
    "VerbatimChar": (None, FONT.format("Source Code Pro") + '<w:sz w:val="18" /><w:szCs w:val="18" />'),
}


def restyle(xml, sid, ppr, rpr):
    m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, xml, re.S)
    block = m.group(0)
    for tag, new in (("pPr", ppr), ("rPr", rpr)):
        if new is None:
            continue
        block = re.sub(r"<w:%s>.*?</w:%s>" % (tag, tag), "", block, flags=re.S)
        block = block.replace("</w:style>", f"<w:{tag}>{new}</w:{tag}></w:style>")
    return xml.replace(m.group(0), block)


def read_docx(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def write_docx(path, parts):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in parts.items():
            z.writestr(n, b)


def make_reference():
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                             capture_output=True, check=True).stdout
    REF.write_bytes(default)
    parts = read_docx(REF)
    styles = parts["word/styles.xml"].decode()
    # one font everywhere: drop theme font references and heading colours, set docDefaults
    styles = re.sub(r"<w:rFonts [^>]*/>", "", styles, flags=re.S)
    styles = re.sub(r"<w:color [^>]*/>", "", styles, flags=re.S)
    styles = re.sub(r"(<w:rPrDefault>\s*<w:rPr>).*?(</w:rPr>)", r"\1" + FONT.format("Old Standard TT")
                    + rf'<w:sz w:val="{BODY_SZ}" /><w:szCs w:val="{BODY_SZ}" /><w:lang w:val="en-US" />\2', styles, flags=re.S)
    styles = re.sub(r"(<w:pPrDefault>\s*<w:pPr>).*?(</w:pPr>)",
                    rf'\1<w:spacing w:after="120" w:line="{LINE}" w:lineRule="auto" />\2', styles, flags=re.S)
    for sid, (ppr, rpr) in STYLES.items():
        styles = restyle(styles, sid, ppr, rpr)
    parts["word/styles.xml"] = styles.replace("</w:styles>", TABLE_TEXT + "</w:styles>").encode()
    doc = parts["word/document.xml"].decode()
    parts["word/document.xml"] = re.sub(r"<w:sectPr>.*?</w:sectPr>", SECT, doc, flags=re.S).encode()
    write_docx(REF, parts)


def rule_tables(doc):
    """Rules above, below and under the header, full text width, Table Text cells, rows kept on one page."""
    def one(m):
        tbl = m.group(0)
        cols = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)" />', tbl)]
        widths = iter(round(w * TEXT_WIDTH / sum(cols)) for w in cols)
        tbl = re.sub(r'<w:gridCol w:w="\d+" />', lambda _: f'<w:gridCol w:w="{next(widths)}" />', tbl)
        borders = "<w:tblBorders>" + RULE.format("top", 8, 0) + RULE.format("bottom", 8, 0) + "</w:tblBorders>"
        tbl = re.sub(r"<w:tblBorders>.*?</w:tblBorders>", "", tbl, flags=re.S)
        tbl = tbl.replace("</w:tblPr>", borders + "</w:tblPr>", 1)
        head = re.search(r"<w:tr[ >].*?</w:tr>", tbl, re.S).group(0)
        under = "<w:tcBorders>" + RULE.format("bottom", 4, 0) + "</w:tcBorders>"
        ruled = re.sub(r"<w:tc><w:tcPr />|<w:tc><w:tcPr>|<w:tc>(?!<w:tcPr)",
                       lambda t: "<w:tc><w:tcPr>" + under + ("" if t.group(0) == "<w:tc><w:tcPr>" else "</w:tcPr>"),
                       head)
        tbl = tbl.replace(head, ruled, 1)
        tbl = re.sub(r'<w:pStyle w:val="[^"]*" />', '<w:pStyle w:val="TableText" />', tbl)
        tbl = re.sub(r"<w:p>(?!<w:pPr>)", '<w:p><w:pPr><w:pStyle w:val="TableText" /></w:pPr>', tbl)
        rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S)
        for row in rows[:-1]:  # keep-with-next on every row but the last holds the table on one page
            tbl = tbl.replace(row, row.replace('<w:pStyle w:val="TableText" />',
                                               '<w:pStyle w:val="TableText" /><w:keepNext />'), 1)
        return tbl + SPACER
    return re.sub(r"<w:tbl>.*?</w:tbl>", one, doc, flags=re.S)


def main():
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "build/report.docx"
    src = sys.argv[2] if len(sys.argv) > 2 else "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_reference()
    subprocess.run(["pandoc", src, "-o", str(out), f"--reference-doc={REF}", "--resource-path=.:.."],
                   cwd=HERE, check=True)
    parts = read_docx(out)
    parts["word/document.xml"] = rule_tables(parts["word/document.xml"].decode()).encode()
    write_docx(out, parts)
    print("wrote", out)


if __name__ == "__main__":
    main()
