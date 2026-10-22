"""Convert docs/paper/draft_method_results.md to a self-contained LaTeX source (docs/paper/latex/main.tex + figs/).
    .venv/bin/python docs/paper/md2tex.py && cd docs/paper/latex && tectonic main.tex
Specific to this draft: # title, ## numbered sections, ### subsections, pipe tables preceded by a bold "Table …" line,
"- " bullets, a "Figure captions" subsection whose bullets become figure environments, **bold**, *italic*, `code`.
"""
import re
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent/'draft_method_results.md'
OUT = Path(__file__).resolve().parent/'latex'
FIGS = Path(__file__).resolve().parent/'figs'

UNI = {'≤': r'$\le$', '≥': r'$\ge$', '×': r'$\times$', '−': r'$-$', '≈': r'$\approx$', '≲': r'$\lesssim$', '±': r'$\pm$',
       '°': r'$^\circ$', '²': r'$^2$', '³': r'$^3$', '·': r'$\cdot$', 'φ₁': r'$\varphi_1$', 'φ₂': r'$\varphi_2$', 'φ': r'$\varphi$',
       'σ': r'$\sigma$', 'θ': r'$\theta$', 'τ': r'$\tau$', 'ε': r'$\varepsilon$', 'α': r'$\alpha$', 'β': r'$\beta$', 'ω': r'$\omega$',
       'Δ': r'$\Delta$', '∈': r'$\in$', 'Σ': r'$\sum$', '∑': r'$\sum$', '←': r'$\leftarrow$', '→': r'$\rightarrow$', '★': r'$\star$',
       '‖': r'$\|$', '—': '---', '–': '--', '…': r'\ldots{}', '≠': r'$\ne$', 'ℝ': r'$\mathbb{R}$', '₁': r'$_1$', '₂': r'$_2$',
       '⁵': r'$^5$', '⁶': r'$^6$', '⁻': r'$^-$', '⁰': r'$^0$', '✓': r'$\checkmark$', '✗': r'$\times$', 'μ': r'$\mu$', '′': "'",
       '“': '``', '”': "''", '‘': '`', '’': "'", '\u2009': r'\,', '\u00a0': '~', 'ρ': r'$\rho$', '∞': r'$\infty$', '≃': r'$\simeq$',
       '≡': r'$\equiv$', '∂': r'$\partial$', '√': r'$\surd$'}


def esc(t):
    t = t.replace('\\', r'\textbackslash{}')
    for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\^{}'), ('$', r'\$')):
        t = t.replace(a, b)
    SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')
    SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    t = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+', lambda m: '$^{'+m.group(0).translate(SUP)+'}$', t)
    t = re.sub(r'(?<![φ])[₀₁₂₃₄₅₆₇₈₉]+', lambda m: '$_{'+m.group(0).translate(SUB)+'}$', t)
    for a, b in UNI.items():
        t = t.replace(a, b)
    t = t.replace('$$', '')                      # adjacent math spans merge: "$\le$$\,$" -> fine either way
    return t


def inline(t):
    """markdown inline -> latex (code spans protected first)."""
    parts = re.split(r'(`[^`]*`)', t)
    out = []
    for p in parts:
        if p.startswith('`') and p.endswith('`') and len(p) >= 2:
            out.append(r'\texttt{'+esc(p[1:-1])+'}')
        else:
            q = esc(p)
            q = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', q)
            q = re.sub(r'(?<![\w\\])\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'\\emph{\1}', q)
            out.append(q)
    return ''.join(out)


def table(lines, caption):
    rows = [[c.strip() for c in l.strip().strip('|').split('|')] for l in lines if not re.match(r'^\s*\|?\s*-{3,}', l)]
    ncol = max(len(r) for r in rows)
    head, body = rows[0], rows[1:]
    spec = 'l'*ncol
    s = ['\\begin{table}[htbp]\\centering\\scriptsize']
    if caption:
        s.append('\\caption{'+inline(caption)+'}')
    s.append('\\resizebox{\\textwidth}{!}{\\begin{tabular}{'+spec+'}\\toprule')
    s.append(' & '.join(inline(c) for c in head+['']*(ncol-len(head)))+' \\\\ \\midrule')
    for r in body:
        s.append(' & '.join(inline(c) for c in r+['']*(ncol-len(r)))+' \\\\')
    s.append('\\bottomrule\\end{tabular}}\\end{table}')
    return '\n'.join(s)


def convert():
    md = SRC.read_text().splitlines()
    OUT.mkdir(exist_ok=True)
    (OUT/'figs').mkdir(exist_ok=True)
    for f in FIGS.glob('*.png'):
        shutil.copy(f, OUT/'figs'/f.name)
    body, i, title, pending_caption, in_captions = [], 0, 'Draft', None, False
    while i < len(md):
        l = md[i]
        if l.startswith('# '):
            title = l[2:].strip()
            i += 1
            continue
        if l.strip() == '---':
            i += 1
            continue
        if l.startswith('## '):
            h = re.sub(r'^\d+\.\s*', '', l[3:]).strip()
            body.append('\\section{'+inline(h)+'}')
            in_captions = False
            i += 1
            continue
        if l.startswith('### '):
            h = re.sub(r'^\d+\.\d+\s*', '', l[4:]).strip()
            in_captions = h.lower().startswith('figure captions')
            body.append('\\subsection{'+inline(h)+'}' if not in_captions else '\\subsection*{Figures}')
            i += 1
            continue
        if l.startswith('|'):
            j = i
            while j < len(md) and md[j].startswith('|'):
                j += 1
            body.append(table(md[i:j], pending_caption))
            pending_caption = None
            i = j
            continue
        if re.match(r'^\*\*Table [^*]+\*\*', l):
            pending_caption = re.sub(r'^\*\*(.+?)\*\*\s*', r'\1 ', l).strip()
            i += 1
            continue
        if l.startswith('- '):
            j = i
            items = []
            while j < len(md) and (md[j].startswith('- ') or (md[j].startswith('  ') and items)):
                if md[j].startswith('- '):
                    items.append(md[j][2:].strip())
                else:
                    items[-1] += ' '+md[j].strip()
                j += 1
            if in_captions:
                for it in items:
                    m = re.match(r'\*\*Fig\. ([^(]+?) \(`([^`]+)`\)\.\*\*\s*(.*)', it)
                    if m:
                        lab, fn, cap = m.group(1).strip(), m.group(2), m.group(3)
                        body.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{figs/'+fn+'}\\caption{\\textbf{Fig. '+inline(lab)+'.} '+inline(cap)+'}\\label{fig:'+re.sub(r'[^A-Za-z0-9]', '', lab)+'}\\end{figure}')
                    else:
                        body.append(inline(it))
            else:
                body.append('\\begin{itemize}\\setlength\\itemsep{2pt}')
                for it in items:
                    body.append('\\item '+inline(it))
                body.append('\\end{itemize}')
            i = j
            continue
        if l.strip() == '':
            body.append('')
            i += 1
            continue
        if l.startswith('*') and l.endswith('*') and not l.startswith('**'):
            body.append('\\noindent\\emph{'+inline(l.strip('*'))+'}')
            i += 1
            continue
        body.append(inline(l))
        i += 1
    tex = r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=2.2cm]{geometry}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{float}
\setlength{\parskip}{4pt}
\setcounter{section}{1}
\title{%s}
\author{}
\date{Draft, %s}
\begin{document}
\maketitle
%s
\end{document}
''' % (inline(title), __import__('datetime').date.today().isoformat(), '\n'.join(body))
    (OUT/'main.tex').write_text(tex)
    print('wrote', OUT/'main.tex', len(tex), 'chars;', len(list((OUT/'figs').glob('*.png'))), 'figures')


if __name__ == '__main__':
    convert()
