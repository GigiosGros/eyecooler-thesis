r"""Table alignment for the thesis (2026-11-12): header cells, the first column and the data columns centred.

For every tabular/longtable in a piece of LaTeX:
  * header rows = the rows between the first top-level \hline and the next one (for a longtable, the same in
    the \endfirsthead and \endhead blocks); each header cell is wrapped in \hc{...} (thesis preamble), a
    \multicolumn header gets spec c;
  * the first column is centred, and so is every other column whose body cells are short (numbers, short
    labels: at most PROSE visible characters); columns of running text stay left-aligned under their centred
    header. Centred: l/r -> c, L/R/p/m{w} -> C{w}.
A table whose body contains the comment "% centring: keep" is left as it is (the List of Abbreviations).
centre() is applied to thesis.tex directly and is called by the table scripts on their output.

Run:  python code/figures/table_centring.py FILE [--dry]
"""
import re
import sys

ENV = re.compile(r"\\begin\{(tabular|longtable)\}(\[[^\]]*\])?\{")
COL = re.compile(r"[LRCpmb]\{[^{}]*\}|[@<>!]\{(?:[^{}]|\{[^{}]*\})*\}|[lcr|]|\s+")
PROSE = 28


def braced(s, i):
    d, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return j + 1
        j += 1
    raise ValueError("unbalanced braces")


def top_level(body):
    """(index, token) of the top-level '\\\\', '\\hline' and '&' in body (outside braces, nested environments, comments)."""
    d = env = k = 0
    while k < len(body):
        if body[k] == "%" and (k == 0 or body[k - 1] != "\\"):
            k = body.find("\n", k)
            if k < 0:
                return
            continue
        if body.startswith("\\begin{", k):
            env += 1
        elif body.startswith("\\end{", k):
            env -= 1
        if body[k] == "\\":
            if d == 0 and env == 0:
                if body.startswith("\\\\", k):
                    yield k, "\\\\"
                elif body.startswith("\\hline", k):
                    yield k, "\\hline"
            k += 2
            continue
        if body[k] == "{":
            d += 1
        elif body[k] == "}":
            d -= 1
        elif body[k] == "&" and d == 0 and env == 0:
            yield k, "&"
        k += 1


def rows(seg):
    """Rows of seg as lists of cell texts (top level)."""
    out, row, last = [], [], 0
    for i, t in top_level(seg):
        if t == "\\hline":
            continue
        row.append(seg[last:i])
        last = i + len(t)
        if t == "\\\\":
            out.append(row)
            row = []
    return out


def visible(cell):
    t = "\n".join(l for l in cell.split("\n") if not l.lstrip().startswith("%"))
    t = re.sub(r"^\s*\[[^\]]*\]", "", t)                         # \\[4pt] remainder
    t = re.sub(r"\\(cite|ref|label)\{[^}]*\}", "[1]", t)
    t = re.sub(r"\\[A-Za-z]+\*?", "", t)
    t = re.sub(r"[{}$~^_]", "", t)
    return len(" ".join(t.split()))


def unwrap(body):
    """Remove an earlier \\hc{...} / \\hcr{...} around the whole cell."""
    m = re.match(r"\\hcr?\{", body)
    if m:
        end = braced(body, m.end() - 1)
        if not body[end:].strip():
            return body[m.end():end - 1]
    return body


def wrap(cell, paragraph):
    """Header cell centred: \\hc in a fixed-width (paragraph) column, \\hcr in an l/r/c column.
    Returns (text, number of columns the cell spans)."""
    lines = cell.split("\n")
    com = [l for l in lines if l.lstrip().startswith("%")]
    txt = "\n".join(l for l in lines if not l.lstrip().startswith("%"))
    a, body, z = re.match(r"(\s*)(.*?)(\s*)$", txt, re.S).groups()
    lead = ("\n".join(com) + "\n") if com else ""
    if not body:
        return lead + txt, 1
    mc = re.match(r"\\multicolumn\{([^}]*)\}\{([^}]*)\}\{", body)
    if mc:
        end = braced(body, mc.end() - 1)
        return (lead + a + f"\\multicolumn{{{mc.group(1)}}}{{c}}{{" + body[mc.end():end - 1] + "}" + body[end:] + z,
                int(mc.group(1)))
    body = unwrap(body).replace("\\begin{tabular}[b]{@{}r@{}}", "\\begin{tabular}[b]{@{}c@{}}")
    return lead + a + ("\\hc{" if paragraph else "\\hcr{") + body + "}" + z, 1


def header_regions(body, env):
    hl = [i for i, t in top_level(body) if t == "\\hline"]
    regs = []
    if env == "tabular":
        if len(hl) >= 2:
            regs.append((hl[0] + 6, hl[1]))
    else:
        stop = body.find("\\endhead")
        for a, b in zip(hl, hl[1:]):
            if b > stop >= 0:
                break
            seg = body[a + 6:b]
            if "\\endfirsthead" in seg or "\\caption" in seg:
                continue
            regs.append((a + 6, b))
    return regs


def col_types(spec):
    """For each column: True if it is a fixed-width (paragraph) column."""
    return [t[0] in "LRCpmb" for t in COL.findall(spec) if re.match(r"[lcr]$|[LRCpmb]\{", t)]


def wrap_region(seg, para):
    res, last, j = "", 0, 0
    for i, t in top_level(seg):
        if t == "\\hline":
            continue
        text, span = wrap(seg[last:i], para[j] if j < len(para) else False)
        res += text + (" & " if t == "&" else " \\\\")
        j = j + span if t == "&" else 0
        last = i + len(t)
    return res + seg[last:]


def centre_spec(spec, centre_cols):
    toks = COL.findall(spec)
    assert "".join(toks) == spec, ("unparsed column spec", spec)
    out, j = [], 0
    for t in toks:
        is_col = bool(re.match(r"[lcr]$|[LRCpmb]\{", t))
        if is_col:
            if j in centre_cols:
                t = "c" if t in ("l", "r") else ("C" + t[1:] if t[0] in "LRpm" else t)
            j += 1
        out.append(t)
    return "".join(out)


def ncols(spec):
    return sum(1 for t in COL.findall(spec) if re.match(r"[lcr]$|[LRCpmb]\{", t))


def process(s, verbose=False):
    out, pos, n = [], 0, 0
    for m in ENV.finditer(s):
        if m.start() < pos:
            continue
        env = m.group(1)
        spec_end = braced(s, m.end() - 1)
        spec = s[m.end():spec_end - 1]
        depth = 0
        for mm in re.finditer(r"\\(begin|end)\{" + env + r"\}", s[spec_end:]):
            if mm.group(1) == "begin":
                depth += 1
            elif depth == 0:
                end = spec_end + mm.start()
                break
            else:
                depth -= 1
        body = s[spec_end:end]
        if "% centring: keep" in body:
            out.append(s[pos:end])
            pos = end
            continue
        n += 1
        regs = header_regions(body, env)
        start = regs[-1][1] if regs else 0
        if env == "longtable" and "\\endhead" in body:
            start = body.index("\\endhead") + len("\\endhead")
        k = ncols(spec)
        width = [0] * k
        for r in rows(body[start:]):
            if any("\\multicolumn" in c for c in r) or len(r) != k:
                continue
            for j, c in enumerate(r):
                width[j] = max(width[j], visible(c))
        centre_cols = {0} | {j for j in range(k) if width[j] <= PROSE}
        new_spec = centre_spec(spec, centre_cols)
        para = col_types(new_spec)
        nb, last = "", 0
        for a, b in regs:
            nb += body[last:a] + wrap_region(body[a:b], para)
            last = b
        nb += body[last:]
        if verbose:
            line = s.count("\n", 0, m.start()) + 1
            print(f"line {line:5d} {env:9s} {{{spec}}} -> {{{new_spec}}}  widths {width}")
        out.append(s[pos:m.end()] + new_spec + "}" + nb)
        pos = end
    out.append(s[pos:])
    return "".join(out), n


def centre(tex):
    """The LaTeX with every table's header cells, first column and data columns centred."""
    return process(tex)[0]


if __name__ == "__main__":
    p = sys.argv[1]
    s = open(p, encoding="utf-8").read()
    t, n = process(s, verbose=True)
    if "--dry" not in sys.argv:
        open(p, "w", encoding="utf-8", newline="").write(t)
        print("written", n, "tables")
