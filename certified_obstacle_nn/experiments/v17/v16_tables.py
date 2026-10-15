"""Writers of the v16 manuscript Tables 2-4 (tab:stefan-cert, tab:two-dimensional, tab:disk-relative) in the exact
format of docs/v16_reference/v16/scripts/build_assets.py (captions, labels, column formats, upward decimal rounding).
They take records whose bound columns were computed by certification/reliability.py; used by the v17 Stage 1 gate
to show that the v17 code path regenerates the v16 tables byte-for-byte. Output goes wherever the caller says
(never into docs/v16_reference/)."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from pathlib import Path


def ceilfmt(x, n=6) -> str:
    """Decimal formatting that never decreases a certified upper endpoint (build_assets.ceilfmt)."""
    return str(Decimal.from_float(float(x)).quantize(Decimal(1).scaleb(-n), rounding=ROUND_CEILING))


def table(path: Path, caption, label, header, body, cols):
    t = ('\\begin{table}[htbp]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\small\n'
         '\\begin{tabular}{' + cols + '}\n\\toprule\n' + header + '\\\\\n\\midrule\n'
         + '\n'.join(' & '.join(x) + '\\\\' for x in body) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    Path(path).write_text(t)


def write_v16_tables(rows: list[dict], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = sorted([r for r in rows if r['problem'] == 'stefan' and r['mode'] == 'uniform' and int(r['size']) == 262144],
               key=lambda r: int(r['seed']))
    table(out_dir / 'stefan.tex', 'Stefan, $512^2$ cells, Arb backend. $E(\\widehat v)$ is the diagnostic weak-norm error of the projected triple. $\\widehat G_U=\\widehat F_U$ is the estimator-route bound \\eqref{eq:projected-gradient-flux}; $\\widehat F_{\\rm dir}$ and $\\widehat G_{\\rm dir}$ are the direct bounds \\eqref{eq:direct-nosplit} with $\\overline c_F^{\\,2}=1/\\pi^2$ rounded upward. All bounds and ratios use validated arithmetic with the analytic reference norm $N$; decimals are rounded upward.', 'tab:stefan-cert', 'Seed & $E(\\widehat v)$ & $\\widehat G_U$ & $\\widehat G_U/N$ & $\\widehat F_{\\rm dir}$ & $\\widehat F_{\\rm dir}/N$ & $\\widehat G_{\\rm dir}$ & $\\widehat G_{\\rm dir}/N$', [[r['seed'], f"{float(r['E_true_vhat']):.5f}", ceilfmt(r['grad_bound_projected'], 4), ceilfmt(r['projected_relative_upper'], 3), ceilfmt(r['flux_bound_direct'], 4), ceilfmt(r['flux_direct_relative_upper'], 3), ceilfmt(r['grad_bound_direct'], 4), ceilfmt(r['grad_direct_relative_upper'], 3)] for r in S], 'rrrrrrrr')
    D = [r for r in rows if r['problem'] in ('disk2d', 'ellipse') and int(r['size']) == 1000000]
    D.sort(key=lambda r: (r['problem'], r['schedule'], int(r['seed'])))
    table(out_dir / 'two_dimensional.tex', 'Two-dimensional results at $10^6$ cells, validated NumPy backend. $E(v)$, $E(\\widehat v)$ and the effectivity ratios are diagnostic. $E_U$ is the weighted original-triple bound, $\\widehat E_U$ the estimator-route projected bound and $\\widehat E_{\\rm dir}$ the direct weak-norm bound; the last two columns are the direct flux and gradient bounds \\eqref{eq:direct-nosplit}. Absolute bounds are rounded upward.', 'tab:two-dimensional', 'Problem & Seed & $E(v)$ & $E_U$ & $\\widehat E_U$ & $\\widehat E_{\\rm dir}$ & $\\widehat E_{\\rm dir}/E(\\widehat v)$ & $\\widehat F_{\\rm dir}$ & $\\widehat G_{\\rm dir}$', [[('Disk ' + r['schedule'] if r['problem'] == 'disk2d' else 'Ellipse'), r['seed'], f"{float(r['E_true_v']):.4f}", ceilfmt(r['E_U_weighted'], 2), ceilfmt(r['E_U_projected'], 2), ceilfmt(r['E_U_direct'], 2), f"{float(r['ratio_E_U_direct_vs_vhat']):.1f}", ceilfmt(r['flux_bound_direct'], 3), ceilfmt(r['grad_bound_direct'], 3)] for r in D], 'lrrrrrrrr')
    DD = [r for r in D if r['problem'] == 'disk2d']
    table(out_dir / 'disk_relative.tex', 'Expanding disk at $10^6$ cells: direct bounds relative to the analytic reference norm $N=\\|\\nabla_xu\\|_{L^2(Q)}=\\|\\sigma\\|_{L^2(Q)}$ of \\Cref{app:arithmetic}, validated (upward rounding, lower endpoint of $N$). The estimator-route ratio $\\widehat G_U/N$ is shown for comparison.', 'tab:disk-relative', 'Schedule & Seed & $\\widehat G_U/N$ & $\\widehat F_{\\rm dir}/N$ & $\\widehat G_{\\rm dir}/N$', [[r['schedule'], r['seed'], ceilfmt(r['projected_relative_upper'], 2), ceilfmt(r['flux_direct_relative_upper'], 2), ceilfmt(r['grad_direct_relative_upper'], 2)] for r in DD], 'lrrrr')
