"""v14 Stage 4: the arb result must not depend on the working precision beyond
the enclosure width. Seed 0, uniform 32x32, traces 1024, arb at 53/64/128 bits."""

import json

from certified_obstacle_nn.certification.enclose import upper_sum as US
from certified_obstacle_nn.certification.enclose.engine import StefanAdapter, TripleEncloser
from certified_obstacle_nn.experiments.v14.common import REPO_ROOT, STEFAN_SEEDS, load_frozen_stefan


def main():
    p, tr = load_frozen_stefan(STEFAN_SEEDS[0])
    enc = TripleEncloser(tr, StefanAdapter(p))
    part, tp = US.uniform_2d(32, 32), US.uniform_1d(1024)
    out = {}
    for prec in (53, 64, 128):
        us, _, _ = US.certify_partition(enc, part, tp, tp, backend="arb", prec=prec, workers=8,
                                        model_frozen=True, constants_validated=True)
        out[prec] = {"U_r": us.U_r, "U_p": us.U_p, "U_c": us.U_c, "U_r_hat": us.U_r_hat, "status": us.status,
                     "s_per_cell_wall_8workers": us.timings["s_per_cell_wall"]}
        print(prec, out[prec], flush=True)
    rel = {k: max(abs(out[a][k] - out[128][k]) / out[128][k] for a in (53, 64)) for k in ("U_r", "U_p", "U_c", "U_r_hat")}
    out["max_rel_diff_vs_128"] = rel
    (REPO_ROOT / "results/v14/stage4/precision_check.json").write_text(json.dumps(out, indent=1))
    print(rel)


if __name__ == "__main__":
    main()
