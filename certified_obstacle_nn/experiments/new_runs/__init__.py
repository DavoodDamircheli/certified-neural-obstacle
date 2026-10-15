"""Shared infrastructure for the NEW_RUNS.md experiments (Phase 1, 2026-09-13).

Every module under experiments/new_runs/ is additive: nothing here is
imported by any existing validated experiment script, and nothing in
training/ or certification/ was modified to build it. See common.py for the
run-record writer, sanity checks, CSV aggregator, and figure helper shared by
all four new experiments (adaptive_multiseed, american_put_improved,
certificate_aware, threshold_sensitivity).
"""
