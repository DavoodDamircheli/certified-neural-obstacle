"""v14 Stage 4: validated per-cell enclosures (two backends, one interface).

intervals_arb  -- reference backend, python-flint arb balls
intervals_np   -- fast backend, NumPy float64 intervals with outward rounding
tm             -- first-order Taylor models over either backend
engine         -- Stefan-class field/density enclosures per cell and per trace cell
upper_sum      -- dyadic partitions, coverage checks, arb reduction, status, adaptivity
"""
