"""Reusable code for the data-driven fixed-bundle UROP.

The live pipeline estimates latent preference scores from sparse implicit
feedback and later evaluates CP-anchored fixed-bundle SBA policies under
declared pseudo-utility scenarios. ``bundle_pricing`` and ``calibration``
preserve the retired CMM and failed monetary-anchor work as archive modules;
they are not the live optimizer or an identified valuation interface.
"""
