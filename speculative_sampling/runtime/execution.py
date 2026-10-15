"""Allowlisted execution choices; never rewrite model or sampling semantics."""
from dataclasses import replace


def apply_native_execution(cfg, options):
    if options is None:
        return cfg
    if (not isinstance(options, dict) or set(options) != {"reshard_after_forward"}
            or type(options["reshard_after_forward"]) is not bool):
        raise ValueError("unsupported native execution option")
    return replace(cfg, parallel=replace(
        cfg.parallel, reshard_after_forward=options["reshard_after_forward"]))
