---
tags:
  - stooq_us_long
  - partial-OOS
---

# Replay decoder options — what `--decoder` actually selects

`ss-replay --decoder` selects what trains on top of the polar Morlet
input bundle. Four options exist and they're not interchangeable —
two are *probes* (fit a head per target, no shared backbone), two
*produce a backbone npz* the [`factor`](../apps/factor.md) app can
consume via `ss_features.load_backbone`. The terminology around
"SSL" is loose in the docs; this page is the precise reference.

!!! note "Doc-correction note (2026-05-09)"
    Earlier findings and app pages described the production backbone
    as "SSL-pretrained" without distinguishing between the
    *supervised-reconstruction `cnn` path* (which is what every
    production `Output/*-cnn-*.npz` actually used) and the
    *strict-SSL `masked-ae` path* (which was implemented but never
    trained as a production npz). The wording read as if `masked-ae`
    was in play when in fact it wasn't. Affected pages have been
    updated in commit
    [`24ba87e`](https://github.com/sughodke/StockSurvey/commit/24ba87e)
    onward to call the production backbone the *supervised-`cnn`
    backbone* (or *indicator-reconstruction backbone*) and to flag
    where "SSL" is being used in the loose sense vs the strict
    masked-AE sense. The table below is the canonical reference for
    which decoder produces what — link to it from any doc that
    needs to be precise.

## The four decoder options

| `--decoder` | Architecture | Training signal | Saves a backbone npz? | Introduced |
|-------------|-------------|-----------------|-----------------------|------------|
| `linear`    | OLS via `np.linalg.lstsq`, one fit per target | Per-target indicator reconstruction (RSI / MACD / vol / CCI / price) — labels are deterministic functions of price | **No** — probe only | [`e4bb049`](https://github.com/sughodke/StockSurvey/commit/e4bb049) (2026-04-26) |
| `mlp`       | Tinygrad MLP, one per target | Same per-target reconstruction | **No** — probe only | [`e4bb049`](https://github.com/sughodke/StockSurvey/commit/e4bb049) (2026-04-26) |
| `cnn`       | 1-D Conv1D shared backbone + per-target linear heads | Same per-target reconstruction; backbone is shared across targets so it learns disentangled indicator-family structure | **Yes** | single-target [`e4bb049`](https://github.com/sughodke/StockSurvey/commit/e4bb049) (2026-04-26); multi-head shared-backbone upgrade [`da5e0af`](https://github.com/sughodke/StockSurvey/commit/da5e0af) (2026-04-27) |
| `masked-ae` | Same conv encoder as `cnn` + masked-CWT autoencoder head | True self-supervised: predict the masked-out region of the CWT input. **No per-target supervision** | **Yes** | [`6cfeb84`](https://github.com/sughodke/StockSurvey/commit/6cfeb84) (2026-04-30) |

## "SSL" — what does and doesn't qualify

In the strict sense (no labels at all, model predicts something
about its own input), only **`masked-ae` is genuinely SSL**. The
`cnn` and `mlp` paths use *self-labelled supervision* — every
target (RSI, MACD, vol, CCI) is computed deterministically from the
input price series, so the model never sees external labels, but
each training row has an explicit target the loss can compare
against. That's regression with self-derived labels, not strict SSL.

In the loose sense the project sometimes uses, all four decoders are
"SSL" because none consume external labels. The
[factor app doc](../apps/factor.md) uses "SSL-pretrained CNN backbone"
in this loose sense — the backbone is `cnn`-trained, not `masked-ae`-
trained.

## Which decoder produces the backbone factor uses?

**All production `Output/*-cnn-*.npz` files were trained with
`--decoder cnn`** (multi-head shared backbone + per-target linear
heads). The factor app's default Modal entrypoint
([`apps/factor/scripts/modal/train_ssl_walkforward.py`](https://github.com/sughodke/StockSurvey/blob/master/apps/factor/scripts/modal/train_ssl_walkforward.py))
points at
`Output/cwtonly-AAPL+294tickers-h631e9d47-rsi+macd+vol+cci-cnn-nogit.npz`
— the `rsi+macd+vol+cci-cnn` token in the filename names the four
reconstruction targets and the `cnn` decoder.

**No `--decoder masked-ae` backbone has been trained as a production
npz.** The encoder shape was designed to be load-compatible (per the
`--decoder` CLI help: *"Encoder shape matches `cnn` so the saved
npz is loadable by `scoring.backbone.load_backbone`"*) but the
end-to-end SSL-pretrain → factor-walkforward loop hasn't been run.
The CLI offers the workflow `--decoder cnn --freeze-backbone <npz>`
to probe a frozen masked-ae encoder against the indicator targets,
which would let us test whether masked-AE captures the same
indicator structure the supervised cnn path captures directly.

## Open question — does masked-AE beat supervised-cnn?

Two competing readings of why the [factor SSL walkforward](factor-ssl-walkforward.md)
mean val IC = **+0.0031** sits below the [+0.012 indicator
baseline](factor-indicator-baseline.md):

1. **Supervision is binding.** The data ceiling for cross-sectional
   20-day return prediction at this universe size is around the
   indicator-baseline floor. Neither `cnn` (current) nor
   `masked-ae` (if we ran it) would lift val IC above that —
   different encoder, same data ceiling.
2. **The encoder is leaving signal on the table.** The `cnn` path
   distils an encoder whose features are constrained to be
   linearly combinable into RSI / MACD / vol / CCI. That constraint
   may *remove* return-predictive geometry the bundle carried.
   `masked-ae` has no such constraint — it just has to encode
   enough to reconstruct masked CWT regions, which is a more
   permissive objective.

Reading (2) makes a falsifiable prediction: train a `masked-ae`
backbone on the same 295-ticker pool, re-run
`train_ssl_walkforward` with that npz, compare aggregate val IC
against the `cnn` row. If masked-AE clears or matches the
indicator baseline, reading (2) is correct and the freeze-and-fresh-
head pattern with `cnn`-pretrain is leaving signal on the table. If
masked-AE lands at ≤+0.0031 too, reading (1) stands and the
encoder choice is no longer the lever.

This experiment is wired but not run. See the
[outstanding question in the factor SSL findings](factor-ssl-walkforward.md#outstanding-question)
for the operational details.

## Notes

- The `linear` and `mlp` probes don't save a backbone but are
  useful for *what's the ceiling on this CWT bundle for indicator
  reconstruction* — running them on a held-out ticker quantifies
  how much information the bundle carries about each target.
- The polar Morlet input bundle (the bundle factor's runtime feeds
  into the loaded backbone) landed in the replay trainer at
  [`954a88a`](https://github.com/sughodke/StockSurvey/commit/954a88a).
  The decoder choice is independent of the input bundle — both
  `cnn` and `masked-ae` see the same polar Morlet `(K=96, F=105)`
  input.
- The replay trainer was extracted from `apps/notebook` to its own
  app at
  [`a81ce1c`](https://github.com/sughodke/StockSurvey/commit/a81ce1c)
  (2026-05-02), at which point the four `--decoder` options moved
  to `apps/replay/src/replay/cli.py`.
