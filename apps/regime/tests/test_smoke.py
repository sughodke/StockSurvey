"""Smoke tests: every regime submodule imports cleanly."""

from __future__ import annotations


def test_imports():
    import regime
    import regime.broker
    import regime.cli
    import regime.inference
    import regime.live
    import regime.persist
    import regime.reporting
    import regime.research.optimize_adam
    from regime.research import backtest_bt, backtest_ranking, optimize_regime

    # Public API surface
    assert hasattr(regime, 'TrainResult')
    assert hasattr(regime, 'Checkpoint')
    assert hasattr(regime, 'train')
    assert hasattr(regime, 'target_weights')
    assert callable(backtest_bt.WEIGHT_BUILDERS['regime'])
    assert callable(backtest_ranking.RANKERS['rsi'])
    assert callable(optimize_regime.weights_regime_parameterized)


def test_cli_help_loads():
    from regime.cli import _build_parser
    parser = _build_parser()
    train_parser = parser._subparsers._group_actions[0].choices['train']
    train_dests = {a.dest for a in train_parser._actions if a.dest != 'help'}
    assert {'data_dir', 'n_trials', 'metric',
            'train_years', 'val_years', 'step_years'}.issubset(train_dests)
    live_parser = parser._subparsers._group_actions[0].choices['live']
    live_dests = {a.dest for a in live_parser._actions if a.dest != 'help'}
    assert {'params', 'dry_run', 'max_position'}.issubset(live_dests)
