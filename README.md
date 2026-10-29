# DQN Rainbow for Trading (EURUSD)

Step-by-step implementation of DQN → Rainbow for reinforcement learning, with the goal of applying it to Forex trading (EURUSD).

Based on the excellent tutorial [Curt-Park/rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need).

## Current Status

- Basic DQN, Double DQN, PER, Dueling, NoisyNet implemented and compared on CartPole-v1
- ForexTradingEnv template ready
- Goal: full Rainbow + walk-forward training on EURUSD

## Structure

```
scripts/
  01_dqn.py
  02_double_q.py
  03_per.py
  04_dueling.py
  05_noisy.py
  compare_dqn.py
envs/
  forex_env.py
data/          # training logs
plots/         # comparison plots
```

## Quick Start

```bash
pip install -r requirements.txt
cd scripts
python 01_dqn.py
python compare_dqn.py
```

## Results (CartPole-v1, 500 episodes, seed=42)

| Model       | Mean   | Std    | Best   | ΔMean vs DQN |
|-------------|--------|--------|--------|--------------|
| 03_per      | 52.70  | 59.23  | 380.0  | +8.83        |
| 04_dueling  | 47.64  | 42.25  | 310.0  | +3.76        |
| 02_double_q | 44.95  | 43.31  | 313.0  | +1.08        |
| 01_dqn      | 43.87  | 41.59  | 402.0  | 0.00         |
| 05_noisy    | 37.19  | 45.60  | 354.0  | -6.68        |

## Next Steps

1. Add 06_categorical_dqn.py
2. Add 07_n_step.py
3. Full Rainbow (08_rainbow.py)
4. Train on ForexTradingEnv with real EURUSD data

## Disclaimer

This is educational. Trading involves risk. Always test thoroughly with walk-forward validation and account for spreads/commissions.
