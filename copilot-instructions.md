---
description: "Project-level instructions for Rainbow DQN trading in this workspace."
---

# Copilot Instructions for dqn-beginer

- Ты работаете над проектом Rainbow DQN для трейдинга, основная цель — создать устойчивую Gym-среду для EURUSD и адаптировать Curt-Park/rainbow-is-all-you-need (8 шагов) под торговлю.
- В контексте: PyTorch, Gymnasium, pandas, pandas-ta, files: `trading_env.py`, `rainbow_agent.py`, `train.py`, `eval.py`.
- Предпочтение: простой читаемый код, тестируемость, разделение train/eval, метрики sharpe/cagr/max_drawdown.
- Учитывать спред, комиссию, slippage, стахастику в трейдинговых вознаграждениях.
