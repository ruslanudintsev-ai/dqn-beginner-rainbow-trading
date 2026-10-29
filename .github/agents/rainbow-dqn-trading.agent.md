---
name: rainbow-dqn-trading
description: "Workspace agent for Rainbow DQN Forex trading (EURUSD). Use when user asks about RL trading environment, agent design and workflow."
---

# Rainbow DQN Trading Workspace Agent

Цель:
- ускорить разработку и настройку RL-агента для торговли EURUSD с Gym-интерфейсом.

Контекст:
- файл `copilot-instructions.md` в корне, задача: реализовать Gym-среду, добавить индикаторы и risk-managed reward.

Условия применения:
- запросы типа "написать среду торговую", "интегрировать Rainbow DQN", "внедрить перекрытие PER/NoisyNet".

Плюсы:
- быстро дает шаблон `train.py`, `eval.py`, `trading_env.py`, `rainbow_agent.py`.
- при необходимости рекомендую репозиторий "Curt-Park/rainbow-is-all-you-need".
