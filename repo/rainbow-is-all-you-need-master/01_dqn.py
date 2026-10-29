# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gymnasium[classic-control]",
#     "marimo",
#     "matplotlib",
#     "moviepy",
#     "numpy",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01. DQN

    [V. Mnih и др., «Human-level control through deep reinforcement learning.» Nature, 518
    (7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

    Обучение с подкреплением может быть нестабильным или сходиться неправильно при использовании нелинейных аппроксиматоров функций, таких как нейронные сети, для представления функции ценности действий (также известной как $Q$). Причинами нестабильности являются корреляции в последовательности наблюдений, то что небольшие изменения в $Q$ могут существенно изменить политику и, следовательно, распределение данных, а также корреляции между значениями $Q$ и целевыми значениями $r + \gamma \max_{a'} Q(s', a')$.

    Авторы предлагают две ключевые идеи для уменьшения этих проблем в варианте Q-learning: буфер воспроизведения (Replay buffer) и фиксированная целевая сеть (Fixed Q-target).

    #### Равномерная случайная выборка из Replay Memory

    Агенты сохраняют переходы последовательно в буфер, поэтому соседние переходы ($s, a, r, s'$) обычно коррелированы. Чтобы убрать эту зависимость, агент выбирает переходы равномерно случайно из пула сохранённых примеров $\big( (s, a, r, s') \sim U(D) \big)$. См. метод `sample_batch` класса `ReplayBuffer` для деталей.

    #### Фиксированная целевая сеть

    DQN использует итеративное обновление, которое подгоняет значения $Q$ к целевым значениям, обновляемым только периодически, что уменьшает корреляцию с целью; без этого обучение может расходиться, так как цель постоянно меняется. Обновление Q-learning на итерации $i$ использует следующую функцию потерь:

    $$
    L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \big[ \big( r + \gamma \max_{a'} Q(s',a';\theta_i^-) - Q(s, a; \theta_i) \big)^2 \big]
    $$

    Здесь $\gamma$ — коэффициент дисконтирования, $\theta_i$ — параметры Q-сети на итерации $i$, а $\theta_i^-$ — параметры целевой сети, используемые для вычисления цели. Параметры целевой сети обновляются из локальной сети каждые $C$ шагов и остаются фиксированными между обновлениями. ($C = 200$ для CartPole-v0)

    #### Для большей стабильности: обрезание градиента

    Авторы также рекомендовали ограничивать величину ошибки из обновления $r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s,a;\theta_i)$ в интервале [-1, 1]. Обрезание ошибки эквивалентно использованию функции потерь, близкой к абсолютной, для больших ошибок, что дополнительно улучшает устойчивость алгоритма.
    """)
    return


@app.cell
def _():
    import os
    import warnings

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    return F, gym, nn, np, optim, os, plt, torch, warnings


@app.cell(hide_code=True)
def _(mo):
        mo.md(r"""
        ## Буфер воспроизведения (Replay buffer)

        Обычно буферы воспроизведения реализуют с использованием одной из трёх структур данных:

            - collections.deque
            - list
            - numpy.ndarray

        **deque** удобно использовать при заданной максимальной длине (например, deque(maxlen=buffer_size)). Однако операция индексирования в deque замедляется по мере роста, так как внутри это двусвязный список. **list** — это массив указателей, поэтому при выборке батчей он обычно быстрее, а амортизированная стоимость операции получения элемента — [O(1)].

        Наконец, **numpy.ndarray** ещё быстрее благодаря однородности и компактному хранению элементов фиксированного размера, что даёт преимущества локальности ссылок (locality of reference). Поэтому здесь реализован буфер на основе `numpy.ndarray`.


        Ссылка: [OpenAI spinning-up](https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py#L10)
        """)
        return


@app.cell
def _(np):
    class ReplayBuffer:
        """Простой буфер воспроизведения (replay buffer) на основе numpy."""

        def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
            self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.acts_buf = np.zeros([size], dtype=np.float32)
            self.rews_buf = np.zeros([size], dtype=np.float32)
            self.terminated_buf = np.zeros(size, dtype=np.float32)
            self.max_size, self.batch_size = size, batch_size
            (
                self.ptr,
                self.size,
            ) = 0, 0

        def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            terminated: bool,
        ):
            self.obs_buf[self.ptr] = obs
            self.next_obs_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.terminated_buf[self.ptr] = terminated
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        def sample_batch(self) -> dict[str, np.ndarray]:
            idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                terminated=self.terminated_buf[idxs],
            )

        def __len__(self) -> int:
            return self.size

    return (ReplayBuffer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Сеть (Network)

    Используется простая архитектура сети с тремя полносвязными слоями и нелинейностью ReLU.
    """)
    return


@app.cell
def _(nn, torch):
    class Network(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            """Инициализация."""
            super(Network, self).__init__()

            self.layers = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Реализация прямого прохода (forward)."""
            return self.layers(x)

    return (Network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DQN Agent

    Краткое описание класса `DQNAgent`.

    | Метод              | Описание                                                |
    | ---                | ---                                                     |
    |select_action       | выбрать действие по текущему состоянию.                 |
    |step                | выполнить действие и вернуть ответ среды.               |
    |compute_dqn_loss    | вычислить DQN-потерю.                                   |
    |update_model        | обновить параметры модели методом градиентного спуска.   |
    |target_hard_update  | жёсткое обновление целевой сети из локальной.           |
    |train               | обучить агента в течение num_frames.                    |
    |test                | протестировать агента (1 эпизод).                       |
    |plot                | отобразить прогресс обучения.                            |
    """)
    return


@app.cell
def _(F, Network, ReplayBuffer, gym, mo, np, optim, plt, torch, warnings):
    class DQNAgent:
        """Агент DQN, взаимодействующий со средой.

        Атрибуты:
            env (gym.Env): среда OpenAI Gym
            memory (ReplayBuffer): буфер воспроизведения для хранения переходов
            batch_size (int): размер батча для выборки
            epsilon (float): параметр для epsilon-greedy политики
            epsilon_decay (float): шаг уменьшения epsilon
            max_epsilon (float): максимальное значение epsilon
            min_epsilon (float): минимальное значение epsilon
            target_update (int): периодичность жёсткого обновления целевой сети
            gamma (float): коэффициент дисконтирования
            dqn (Network): обучаемая сеть для Q
            dqn_target (Network): целевая сеть
            optimizer (torch.optim): оптимизатор для обучения dqn
            transition (list): информация о переходе: state, action, reward, next_state, done
        """

        def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            seed: int,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
        ):
            """Initialization.

            Args:
                env (gym.Env): openAI Gym environment
                memory_size (int): length of memory
                batch_size (int): batch size for sampling
                target_update (int): period for target model's hard update
                epsilon_decay (float): step size to decrease epsilon
                lr (float): learning rate
                max_epsilon (float): max value of epsilon
                min_epsilon (float): min value of epsilon
                gamma (float): discount factor
            """
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            self.env = env
            self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
            self.batch_size = batch_size
            self.epsilon = max_epsilon
            self.epsilon_decay = epsilon_decay
            self.seed = seed
            self.max_epsilon = max_epsilon
            self.min_epsilon = min_epsilon
            self.target_update = target_update
            self.gamma = gamma

            # устройство: cpu / gpu
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(self.device)
            
            # сети: dqn, dqn_target
            self.dqn = Network(obs_dim, action_dim).to(self.device)
            self.dqn_target = Network(obs_dim, action_dim).to(self.device)
            self.dqn_target.load_state_dict(self.dqn.state_dict())
            self.dqn_target.eval()

            # оптимизатор
            self.optimizer = optim.Adam(self.dqn.parameters())

            # буфер для перехода, который сохраняем в память
            self.transition = list()

            # режим: обучение / тест
            self.is_test = False

        def select_action(self, state: np.ndarray) -> np.ndarray:
            """Выбрать действие по текущему состоянию."""
            # epsilon-greedy политика (отключена в режиме теста)
            if not self.is_test and self.epsilon > np.random.random():
                selected_action = self.env.action_space.sample()
            else:
                selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
                selected_action = selected_action.detach().cpu().numpy()

            if not self.is_test:
                self.transition = [state, selected_action]

            return selected_action

        def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float64, bool]:
            """Выполнить действие и вернуть ответ среды."""
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if not self.is_test:
                self.transition += [reward, next_state, terminated]
                self.memory.store(*self.transition)

            return next_state, reward, done

        def update_model(self) -> torch.Tensor:
            """Обновить модель методом градиентного спуска."""
            samples = self.memory.sample_batch()

            loss = self._compute_dqn_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def train(self, num_frames: int, plotting_interval: int = 200):
            """Обучение агента."""
            self.is_test = False

            state, _ = self.env.reset(seed=self.seed)
            update_cnt = 0
            epsilons = []
            losses = []
            scores = []
            score = 0

            for frame_idx in range(1, num_frames + 1):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                # если эпизод завершился
                if done:
                    state, _ = self.env.reset(seed=self.seed)
                    scores.append(score)
                    score = 0

                # если готовы к обучению
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

                    # линейно уменьшаем epsilon
                    self.epsilon = max(
                        self.min_epsilon,
                        self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    )
                    epsilons.append(self.epsilon)

                    # если нужно жёсткое обновление целевой сети
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # отрисовка/построение графиков
                if frame_idx % plotting_interval == 0:
                    self._plot(frame_idx, scores, losses, epsilons)

            self.env.close()

        def test(self, video_folder: str) -> None:
            """Протестировать агента (один эпизод)."""
            self.is_test = True

            # запись видео
            naive_env = self.env
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

            state, _ = self.env.reset(seed=self.seed)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            self.env.close()

            # восстановим исходную среду
            self.env = naive_env

            return score

        def _compute_dqn_loss(self, samples: dict[str, np.ndarray]) -> torch.Tensor:
            """Вычислить и вернуть DQN-потерю."""
            device = self.device  # for shortening the following lines
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
            reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
            terminated = torch.FloatTensor(samples["terminated"].reshape(-1, 1)).to(device)

            # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            curr_q_value = self.dqn(state).gather(1, action)
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
            mask = 1 - terminated
            target = (reward + self.gamma * next_q_value * mask).to(self.device)

            # calculate dqn loss
            loss = F.smooth_l1_loss(curr_q_value, target)

            return loss

        def _target_hard_update(self):
            """Жёсткое обновление: target <- local."""
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        def _plot(
            self,
            frame_idx: int,
            scores: list[float],
            losses: list[float],
            epsilons: list[float],
        ):
            """Построить графики прогресса обучения."""
            plt.close("all")
            plt.figure(figsize=(20, 5))
            plt.subplot(131)
            plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-10:])))
            plt.plot(scores)
            plt.subplot(132)
            plt.title("loss")
            plt.plot(losses)
            plt.subplot(133)
            plt.title("epsilons")
            plt.plot(epsilons)
            mo.output.replace(mo.as_html(plt.gcf()))

    return (DQNAgent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment

    You can see the [code](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py) and [configurations](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py#L91) of CartPole-v1 from Farama Gymnasium's repository.
    """)
    return


@app.cell
def _(gym):
    # environment
    env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set random seed
    """)
    return


@app.cell
def _(np, torch):
    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    seed_torch(seed)
    return (seed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize
    """)
    return


@app.cell
def _(DQNAgent, env, seed):
    # parameters
    num_frames = 20000
    memory_size = 10000
    batch_size = 32
    target_update = 100
    epsilon_decay = 1 / 2000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
    return agent, num_frames


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train
    """)
    return


@app.cell
def _(agent, num_frames):
    agent.train(num_frames)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test

    Run the trained agent (1 episode).
    """)
    return


@app.cell
def _(agent, mo):
    video_folder = "videos/dqn"
    score = agent.test(video_folder=video_folder)
    mo.output.replace(mo.md(f"**Test score: {score}**"))
    return (video_folder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render
    """)
    return


@app.cell
def _(mo, os, video_folder):
    import glob

    def show_latest_video(video_folder: str):
        list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    latest_file = show_latest_video(video_folder=video_folder)
    mo.output.replace(mo.video(src=open(latest_file, "rb").read()))
    return


if __name__ == "__main__":
    app.run()
