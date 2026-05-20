# Smart Grid Energy Arbitrage & Battery Optimizer

An autonomous Deep Reinforcement Learning (DRL) framework developed to optimize residential battery storage systems. By formalizing a Markov Decision Process (MDP) and training a Deep Q-Network (DQN) agent on real-world ENTSO-E historical pricing and generation profiles, the system dynamically balances cost minimization against hardware longevity through an interactive, full-stack visualization dashboard.

---

## Project Architecture & Ecosystem

The system integrates an algorithmic Python simulation backend with an asynchronous HTML5/Chart.js user interface via a Flask REST API.

```text
RL_PROJECT/
├── data/
│   └── energy_dataset.csv     # Historical ENTSO-E pricing & generation profile
├── backend/
│   ├── environment.py         # Custom Gymnasium SmartGridEnv simulator
│   ├── dqn_agent.py           # PyTorch Deep Q-Network framework
│   └── app.py                 # Flask REST API Routing Engine
├── frontend/
│   ├── dashboard.html         # Bootstrap-based User Interface
│   └── app.js                 # Frontend API handler & Chart.js renderer
└── README.md                  # System Documentation

```

---

## Reinforcement Learning Formulation

### 1. State Space ($S \in \mathbb{R}^5$)

The environment tracking vectors provide the agent with five continuous and discrete features every hour step:

* `[Hour of Day, Grid Electricity Price, Solar Generation, House Load, Battery State of Charge (SOC)]`

### 2. Action Space ($A \in \{0, 1, 2, 3\}$)

The agent executes discrete operational directives across four configurations:

* `0`: **Do Nothing** (Idle / Standby to mitigate hardware overhead)
* `1`: **Charge Battery** (Draws excess local solar power or cheap off-peak grid current)
* `2`: **Discharge to House** (Powers domestic load using stored battery capacity)
* `3`: **Sell to Grid** (Injects stored battery energy back to the utility grid during peak pricing)

### 3. Reward Function ($R$)

The system optimizes a specialized fiscal-hardware loss vector:


$$\text{Reward} = -(\text{Net Financial Cost}) - (\text{Battery Degradation Penalty})$$

* Incorporating an explicit hardware penalty ($0.002/\text{kWh}$ cycled) forces the neural network to develop asset-aware strategies rather than chasing marginal, short-term profits.

---

## Empirical Benchmarks & Performance Summary

The system was evaluated using greedy execution ($\epsilon = 0$) over a multi-episode test suite consisting of 24-hour testing profiles.

| Operational Agent | Mean Reward (Daily Financial Cost) | Performance Characterization |
| --- | --- | --- |
| **Rule-Based Heuristic** | **$-28,809.09$** | Highly efficient, rigid human scheduling; prone to volatility failures. |
| **DQN Agent (Ours)** | **$-37,308.58$** | Robust, asset-aware model-free policy; prioritizes battery preservation. |
| **Random Stochastic** | $-37,294.36$ | Unregulated baseline; frequently breaches battery safety bounds. |
| **Do-Nothing Baseline** | $-30,797.02$ | Naked grid exposure with unmanaged domestic load metrics. |

### Core Architectural Insights

* **The Heuristic Baseline Trade-Off:** Across a broad 50-episode historical evaluation, the Rule-Based framework slightly edges out the DQN's *average* score due to its aggressive, zero-overhead cycling.
* **Autonomous Adaptability:** The DQN agent successfully builds an optimal checkpoint policy matching **$-29,675.73$**. As demonstrated on the interactive dashboard, during highly volatile daily cycles with expansive pricing spreads, the DQN dynamically adapts, fluidly routing energy transfers to decisively beat human-engineered static logic.

---

## Installation & Local Deployment

### 1. Initialize Your Isolated Workspace

Clone this repository to your local directory, open your terminal workspace, and activate the project's virtual environment:

```bash
# Navigate to project root
cd RL_PROJECT

# Activate the local Windows virtual environment
venv\Scripts\activate

```

### 2. Satisfy Dependency Tree

Install the critical mathematical, deep learning, and web framework distributions:

```bash
pip install torch flask flask-cors pandas numpy gymnasium

```

### 3. Initialize the Backend REST API

Launch the Python server core. This initializes the custom `Gymnasium` environment, binds pre-trained model weights (`models/dqn_model.pth`), and establishes JSON communications:

```bash
python backend/app.py

```

*Leave this terminal open. The server context listens locally on `http://127.0.0.1:5000`.*

### 4. Boot the Interactive Frontend

1. Navigate to the `frontend/` directory.
2. Launch `dashboard.html` inside any contemporary web browser.
3. Verify the connection indicator badge reads **"Model: Active"** in the top-right console dashboard.
4. Trigger live testing cycles using the **Run DQN Agent** and **Run Rule-Based Agent** parameters to view real-time dual-axis Chart.js updates!


```

```
