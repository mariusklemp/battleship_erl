# 🧠 ERL-Battleship: Hybrid AI Agents for the Game of Battleship

This repository contains the source code for the master's thesis titled:

**"ERL Battleship: Applying Evolutionary Reinforcement Learning to Battleship"**

Developed as part of a research project investigating how hybrid methods—combining Evolutionary Algorithms (EA) and Reinforcement Learning (RL)—can create competitive agents in strategic games with hidden information.

---

## 📚 Thesis Overview

- **Search Agent:** Uses a CNN evolved via a custom NEAT-CNN framework and trained with MCTS-based reinforcement learning.
- **Placing Agent:** Uses a genetic algorithm (GA) to evolve robust ship configurations.
- **Competitive Co-evolution:** Agents evolve by playing against each other, forming an evolutionary arms race.
- **Technologies:** PyTorch, neat-python, DEAP, Pygame.

---

## 🚀 Features

- ✅ Modular Battleship environment
- ✅ NEAT-CNN architecture evolution with weight inheritance
- ✅ Custom GA for valid ship placement
- ✅ MCTS integration with domain-specific enhancements
- ✅ Replay buffer and training pipeline
- ✅ Visualization via Pygame interface

---

## 📁 Project Structure

```
├── ai/                    # AI-related code (MCTS, neural networks)
├── config/               # Configuration files for NEAT and other components
├── deap_system/         # Genetic Algorithm implementation using DEAP
├── game_logic/          # Core game mechanics and rules
├── models/              # Saved model checkpoints
├── neat_system/         # NEAT implementation for neural network evolution
├── placement_population/ # Evolved ship placement strategies
├── rbuf/                # Replay buffer implementation
├── strategies/          # Various agent strategies
├── test/               # Test files
├── main.py             # Main entry point for running games
├── outer_loop.py       # Outer evolutionary loop implementation
├── inner_loop.py       # Inner training loop implementation
├── visualize.py        # Visualization utilities
└── gui.py             # Pygame-based GUI implementation
```

## 🎮 Running the Project

### Prerequisites

1. Python 3.8 or higher
2. Required packages:
   - PyTorch
   - neat-python
   - DEAP
   - Pygame
   - NumPy
   - Matplotlib

### Running Modes

#### Playing Games (`main.py`)
This is where you can play against trained models or watch AI vs AI games:
```bash
# Play against a trained AI
python main.py --mode human_vs_ai --board_size 10 --search_strategy mcts --placing_strategy ga

# Watch two AI agents play against each other
python main.py --mode ai_vs_ai --board_size 10 --search_strategy mcts --placing_strategy ga
```

#### Training Models

There are two levels of training:

1. **Inner Loop (Reinforcement Learning)**
   - Trains individual agents using MCTS and reinforcement learning
   - Run with:

2. **Outer Loop (Evolutionary Training)**
   - Runs the complete ERL system, combining evolution and reinforcement learning
   - Evolves both search and placement strategies

### Configuration

- Board size and game parameters can be modified in `config/game_config.json`
- NEAT parameters are in `config/neat_config.txt`
- Training parameters can be adjusted in `config/training_config.json`

## Modifying the Evolution Process

- Adjust evolution parameters in `config/neat_config.txt`
- Modify selection and mutation operators in `deap_system/`
- Customize fitness functions in `outer_loop.py`
- Customize game paramerters and mcts parameters in `config/mcts_config.json`

## 📊 Results and Analysis

Training metrics and results are stored in:
- `metrics/` - Training statistics and performance metrics
- `plots/` - Generated visualization plots
- `models/` - Saved model checkpoints
