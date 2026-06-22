# INCREMENTUM



---

### Project Overview

Model is an experimental artificial intelligence system designed to simulate and learn military strategy on a dynamic battlefield. The system employs two distinct neural network agents that compete against each other: one assuming the role of an attacking force, the other as a defending force. Through adversarial reinforcement learning, both agents develop and refine their strategic capabilities over thousands of simulated battles.

The project explores how artificial intelligence can learn complex spatial reasoning, terrain utilization, and tactical decision-making in a competitive environment.

---

### Core Concept

Two independent neural networks are trained simultaneously through adversarial competition. The Attacker learns to penetrate defenses, execute flanking maneuvers, and capture strategic positions. The Defender learns to establish defensive lines, utilize terrain advantages, and mount counter-offensives. Each episode provides both agents with experience, rewards, and penalties, driving continuous improvement in their strategic capabilities.

The learning process mirrors how military commanders develop expertise through experience, analysis, and adaptation to opposing tactics.

---

### Battlefield Environment

The simulation environment features procedurally generated terrain with realistic topographical elements that influence tactical decisions.

Terrain Generation
The battlefield is created using layered noise functions to produce natural-looking height maps. Hills, valleys, and ridges are distributed across the terrain, creating defensible positions and natural chokepoints. Forests and other cover elements are placed procedurally, providing concealment and defensive bonuses to units positioned within them.

Elevation and Combat
Combat outcomes are significantly influenced by elevation differences. Units positioned on higher ground gain substantial offensive and defensive bonuses, simulating the historical tactical advantage of commanding the high ground. Conversely, units in low-lying areas suffer penalties, encouraging strategic positioning and terrain control.

Cover System
Natural and artificial cover elements provide defensive bonuses, reducing incoming damage. Forests, ruins, and other terrain features create opportunities for ambushes and defensive strongpoints, adding tactical depth to unit positioning.

---

### Unit System

The simulation includes five distinct unit types, each with unique combat statistics and tactical roles.

Infantry
Balanced general-purpose units with moderate attack, defense, and speed. Effective in most combat situations and capable of utilizing cover effectively.

Tanks
Heavily armored units with high attack and defense values. Slower than infantry but capable of breaking through defensive lines.

Artillery
Long-range units with high damage output but low defense. Effective for softening enemy positions from a distance.

Reconnaissance
Fast, lightly armed units ideal for scouting and flanking operations. Capable of covering ground quickly and identifying enemy positions.

Snipers
Specialized units with extreme range and high damage output. Most effective when positioned on high ground.

Each unit type possesses unique strengths and weaknesses, requiring strategic composition and positioning to maximize effectiveness.

---

### Strategic Objectives

The battle is not solely determined by unit elimination. Multiple strategic objectives create layered victory conditions.

Capture Points
Designated positions on the map that generate strategic value when controlled. Maintaining control of these points provides ongoing rewards and contributes to victory.

High Ground Control
Dominating elevated positions provides tactical advantages and generates strategic rewards. Controlling key high ground positions can determine the outcome of engagements.

Flanking Maneuvers
Attacking enemy units from the side or rear provides additional rewards, encouraging tactical positioning and movement rather than direct frontal assaults.

---

### AI Architecture

Both Attacker and Defender agents share a common neural architecture optimized for spatial reasoning and strategic planning.

Map Processing
A convolutional neural network processes the height map, extracting terrain features and identifying defensible positions, chokepoints, and advantageous terrain.

Unit Processing
A transformer-based encoder processes unit positions, types, health status, and other attributes, maintaining awareness of the entire battlefield state.

Strategic Output
The network generates strategic lines representing advance paths or defensive positions, direction vectors indicating primary attack or counter-attack axes, and formation configurations for coordinating multiple units.

Each network produces both value estimates and policy outputs, enabling reinforcement learning through temporal difference methods.

---

### Training Methodology

The system implements a modified Proximal Policy Optimization algorithm with adversarial training dynamics.

Episode Structure
Each training episode begins with both armies positioned on opposite sides of the battlefield. The Attacker and Defender issue strategic orders, units move and engage in combat, and objectives are contested. Episodes conclude when one army is eliminated, all strategic objectives are captured, or a maximum step limit is reached.

Reward Design
The reward structure balances immediate tactical gains with long-term strategic success. Unit eliminations provide immediate rewards, while capturing and holding strategic points generates ongoing value. Terrain utilization, flanking maneuvers, and formation maintenance are also rewarded, encouraging sophisticated strategic behavior.

Learning Dynamics
Both agents learn simultaneously, creating a dynamically evolving competitive environment. Improvements by one agent create pressure for the other to adapt, driving continuous strategic evolution. This adversarial dynamic prevents convergence to suboptimal strategies and encourages emergent complexity.

---

### Visualization and Analysis

A real-time visualization system enables observation of training progress and strategic behavior. The visualizer displays terrain features, unit positions and health, strategic objective status, and AI-generated strategic lines. This visualization provides insight into how the AI models develop and refine their strategic understanding.

---

### Technical Implementation

The system is implemented in Python using PyTorch for neural network operations and Pygame for visualization. The simulation environment is built with NumPy and Shapely for spatial computations.

Neural networks utilize a combination of convolutional and transformer architectures, with approximately 1.1 million parameters per agent model. Training is performed on GPU hardware when available, with CPU fallback capability.

---

### Current Status and Performance

Training episodes demonstrate progressive improvement in strategic capabilities. Initial episodes exhibit basic unit movement and engagement, while later episodes show coordinated maneuvers, terrain utilization, and objective-focused behavior.

Performance metrics including win rates, average episode length, and accumulated rewards indicate successful learning progression across thousands of training episodes.

---

### Future Development Directions

Potential enhancements include expanded map sizes supporting larger-scale operations, additional unit types introducing new tactical possibilities, supply line mechanics adding logistical complexity, multi-agent coordination enabling combined arms operations, and historical scenario replication for comparative analysis.

---

### Project Rationale

Tactical Strategos contributes to understanding how artificial intelligence can learn strategic reasoning in complex, competitive environments. The project explores questions of spatial awareness, terrain evaluation, resource allocation, and adaptive strategy in systems that must balance offensive and defensive considerations.

The adversarial training approach mirrors real-world competitive dynamics, where strategic capabilities are refined through opposition and adaptation. This methodology has potential applications beyond wargaming, including autonomous systems, resource management, and decision support systems.

---

### Conclusion

Tactical Strategos represents an exploration of artificial intelligence applied to strategic decision-making in competitive environments. The system demonstrates how neural networks can learn sophisticated tactical behavior through reinforcement learning, developing capabilities that emerge from adversarial experience rather than explicit programming.

The project continues to evolve, with ongoing research into more complex strategic behaviors and decision-making architectures.

---

### Requirements

Python 3.8 or later with PyTorch, NumPy, Pygame, Shapely, and TQDM libraries.

---

### License

This project is released for research and educational purposes.

---

Tactical Strategos is a research project exploring the intersection of artificial intelligence, strategic reasoning, and competitive learning in simulated battlefield environments.
