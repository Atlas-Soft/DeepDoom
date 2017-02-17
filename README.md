# DeepDoom
![Current Version](https://img.shields.io/badge/version-0.0.2-red.svg)

Last Updated: **February 16, 2016**

## Overview
Senior Project:

Applying Deep Reinforcement Learning Techniques on the ViZDoom environment to learn
navigation behaviors. Our goal is to train Deep Q-Learning Networks on simple navigational
tasks and combine them to solve more complex navigational tasks.

Our Deep Q-Learning implementation is based on the following sources:

- [Qlearning4k](https://github.com/farizrahman4u/qlearning4k)

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

## Scenarios

We designed a set of scenarios where the agent will learn specific behaviors. These
scenarios where created using Doom Builder and ViZDoom. The following are descriptions of
the scenarios:

##  

### Scenario 1 : Corridors

#### Description:
The purpose of this scenario is to train the AI on walking. Map is a straight corridor. Player gets rewarded for navigating deeper along the corridor and gets penalized for hitting walls.

*Available Actions* : [MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT]

#### Goal Function:

- **+50** reward checkpoints
- **+100** level exit
- **-10** hitting walls
- **-1** living reward

##### Files:
- [corridors.wad](src/wads/corridors.wad)
- [corridors.cfg](src/configs/corridors.cfg)

##  

### Scenario 2 : Rigid Turning

##### Description:
The purpose of this scenario is to train the AI on turning, specifically via rigid turns. Map is a rigid M-shape with grey walls, ceilings, and floors. Player gets rewarded for navigating from one end of the M-shape to the other end and gets penalized for hitting walls.

*Available Actions* : [MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT]

##### Goal Function:

- **+50** reward checkpoints
- **+100** level exit
- **-10** hitting walls
- **-1** living reward

##### Files:
- [rigid_turning.wad](src/wads/rigid_turning.wad)
- [rigid_turning.cfg](src/configs/rigid_turning.cfg)

##  

### Scenario 3 : Curved Turning

##### Description:
The purpose of this scenario is to train the AI on turning, specifically via curved turns. Map is a sine graph with grey walls, ceilings, and floors. Player gets rewarded for navigating from one end of the sine graph to the other end and gets penalized for hitting walls.

*Available Actions* : [MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT]

##### Goal Function:

- **+50** reward checkpoints
- **+100** level exit
- **-10** hitting walls
- **-1** living reward

##### Files:
- [curved_turning.wad](src/wads/curved_turning.wad)
- [curved_turning.cfg](src/configs/curved_turning.cfg)

## Results:

N/A

## Getting Started

#### Requirements:

Requires Python 3.5.

Requires the following Python Packages:

-[ViZDoom](https://github.com/Marqt/ViZDoom)

-[Keras](https://github.com/fchollet/keras)

-[Matplotlib](http://matplotlib.org/)

#### Setup and Installation:

Download or clone repository and install required packages.

The [/src/](src) folder includes all scripts used for this project.

The [/src/wads/](src/wads) folder contains the wad files for the scenarios.

The [/src/configs/](src/configs) folder contains the config files for the scenarios.
