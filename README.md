# DeepDoom
![Current Version](https://img.shields.io/badge/version-0.0.2-red.svg)

Last Updated: **February 16, 2016**

## Overview
Senior Project:

Applying Deep Reinforcement Learning Techniques on the VizDoom environment to learn
navigation behaviors. Our goal is to train Deep Q-Learning Networks on simple navigational
tasks and combining them to solve more complex navigational tasks.

Our Deep Q-Learning implementation is based on the following sources:

- [Qlearning4k](https://github.com/farizrahman4u/qlearning4k)

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

## Scenarios

We designed a set of scenarios where the agent will learn specific behaviors. These
scenarios where created using Doom Builder and VizDoom. The following are descriptions of
the scenarios:

##  

### Scenario 1 : Corridor

#### Description:
A straight corrider. Player gets rewarded for navigating deeper and gets penalized
for hitting walls.
*Available Actions* : [FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT]

#### Goal Function:

- **+50** reward checkpoints
- **+100** level exit
- **-10** hitting walls
- **-1** living reward

###### Files:
- [corridors.wad](src/wads/corridors.wad)
- [corridors.cfg](src/configs/corridors.cfg)

##  

### Scenario 1 : Corridor

##### Description:

##### Goal Function:

###### Files:

## Results:

N/A

## Getting Started

#### Requirements:

Requires Python 3.5.

Requires the following Python Packages:

-[VizDoom](https://github.com/Marqt/ViZDoom)

-[Keras](https://github.com/fchollet/keras)

-[Matplotlib](http://matplotlib.org/)

#### Setup and Installation:

Download or clone repository and install required packages.

The [/src/](src) folder includes all scripts used for this project.

The [/src/wads/](src/wads) folder contains the wad files for the scenarios.

The [/src/configs/](src/configs) folder contains the config files for the scenarios.
