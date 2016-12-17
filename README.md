# Visual-Doom-AI
![Current Version](https://img.shields.io/badge/version-0.0.0-red.svg)

Last Updated: **December 15, 2016**

## Overview
Senior Project: Creating an A.I. capable of playing Doom using visual data.

## Getting Started

#### Requirements:

Requires Python 3.5.

Requires the following Python Packages:

-[VizDoom](https://github.com/Marqt/ViZDoom)

-[Keras](https://github.com/fchollet/keras)

#### Setup and Installation:

Download or clone repository and install required packages.

The [/src/](src) folder includes all scripts used for this project. The following
are short descriptions of each script:

- [DoomSpectatorPlay.py](src/DoomSpectatorPlay.py) - Allows you to play Doom and saves replay in [data/doom_spectator_run/](data/doom_spectator_run) as .lmp file.
```
$Python3 DoomSpectatorPlay.py -m <doom_map> -p <player_name>
```
- [DoomAIPlay.py](src/DoomAIPlay.py) - Allows AI to play Doom and saves replay in [data/doom_ai_run/](data/doom_ai_run) as .lmp file.
```
$Python3 DoomAIPlay.py -m <doom_map>
```
- [DoomReplay.py](src/DoomReplay.py) - Replays .lmp replay files from [data/doom_ai_run/](data/doom_ai_run) or [data/doom_spectator_run/](data/doom_spectator_run).
```
$Python3 DoomReplay.py -f <replay_file>
```
