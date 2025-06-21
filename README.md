# Overview
What punch is this? It's a (jab, uppercut, straight, hook (?), or idle). Admittedly, my boxing skills aren't great, so data is probably garbage.

## Sample usage
```bash
Train data size: 110
Test data size: 59

STARTING...
EPOCH 1 >>>
TEST, ACCURACY: 20.34%, LOSS: 1.5560
EPOCH 2 >>>
TEST, ACCURACY: 61.02%, LOSS: 1.3785
...
EPOCH 19 >>>
TEST, ACCURACY: 96.61%, LOSS: 0.0915
EPOCH 20 >>>
TEST, ACCURACY: 98.45%, LOSS: 0.0586
Save model weights? (y/n) >>> y
Save model weights to? >>> idk2

Done! Weights saved to 'idk2.pkl'
Peak Accuracy: 98.45% @ Epoch 20
Time spent training: 121.34s
```

## Features

- Custom data 
- Predicts punches (?)
- Can't predit hooks

## What I Learned

- Concept 1
- Concept 2
- Concept 3

## Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)  
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-enabled-blue?logo=opencv)
![Pandas](https://img.shields.io/badge/Pandas-enabled-yellow?logo=pandas&logoColor=black)
![Pillow](https://img.shields.io/badge/Pillow-enabled-lightgrey?logo=python)

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

Clone the repo:

```bash
git clone https://github.com/username/repo.git
cd repo
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app

```bash
python3 main.py
```

or

```bash
python main.py
```

## Contributing

See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for how to help.

## Roadmap

- [x] Basic structure
- [ ] Redo data and clean it up
- [ ] Make the model differentiate hooks & uppercuts
- [ ] End to end

## Support

Need help? Ping me on [discord](https://discord.com/users/521872289231273994)

## Acknowledgements

Thanks to all contributors and open-source tools that made this possible ❤️
