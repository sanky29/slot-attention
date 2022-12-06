# slot-attention
We implemented slot-attention from scratch. We also implemented three different ideas

1. Auxilary loss:
Adding multiplication of pair of masks in the loss

2. Initiation:
Initiated the slots with the function of input frame X

3. Variable number of slots:
Estimated number of slots from the output of encoder using LSTM.

You need to change the args in the ```src/main.py``` with suitable arguments.
The do
```
cd src
python main.py
```
