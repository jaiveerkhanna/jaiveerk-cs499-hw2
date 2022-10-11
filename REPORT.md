 (10pt) Report your results through an .md file in your submission; discuss your implementation choices and document the performance of your model (both training and validation performance, both in vitro and in vivo) under the conditions you settled on (e.g., what hyperparameters you chose) and discuss why these are a good set. If you implemented any bonus experiments, detail those in clearly delineated sections as well. Finally, in this report I'd also like you to do a little bit of analysis of the released code and discuss what's going on. In particular, what are the in vitro and in vivo tasks being evaluated? On what metrics are they measured? If there are assumptions or simplifications that can affect the metrics, what are they and what might go "wrong" with them that over- or under-estimate model performance?


Implementation Choices:


Hyperparameters Chosen:
- Context Window:
    - Using a context of 4 words to the left
- minibatch size:
    - arbitrarily 10
- Vocab Size (fixed at 3000 by project)
- Embedding dim
    - currently set at 100
- Loss Criterion
    - CrossEntropy Loss
- Optimizer
    - SGD
    - LEARNING_RATE = 0.005



QUESTIONS:
- How to deal with sets instead of arrays of fixed size