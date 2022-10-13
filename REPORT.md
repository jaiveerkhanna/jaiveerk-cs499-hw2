 (10pt) Report your results through an .md file in your submission; discuss your implementation choices and document the performance of your model (both training and validation performance, both in vitro and in vivo) under the conditions you settled on (e.g., what hyperparameters you chose) and discuss why these are a good set. If you implemented any bonus experiments, detail those in clearly delineated sections as well. Finally, in this report I'd also like you to do a little bit of analysis of the released code and discuss what's going on. In particular, what are the in vitro and in vivo tasks being evaluated? On what metrics are they measured? If there are assumptions or simplifications that can affect the metrics, what are they and what might go "wrong" with them that over- or under-estimate model performance?


Implementation Choices:
Input/Output Table:
- I chose to only use entries that 

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


Finally, in this report I'd also like you to do a little bit of analysis of the released code and discuss what's going on. In particular, what are the in vitro and in vivo tasks being evaluated? On what metrics are they measured? If there are assumptions or simplifications that can affect the metrics, what are they and what might go "wrong" with them that over- or under-estimate model performance?

The in vitro task being evaluated is how well the model can predict a word given its context. The model takes a input output pair, inserts the input into the model, and compares the output that the model generated with the actual output from the original input/output pair. There are two comparisons being made: accuracy and loss (which I will discuss in further detail below)

This is being measured on both "accuracy" and "loss". Accuracy is defined by if the predicted matches the actual, and the loss used in my implementation is Cross Entropy loss (which works for CBOW) (More explanation above under "Hyperparameters")


The in vivo task considers how the model performs on a task it was NOT specifically trained on. 




QUESTIONS:
- How to deal with sets instead of arrays of fixed size


CONSIDER:
- Dropping all padding settings

CLASS:
- What should we expect for downstream eval task?
- One-Hot vs. Multi-Hot
    - CBOW inputs/outputs --> need to slide over entire which maps (context window) to (output)
- Loss function
    - Binary Cross Entropy expects logits over whole vocabulary --> CBOW
    - Prediction of model will be vocabulary by batch, targets expect batch by vocabulary


