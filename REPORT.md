 (10pt) Report your results through an .md file in your submission; discuss your implementation choices and document the performance of your model (both training and validation performance, both in vitro and in vivo) under the conditions you settled on (e.g., what hyperparameters you chose) and discuss why these are a good set. If you implemented any bonus experiments, detail those in clearly delineated sections as well.


Implementation Choices:
Input/Output Table:
- I took every sentence and iterated through each token in the sentence. For each token, I looked context size to the left and right to create the correct input, then took the token and set that as the output.
- In order to speed up / maximize training, I ommited the following contexts:
    - Empty Context (made up entirely of paddings)
    - Tiny context (If a full context is 2*context_size, I only look at contexts range (context_size, 2*context_size))
    - I also had to spend a considerable effort looking through encoding data code and ensuring I was not looping through (total number of entries * suggested padding length), as this was way too intensive. Rather, I utilized the lens data structure returned from encode data to only loop through the parts of each "encoded sentence" that contained useful data (and ignore the hundreds of padding tokens)

Train/Test Split:
- Decided to stick to the conventional 80-20% split, also remembering the mistake i made last project, ensured it was 80% TRAINING and 20% TESTING, rather than vice versa
- Just made first 80% training, rather than randomizing. I figured that because the books were parsed in order, the first 80% of data will contain words from different books than the last 20% and thus we will still be able to test on different data distributions

Hyperparameters Chosen:
- Context Size: 2
    - Context window looks both to the left and to the right of target word. This is similar to the CBOW implementation and makes sense to me as I feel like bidirectional context is more aligned with firth's "know a word by its company" than just looking in one direction. Often in language, the words before AND after a word are essential to its meaning.
    - For a context size 2, it means we look 2 words forward and backword for a total context size of 4. This default performed well, but i discuss how other context sizes performed in the BONUS section.
- minibatch size:
    - Changed default to 256 because it was just too slow at 32. Considered changing the things that I included in my input/output table but ultimately settled on this implementation
- Vocab Size (fixed at 3000 by project)
- Embedding dim
    - currently set at 128 (was 100 but decided to keep it as a multiple of 2 --> pretty sure this is good for computation speed after discussion in class). Furthermore, was suffering on performance at the 100 level, 128 takes a bit more time but improved performance, will also be experimenting with higher embedding dims as we need to represent a vocab of size 3000
    - UPDATE: ended up switching embedding dim to 256 after researching onling. Typically word embeddings range 100-300 size because capturing the essence of the word is hard to do sub 100 and diminishing returns above 300. While 128 was working well, I experimented with 256 and this worked better. Since our model is pretty simple, it is vital for the embedding layer to be working well and I spent a good amount of time tweaking this
- Loss Criterion
    - CrossEntropy Loss
- Optimizer
    - SGD initially, but eventually switched to Adam after reviewing project 1 discussion and talks about how Adam might be a more efficient way to do this. After switching, I found this to be true
    - LEARNING_RATE = 0.01

Average Epoch Training Time given above hyperparameters on 30 books: ~ 3 min

(10pt continue) Finally, in this report I'd also like you to do a little bit of analysis of the released code and discuss what's going on. In particular, what are the in vitro and in vivo tasks being evaluated? On what metrics are they measured? If there are assumptions or simplifications that can affect the metrics, what are they and what might go "wrong" with them that over- or under-estimate model performance?

The in vitro task being evaluated is how well the model can predict a word given its context. The model takes a input output pair, inserts the input into the model, and compares the output that the model generated with the actual output from the original input/output pair. There are two comparisons being made: accuracy and loss (which I will discuss in further detail below)

This is being measured on both "accuracy" and "loss". Accuracy is defined by if the predicted matches the actual, and the loss used in my implementation is Cross Entropy loss (which works for CBOW) (More explanation above under "Hyperparameters")

The in vivo task considers how the model performs on a task it was NOT specifically trained on. It is an analytical reasoning task that was discussed in class For a given relationship, we take 2 word pairs that present that relationship (A is ot B as C is to D). we then estimate V(D) given V(C) and compare it to V(C)+V(B)-V(A). We see if the closest word to our calculated V(D) is the word D (excluding A, B, and C). The intuition here is that similar relationships should have similar "arrows" from the first word to the second word in the pair.

Looking at the code implementation, it is clear that these assumptions/simplifications are stricter than they need to be and might lead to an under-estimation of the model's performance. 

"Exact" --> refers to if the guessed word is exactly the top word (closest vector)

MRR -> takes the reciprocal of its index+1 in the topn words, this means the first few steps off of being the top word have a HUGE impact on a words MRR score (if it is the second most prevalent word it is 1/2 ...). This means even getting something like top 10 has an incredibly low MRR score

Another reason why these metrics might be a bit tough is because, if one were to look at lecture slides, CBOW was evaluated on vocab sizes on the order of 100's of millions or billions, compared to our 3000! Not really a fair comparison to expect the same /similar performance at such a small vocab size.

BONUS:

5pt) We have hypothesized that the context window size affects syntactic versus semantic performance. Evaluate that hypothesis with your model by varying the context window and looking for relationships to syntax versus semantic analogical task performance.

I varied the context window size a lot during testing, and here are some of the results I was able to obtain:

Context Size:2 

Context Size: 4

Context Size: 6

Summary: It seems like the lower context sizes were able to handle syntactic performance better, with larger context sizes being better equipped at handling semantic performance. This is pretty in line with our results from class!