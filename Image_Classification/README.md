# Lecture 2: Image Classification

### Summary

Lecture talks about:

- Limitations in computer vision
- A few notable datasets.
- Intro. to classifiers
  - Nearest Neighbor

### Notes

- Nearest Neighbor
  - Memorizes _all data_
  - Define _Distance Metric_
    - L1 or "Manhattan Distance": a sum of the differences between pixels from both images, as in (a11 - b11) + (a12 - b12) + ... (aij - bij)

### Implementations

- [x] Nearest Neighbor

### Conclusion:

**Nearest Neighbor** is a relatively simple algorithm. I haven't fed real images to it yet, because the main objective of this study was to actually use as little as possible built-in functions to implement the algorithm - _that is why I created a method even to compute the absolute difference between two matrices, to avoid using numpy.abs()_.

I liked implementing this algorithm because I got to refresh a few OOP principles, as well as this marked the day I first wrote a unit testing in Python - I had a few experiences running unit tests in Dart/Flutter back in 2020, but I was mainly following tutorials. This time I got to start thinking the tests by myself and figuring out ways to make them work and reflect the actual functionality of the program. In fact, I even added some exceptions to make the Nearest Neighbor class methods more fault proof.

Nearest Neighbor has a complexity of O(1) at training time and O(n) at test time, which is "the opposite" of what we want. It is preferrable that the computing time burden is shifted towards the training rather than testing.

In this lecture, Justin shows that kNN algorithm is not a good choice to classify over raw pixels. One reason in particular is the **Curse of Dimensionality**, where the amount of training samples needed by kNN to perform well grows exponentially with training data dimension.

Finally, Justin leaves a cliff hanger by sharing that kNNs does actually provide value by computing over feature vectors, rather than raw pixels - as when those feature vectors were provided by a CNN.

### References:

- Justin Johnson's [Lecture 2: Image Classification](https://www.youtube.com/watch?v=0nqvO3AM2Vw&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=2)
