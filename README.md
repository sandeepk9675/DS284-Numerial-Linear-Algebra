# Numerical Linear Algebra Assignments

This repository contains solutions for three Numerical Linear Algebra assignments, showcasing the application of advanced computational techniques to solve real-world problems.

---

### **Assignment 3: Image Compression Using SVD**
- **Objective**: Solve the problem of transmitting high-resolution astronomical images with reduced data size.
- **Solution**:
  - Used Singular Value Decomposition (SVD) to compress the input image (`input_image.png`).
  - Determined the number of singular values needed for near-lossless compression.
  - Computed the 2-Norm and Frobenius-Norm errors between the original and compressed images.
  - Verified error-related theorems for SVD approximations.

---

### **Assignment 4: QR Decomposition and Numerical Stability**
- **Objective**: Explore the numerical stability of QR decomposition algorithms in finite precision arithmetic.
- **Solution**:
  - Implemented Classical Gram-Schmidt (CGS) and Modified Gram-Schmidt (MGS) methods to compute QR decomposition.
  - Used Householder transformations to further analyze the stability and accuracy of the results.
  - Compared the orthogonality of resulting `Q` matrices from different methods.

---

### **Assignment 5: PageRank Computation Using Power Iteration**
- **Objective**: Calculate the PageRank of nodes in a graph using Power Iteration and Markov transition matrices.
- **Solution**:
  - Constructed the Markov transition matrix for the graph.
  - Implemented the Power Iteration algorithm to compute the dominant eigenvector.
  - Plotted:
    - Residual norms for the eigenvalue problem across iterations.
    - Convergence of successive iterates.
    - Rayleigh quotient convergence.
  - Identified nodes with the highest and lowest PageRanks.

---

## Usage Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/sandeepk9675/DS284-Numerial-Linear-Algebra

