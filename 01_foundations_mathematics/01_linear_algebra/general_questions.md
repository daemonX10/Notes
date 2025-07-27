# Linear Algebra Interview Questions - General Questions

## Question 1

**How do you perform matrix addition and subtraction?**

**Answer:** Matrix addition and subtraction are fundamental operations performed element-wise on matrices of the same dimensions:

**Matrix Addition:**
- **Rule**: Add corresponding elements from two matrices
- **Requirement**: Matrices must have identical dimensions (same number of rows and columns)
- **Formula**: If A = [aᵢⱼ] and B = [bᵢⱼ], then C = A + B = [aᵢⱼ + bᵢⱼ]

**Example:**
```
A = [1  2]    B = [5  6]    A + B = [1+5  2+6] = [6   8]
    [3  4]        [7  8]            [3+7  4+8]   [10 12]
```

**Matrix Subtraction:**
- **Rule**: Subtract corresponding elements of the second matrix from the first
- **Requirement**: Matrices must have identical dimensions
- **Formula**: If A = [aᵢⱼ] and B = [bᵢⱼ], then C = A - B = [aᵢⱼ - bᵢⱼ]

**Example:**
```
A = [5  8]    B = [1  3]    A - B = [5-1  8-3] = [4  5]
    [6  9]        [2  4]            [6-2  9-4]   [4  5]
```

**Properties:**
1. **Commutative**: A + B = B + A
2. **Associative**: (A + B) + C = A + (B + C)
3. **Identity Element**: A + 0 = A (zero matrix)
4. **Inverse Element**: A + (-A) = 0
5. **Distributive with scalar multiplication**: k(A + B) = kA + kB

**Applications:**
- Combining datasets in data science
- Image processing (adding/subtracting image matrices)
- Economic modeling (combining cost/revenue matrices)
- Physics simulations (superposition of fields)

---

## Question 2

**Define the transpose of a matrix.**

**Answer:** The transpose of a matrix is a fundamental operation that reflects the matrix across its main diagonal:

**Definition:**
The transpose of matrix A, denoted as Aᵀ or A', is formed by interchanging the rows and columns of A. If A is an m×n matrix, then Aᵀ is an n×m matrix.

**Mathematical Notation:**
If A = [aᵢⱼ], then Aᵀ = [aⱼᵢ]

**Example:**
```
A = [1  2  3]     Aᵀ = [1  4]
    [4  5  6]          [2  5]
                       [3  6]
```

**Key Properties:**
1. **(Aᵀ)ᵀ = A** - Transpose of transpose returns original matrix
2. **(A + B)ᵀ = Aᵀ + Bᵀ** - Transpose of sum equals sum of transposes
3. **(AB)ᵀ = BᵀAᵀ** - Transpose of product reverses order
4. **(kA)ᵀ = kAᵀ** - Scalar factor can be factored out
5. **det(Aᵀ) = det(A)** - Determinant unchanged by transpose

**Special Cases:**
- **Symmetric Matrix**: A = Aᵀ (matrix equals its transpose)
- **Skew-Symmetric Matrix**: A = -Aᵀ (matrix equals negative of its transpose)
- **Orthogonal Matrix**: AᵀA = I (transpose equals inverse)

**Applications:**
- **Statistics**: Covariance matrices (XᵀX)
- **Machine Learning**: Normal equations (XᵀX)β = Xᵀy
- **Physics**: Converting between row and column vectors
- **Computer Graphics**: Matrix transformations
- **Data Science**: Feature matrix manipulations

**Geometric Interpretation:**
Transpose represents a reflection across the main diagonal, effectively rotating the matrix coordinate system by swapping axes.

---

## Question 3

**How do you calculate the norm of a vector and what does it represent?**

**Answer:** The norm of a vector is a measure of its length or magnitude in vector space, providing essential geometric and analytical insights:

**Definition:**
A norm is a function that assigns a non-negative real number to each vector, representing its "size" or "length."

**Common Types of Vector Norms:**

**1. L2 Norm (Euclidean Norm):**
- **Formula**: ||v||₂ = √(v₁² + v₂² + ... + vₙ²)
- **Most Common**: Standard geometric length
- **Example**: For v = [3, 4], ||v||₂ = √(3² + 4²) = √25 = 5

**2. L1 Norm (Manhattan Norm):**
- **Formula**: ||v||₁ = |v₁| + |v₂| + ... + |vₙ|
- **Interpretation**: Sum of absolute values
- **Example**: For v = [3, -4], ||v||₁ = |3| + |-4| = 7

**3. L∞ Norm (Maximum Norm):**
- **Formula**: ||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
- **Interpretation**: Largest absolute component
- **Example**: For v = [3, -7, 2], ||v||∞ = 7

**4. General Lp Norm:**
- **Formula**: ||v||ₚ = (|v₁|ᵖ + |v₂|ᵖ + ... + |vₙ|ᵖ)^(1/p)
- **Special Cases**: p=1 (L1), p=2 (L2), p→∞ (L∞)

**Mathematical Properties:**
1. **Non-negativity**: ||v|| ≥ 0, and ||v|| = 0 iff v = 0
2. **Homogeneity**: ||cv|| = |c| · ||v|| for scalar c
3. **Triangle Inequality**: ||u + v|| ≤ ||u|| + ||v||
4. **Subadditivity**: ||u - v|| ≥ ||u|| - ||v||

**What Norms Represent:**

**Geometric Interpretation:**
- **L2 Norm**: Straight-line distance from origin
- **L1 Norm**: City-block distance (Manhattan distance)
- **L∞ Norm**: Chebyshev distance (maximum coordinate difference)

**Physical Interpretations:**
- **Magnitude**: Vector strength or intensity
- **Energy**: In physics, ||v||₂² often represents energy
- **Error**: Distance between actual and predicted values
- **Similarity**: Smaller norm differences indicate similarity

**Applications:**

**1. Machine Learning:**
- **Regularization**: L1 (Lasso), L2 (Ridge) regression
- **Distance Metrics**: k-NN, clustering algorithms
- **Gradient Descent**: Step size and convergence criteria

**2. Signal Processing:**
- **Signal Power**: ||signal||₂²
- **Noise Measurement**: Error norms
- **Filter Design**: Frequency response norms

**3. Optimization:**
- **Convergence Criteria**: ||gradient|| < tolerance
- **Constraint Bounds**: ||x|| ≤ radius
- **Penalty Functions**: Norm-based regularization

**4. Computer Graphics:**
- **Vector Normalization**: Converting to unit vectors
- **Distance Calculations**: Object positioning
- **Collision Detection**: Proximity testing

**Calculation Examples:**

```python
# Vector v = [1, -2, 3, -4]

# L2 norm (Euclidean)
L2 = sqrt(1² + (-2)² + 3² + (-4)²) = sqrt(30) ≈ 5.477

# L1 norm (Manhattan)
L1 = |1| + |-2| + |3| + |-4| = 10

# L∞ norm (Maximum)
L_inf = max(|1|, |-2|, |3|, |-4|) = 4
```

**Unit Vectors:**
A unit vector has norm 1: ||u|| = 1
- **Normalization**: u = v/||v|| creates unit vector in direction of v
- **Purpose**: Represents direction without magnitude
- **Applications**: Coordinate systems, direction vectors

**Relationship to Inner Products:**
For real vectors: ||v||₂ = √⟨v,v⟩ where ⟨v,v⟩ is the inner product

**Practical Considerations:**
- **Numerical Stability**: Use robust algorithms for very large/small values
- **Computational Complexity**: L2 requires square root, L1 and L∞ don't
- **Choice of Norm**: Depends on application requirements and geometric properties needed

---

## Question 4

**Define the concept of orthogonality in linear algebra.**

**Answer:** Orthogonality is a fundamental concept representing perpendicularity and independence in vector spaces, with broad applications across mathematics and engineering:

**Basic Definition:**
Two vectors u and v are orthogonal if their dot product (inner product) equals zero: u · v = 0

**Geometric Interpretation:**
Orthogonal vectors meet at a 90-degree angle, representing perpendicular directions in space.

**Mathematical Formulation:**
For vectors u = [u₁, u₂, ..., uₙ] and v = [v₁, v₂, ..., vₙ]:
- **Orthogonal**: u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ = 0
- **Orthonormal**: Orthogonal AND ||u|| = ||v|| = 1 (unit length)

**Examples:**
```
2D: u = [1, 0], v = [0, 1] → u · v = 1×0 + 0×1 = 0 ✓ orthogonal
3D: u = [1, 1, 0], v = [1, -1, 0] → u · v = 1×1 + 1×(-1) + 0×0 = 0 ✓ orthogonal
```

**Extended Concepts:**

**1. Orthogonal Sets:**
- A set of vectors where every pair is orthogonal
- **Example**: {[1,0,0], [0,1,0], [0,0,1]} - standard basis vectors
- **Property**: Linearly independent (unless containing zero vector)

**2. Orthonormal Sets:**
- Orthogonal set where all vectors have unit length
- **Advantage**: Simplifies calculations and transformations
- **Construction**: Normalize orthogonal vectors: eᵢ = vᵢ/||vᵢ||

**3. Orthogonal Matrices:**
- Square matrix Q where QᵀQ = I
- **Columns**: Form orthonormal set
- **Properties**: Preserves lengths and angles
- **Determinant**: det(Q) = ±1

**4. Orthogonal Subspaces:**
- Two subspaces V and W where every vector in V is orthogonal to every vector in W
- **Notation**: V ⊥ W
- **Example**: Row space and null space of a matrix

**5. Orthogonal Complement:**
- For subspace V, orthogonal complement V⊥ contains all vectors orthogonal to V
- **Property**: V ∩ V⊥ = {0} and V ⊕ V⊥ = Rⁿ

**Key Properties:**

**1. Pythagorean Theorem:**
If u ⊥ v, then ||u + v||² = ||u||² + ||v||²

**2. Orthogonal Projection:**
Projection of vector v onto orthogonal vector u:
proj_u(v) = (v · u / ||u||²) × u

**3. Independence:**
Orthogonal vectors (except zero) are linearly independent

**4. Preservation:**
Orthogonal transformations preserve angles and lengths

**Applications:**

**1. Machine Learning:**
- **PCA**: Principal components are orthogonal
- **Feature Engineering**: Creating independent features
- **Regularization**: Orthogonal constraints in neural networks

**2. Signal Processing:**
- **Fourier Transform**: Orthogonal basis functions
- **Wavelet Analysis**: Orthogonal wavelet families
- **Compression**: Orthogonal transforms for data compression

**3. Computer Graphics:**
- **Coordinate Systems**: Orthogonal axes
- **Rotations**: Orthogonal transformation matrices
- **Projection**: Orthogonal projection onto viewing planes

**4. Statistics:**
- **Regression**: Orthogonal residuals
- **ANOVA**: Orthogonal contrasts
- **Experimental Design**: Orthogonal factors

**5. Numerical Methods:**
- **QR Decomposition**: Orthogonal matrix Q
- **Gram-Schmidt Process**: Creating orthogonal bases
- **Iterative Methods**: Orthogonal search directions

**Construction Methods:**

**1. Gram-Schmidt Process:**
```
Input: Linearly independent vectors {v₁, v₂, ..., vₖ}
Output: Orthogonal vectors {u₁, u₂, ..., uₖ}

u₁ = v₁
u₂ = v₂ - proj_u₁(v₂)
u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)
...
```

**2. QR Decomposition:**
Any matrix A can be factored as A = QR where Q is orthogonal and R is upper triangular

**3. Householder Reflections:**
Orthogonal transformations that reflect vectors across hyperplanes

**Important Theorems:**

**1. Orthogonal Decomposition Theorem:**
Every vector space can be decomposed into orthogonal subspaces

**2. Spectral Theorem:**
Symmetric matrices have orthogonal eigenvectors

**3. Fundamental Theorem of Linear Algebra:**
Four fundamental subspaces have orthogonal relationships

**Practical Benefits:**

**1. Computational Advantages:**
- Simplified dot product calculations
- Stable numerical algorithms
- Efficient projections

**2. Geometric Clarity:**
- Clear spatial relationships
- Intuitive transformations
- Simplified coordinate systems

**3. Statistical Independence:**
- Uncorrelated features
- Independent components
- Reduced multicollinearity

**Common Misconceptions:**
- Orthogonality ≠ linear independence (but orthogonal ⇒ independent)
- Orthogonal matrices preserve MORE than just orthogonality
- Zero vector is orthogonal to all vectors (trivial case)

**Testing Orthogonality:**
1. **Vectors**: Check if dot product equals zero
2. **Matrices**: Verify if AᵀA = I
3. **Subspaces**: Check if all vector pairs have zero dot product
4. **Numerical**: Use tolerance for floating-point comparisons

---

## Question 5

**Define what a symmetric matrix is.**

**Answer:** A symmetric matrix is a square matrix that equals its own transpose, representing perfect symmetry across its main diagonal:

**Mathematical Definition:**
A matrix A is symmetric if and only if A = Aᵀ, which means aᵢⱼ = aⱼᵢ for all i, j.

**Visual Representation:**
```
Symmetric Matrix:        Non-Symmetric Matrix:
[a  b  c]               [1  2  3]
[b  d  e]               [4  5  6]  
[c  e  f]               [7  8  9]
```

**Examples:**
```
2×2 Symmetric:          3×3 Symmetric:
[1   3]                 [2   -1   4]
[3   5]                 [-1   3   0]
                        [4    0   1]
```

**Key Properties:**

**1. Eigenvalue Properties:**
- All eigenvalues are **real numbers** (no complex eigenvalues)
- Eigenvectors corresponding to different eigenvalues are orthogonal
- Can be diagonalized by an orthogonal matrix: A = QΛQᵀ

**2. Spectral Decomposition:**
Every symmetric matrix can be written as A = Σᵢ λᵢvᵢvᵢᵀ where λᵢ are eigenvalues and vᵢ are orthonormal eigenvectors

**3. Quadratic Forms:**
Symmetric matrices naturally arise in quadratic forms: xᵀAx

**4. Definiteness:**
Symmetric matrices can be classified as:
- **Positive Definite**: All eigenvalues > 0
- **Positive Semi-definite**: All eigenvalues ≥ 0
- **Negative Definite**: All eigenvalues < 0
- **Negative Semi-definite**: All eigenvalues ≤ 0
- **Indefinite**: Mixed positive and negative eigenvalues

**Special Types of Symmetric Matrices:**

**1. Identity Matrix:**
```
I = [1  0  0]
    [0  1  0]
    [0  0  1]
```

**2. Diagonal Matrices:**
```
D = [a  0  0]
    [0  b  0]
    [0  0  c]
```

**3. Covariance Matrices:**
Always symmetric and positive semi-definite

**4. Correlation Matrices:**
Symmetric with 1's on diagonal and values between -1 and 1

**Mathematical Operations:**

**1. Addition/Subtraction:**
Sum/difference of symmetric matrices is symmetric

**2. Scalar Multiplication:**
Scalar multiple of symmetric matrix is symmetric

**3. Matrix Multiplication:**
- A symmetric × B symmetric ≠ necessarily symmetric
- But AᵀBA is symmetric if A and B exist

**4. Powers:**
If A is symmetric, then A² is symmetric (and positive semi-definite)

**Applications:**

**1. Statistics and Data Science:**
- **Covariance Matrices**: Measure relationships between variables
- **Correlation Matrices**: Normalized covariance matrices
- **Gram Matrices**: XᵀX in regression and PCA
- **Distance Matrices**: Symmetric distance/similarity measures

**2. Machine Learning:**
- **Kernel Matrices**: Symmetric positive semi-definite
- **Hessian Matrices**: Second derivatives in optimization
- **Feature Covariance**: Understanding feature relationships
- **Regularization**: Ridge regression uses symmetric terms

**3. Physics and Engineering:**
- **Moment of Inertia**: Tensor representations
- **Stress/Strain Tensors**: Material property matrices
- **Network Analysis**: Adjacency matrices for undirected graphs
- **Vibration Analysis**: Mass and stiffness matrices

**4. Optimization:**
- **Quadratic Programming**: Objective functions with symmetric Q
- **Newton's Method**: Hessian matrices
- **Convex Optimization**: Positive definite symmetric matrices

**Computational Advantages:**

**1. Storage Efficiency:**
Only need to store n(n+1)/2 elements instead of n²

**2. Numerical Stability:**
- Symmetric eigenvalue algorithms are more stable
- Cholesky decomposition for positive definite matrices
- Specialized algorithms exploit symmetry

**3. Parallel Computing:**
Symmetry enables efficient parallel algorithms

**Decomposition Methods:**

**1. Eigendecomposition:**
A = QΛQᵀ where Q is orthogonal and Λ is diagonal

**2. Cholesky Decomposition (if positive definite):**
A = LLᵀ where L is lower triangular

**3. LDL Decomposition:**
A = LDLᵀ where L is unit lower triangular and D is diagonal

**Recognition Techniques:**

**1. Visual Inspection:**
Check if matrix equals its transpose

**2. Element-wise Check:**
Verify aᵢⱼ = aⱼᵢ for all i, j

**3. Computational Verification:**
```python
def is_symmetric(A, tolerance=1e-10):
    return np.allclose(A, A.T, atol=tolerance)
```

**Common Sources of Symmetric Matrices:**

**1. Gram Matrices:**
Given matrix X, then XᵀX is always symmetric

**2. Quadratic Forms:**
Matrices representing quadratic expressions

**3. Physical Systems:**
Many physical laws naturally produce symmetric relationships

**4. Optimization Problems:**
Second-order conditions often involve symmetric Hessians

**Important Theorems:**

**1. Spectral Theorem:**
Every real symmetric matrix can be diagonalized by an orthogonal matrix

**2. Principal Axis Theorem:**
Symmetric matrices correspond to conic sections aligned with coordinate axes

**3. Sylvester's Criterion:**
Tests for positive definiteness using leading principal minors

**Practical Considerations:**

**1. Numerical Precision:**
Use appropriate tolerances when checking symmetry computationally

**2. Memory Optimization:**
Store only upper or lower triangular part

**3. Algorithm Selection:**
Choose algorithms designed for symmetric matrices

**4. Conditioning:**
Symmetric matrices can still be ill-conditioned despite nice theoretical properties

---

## Question 6

**Define positive definiteness of a matrix.**

**Answer:** Positive definiteness is a crucial property of symmetric matrices that ensures they behave like "positive numbers" in the matrix world, with fundamental implications for optimization, stability, and geometric interpretations:

**Mathematical Definition:**
A real symmetric matrix A is **positive definite** if for every non-zero vector x:
**xᵀAx > 0**

**Related Definitions:**
- **Positive Semi-definite**: xᵀAx ≥ 0 for all x (allows zero)
- **Negative Definite**: xᵀAx < 0 for all non-zero x
- **Negative Semi-definite**: xᵀAx ≤ 0 for all x
- **Indefinite**: xᵀAx can be positive, negative, or zero for different x

**Equivalent Characterizations:**

**1. Eigenvalue Test:**
A is positive definite ⟺ All eigenvalues λᵢ > 0

**2. Principal Minor Test (Sylvester's Criterion):**
A is positive definite ⟺ All leading principal minors > 0
```
For 3×3 matrix: det(A₁₁) > 0, det([A₁₁ A₁₂; A₂₁ A₂₂]) > 0, det(A) > 0
```

**3. Cholesky Decomposition:**
A is positive definite ⟺ A = LLᵀ exists with L lower triangular and positive diagonal

**4. Quadratic Form:**
A is positive definite ⟺ The quadratic form Q(x) = xᵀAx defines an ellipsoid

**Examples:**

**Positive Definite:**
```
A = [2  1]    →    Eigenvalues: λ₁ = 3, λ₂ = 1 (both > 0) ✓
    [1  2]

Check: For x = [1, 1]ᵀ, xᵀAx = [1 1][2 1][1] = [1 1][3] = 6 > 0 ✓
                                    [1 2][1]       [3]
```

**Not Positive Definite:**
```
B = [1  2]    →    Eigenvalues: λ₁ = 3, λ₂ = -1 (one negative) ✗
    [2  1]
```

**Geometric Interpretation:**

**1. Quadratic Forms:**
- Positive definite: Creates "bowl-shaped" surfaces (ellipsoids)
- Positive semi-definite: Flat in some directions
- Indefinite: Saddle-shaped surfaces

**2. Distance Metrics:**
Positive definite matrices define valid distance metrics via:
d(x,y) = √[(x-y)ᵀA(x-y)]

**3. Energy Functions:**
In physics, positive definite matrices ensure energy is always positive

**Applications:**

**1. Optimization:**
- **Convex Functions**: f(x) = xᵀAx + bᵀx + c is convex iff A is positive semi-definite
- **Local Minima**: Second derivative test requires positive definite Hessian
- **Global Minima**: Guaranteed for positive definite quadratic functions
- **Newton's Method**: Uses positive definite Hessian approximations

**2. Machine Learning:**
- **Covariance Matrices**: Always positive semi-definite
- **Kernel Matrices**: Must be positive semi-definite for valid kernels
- **Regularization**: Adding positive definite terms ensures stability
- **Gaussian Distributions**: Precision matrices are positive definite

**3. Statistics:**
- **Multivariate Normal**: Covariance matrix must be positive definite
- **Fisher Information**: Information matrix is positive semi-definite
- **Confidence Regions**: Elliptical regions from positive definite matrices

**4. Numerical Analysis:**
- **System Solving**: Positive definite systems have unique solutions
- **Iterative Methods**: Convergence guaranteed for positive definite systems
- **Stability**: Positive definite matrices ensure numerical stability

**Testing for Positive Definiteness:**

**1. Eigenvalue Method:**
```python
eigenvalues = np.linalg.eigvals(A)
is_pos_def = np.all(eigenvalues > 0)
```

**2. Cholesky Decomposition:**
```python
try:
    np.linalg.cholesky(A)
    is_pos_def = True
except np.linalg.LinAlgError:
    is_pos_def = False
```

**3. Sylvester's Criterion:**
Check all leading principal minors are positive

**4. Quadratic Form Sampling:**
Test xᵀAx > 0 for many random vectors x

**Special Cases and Properties:**

**1. Diagonal Matrices:**
Positive definite ⟺ All diagonal elements > 0

**2. Sum of Positive Definite Matrices:**
A + B is positive definite if A, B are positive definite

**3. Congruent Transformations:**
If A is positive definite and P is invertible, then PᵀAP is positive definite

**4. Schur Complement:**
For block matrix [A B; Bᵀ C], positive definiteness relates to Schur complements

**Practical Considerations:**

**1. Numerical Issues:**
- **Conditioning**: Well-conditioned positive definite matrices are numerically stable
- **Regularization**: Add λI to make nearly positive definite matrices stable
- **Tolerances**: Use appropriate thresholds for eigenvalue tests

**2. Computational Efficiency:**
- **Cholesky**: O(n³/3) vs O(n³) for general LU decomposition
- **Specialized Algorithms**: Many algorithms optimized for positive definite case
- **Memory**: Can store only lower triangular part

**3. Modifications:**
- **Regularization**: A + λI where λ > 0
- **Pivoting**: Modified Cholesky for indefinite matrices
- **Projection**: Project onto positive definite cone

**Common Errors and Misconceptions:**

**1. Symmetry Requirement:**
Positive definiteness only applies to symmetric (or Hermitian) matrices

**2. Element Signs:**
Positive diagonal elements ≠ positive definite (counterexample: [[1, 2], [2, 1]])

**3. Determinant:**
Positive determinant ≠ positive definite (could be indefinite)

**4. Semi-definite vs Definite:**
Positive semi-definite allows zero eigenvalues (singular matrices)

**Applications in Different Fields:**

**1. Economics:**
- **Utility Functions**: Concave utility requires negative definite Hessian
- **Production Functions**: Convexity constraints
- **Portfolio Optimization**: Covariance matrices in mean-variance optimization

**2. Engineering:**
- **Control Systems**: Lyapunov stability analysis
- **Structural Analysis**: Stiffness matrices must be positive definite
- **Signal Processing**: Autocorrelation matrices

**3. Computer Science:**
- **Graphics**: Metric tensors in rendering
- **Robotics**: Positive definite dynamics for stability
- **Machine Learning**: Kernel methods and optimization

The concept of positive definiteness is fundamental because it bridges linear algebra with optimization, geometry, and probability, providing both theoretical foundations and practical computational advantages.

---

## Question 7

**How do you represent asystem of linear equationsusingmatrices?**

**Answer:** _[To be filled]_

---

## Question 8

**Define and differentiate betweenhomogenousandnon-homogenoussystems.**

**Answer:** _[To be filled]_

---

## Question 9

**How do you compute theinverseof amatrixand when is it possible?**

**Answer:** _[To be filled]_

---

## Question 10

**How do you performQR decomposition?**

**Answer:** _[To be filled]_

---

## Question 11

**How can you representlinear transformationusing amatrix?**

**Answer:** _[To be filled]_

---

## Question 12

**How islinear regressionrelated tolinear algebra?**

**Answer:** _[To be filled]_

---

## Question 13

**How doeigenvaluesandeigenvectorsapply toPrincipal Component Analysis (PCA)?**

**Answer:** _[To be filled]_

---

## Question 14

**What would you consider when choosing a library forlinear algebra operations?**

**Answer:** _[To be filled]_

---

## Question 15

**How do you ensurenumerical stabilitywhen performingmatrix computations?**

**Answer:** _[To be filled]_

---

## Question 16

**How dograph theoryandlinear algebraintersect inmachine learning?**

**Answer:** _[To be filled]_

---

## Question 17

**Given adataset, determine ifPCAwould be beneficial and justify your approach.**

**Answer:** _[To be filled]_

---

## Question 18

**Design alinear algebra solutionfor acollaborative filteringproblem in amovie recommendation system.**

**Answer:** _[To be filled]_

---

