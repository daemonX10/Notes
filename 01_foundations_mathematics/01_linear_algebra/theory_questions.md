# Linear Algebra Interview Questions - Theory Questions

## Question 1

**What is a vector and how is it used in machine learning?**

**Answer:** A vector is a mathematical object that has both magnitude and direction, represented as an ordered collection of numbers (components). In machine learning:

- **Feature Representation**: Each data point is represented as a feature vector where each component represents a specific attribute or feature
- **Model Parameters**: Weight vectors store the learned parameters of models
- **Embeddings**: Words, images, or other data are converted into vector representations in high-dimensional spaces
- **Computations**: Operations like dot products, distance calculations, and transformations are performed using vector arithmetic
- **Examples**: A house might be represented as [bedrooms=3, bathrooms=2, sqft=1500, price=300000]

---

## Question 2

**Explain the difference between a scalar and a vector.**

**Answer:** 

**Scalar:**
- A single numerical value with magnitude only
- Has no direction (0-dimensional)
- Examples: temperature (25°C), mass (5kg), speed (60 mph)
- Represented by simple numbers: 5, -3.14, 100

**Vector:**
- An ordered collection of numbers with both magnitude and direction
- Multi-dimensional (1D, 2D, 3D, or higher)
- Examples: velocity (60 mph northeast), force (10N at 45°), position coordinates (x=3, y=4)
- Represented as arrays: [3, 4], [-1, 2, 5]

**Key Differences:**
- **Dimensionality**: Scalars are 0D, vectors are 1D or higher
- **Operations**: Scalars use basic arithmetic, vectors use specialized operations (dot product, cross product)
- **Representation**: Scalars are single values, vectors are arrays/matrices

---

## Question 3

**What is a matrix and why is it central to linear algebra?**

**Answer:** A matrix is a 2D array of numbers arranged in rows and columns, represented as:

```
A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]
    [a₃₁  a₃₂  a₃₃]
```

**Why matrices are central to linear algebra:**

1. **Linear Transformations**: Matrices represent linear transformations between vector spaces
2. **System of Equations**: Solve multiple linear equations simultaneously (Ax = b)
3. **Data Organization**: Store and manipulate large datasets efficiently
4. **Composition**: Combine multiple transformations through matrix multiplication
5. **Eigenvalue Problems**: Find characteristic vectors and values
6. **Dimensionality**: Work with high-dimensional spaces

**Applications in ML:**
- **Dataset Representation**: Each row = sample, each column = feature
- **Neural Networks**: Weight matrices connect layers
- **PCA**: Covariance matrices for dimensionality reduction
- **Transformations**: Rotation, scaling, translation operations

---

## Question 4

**Explain the concept of a tensor in the context of machine learning.**

**Answer:** A tensor is a generalization of scalars, vectors, and matrices to arbitrary dimensions:

**Tensor Hierarchy:**
- **0D Tensor**: Scalar (single number)
- **1D Tensor**: Vector (array of numbers)
- **2D Tensor**: Matrix (2D array)
- **3D Tensor**: Cube of numbers (height × width × depth)
- **nD Tensor**: n-dimensional array

**In Machine Learning:**

1. **Data Representation**:
   - **Images**: 3D tensors (height × width × channels)
   - **Video**: 4D tensors (time × height × width × channels)
   - **Batch Processing**: Add batch dimension (batch × features)

2. **Deep Learning**:
   - **Input**: Multi-dimensional data tensors
   - **Weights**: Parameter tensors of various shapes
   - **Activations**: Feature maps as tensors

3. **Operations**:
   - **Tensor Addition**: Element-wise operations
   - **Tensor Multiplication**: Generalized matrix multiplication
   - **Reshaping**: Change dimensions while preserving data

**Example**: A batch of 32 RGB images (224×224) = tensor shape [32, 224, 224, 3]

---

## Question 5

**What are the properties ofmatrix multiplication?**

**Answer:** _[To be filled]_

---

## Question 6

**Explain thedot productof twovectorsand its significance inmachine learning.**

**Answer:** _[To be filled]_

---

## Question 7

**What is thecross productofvectorsand when is it used?**

**Answer:** _[To be filled]_

---

## Question 8

**What is thedeterminantof amatrixand what information does it provide?**

**Answer:** _[To be filled]_

---

## Question 9

**Can you explain what aneigenvectorandeigenvalueare?**

**Answer:** _[To be filled]_

---

## Question 10

**How is thetraceof amatrixdefined and what is its relevance?**

**Answer:** _[To be filled]_

---

## Question 11

**What is adiagonal matrixand how is it used inlinear algebra?**

**Answer:** _[To be filled]_

---

## Question 12

**Explain the properties of anidentity matrix.**

**Answer:** _[To be filled]_

---

## Question 13

**What is aunit vectorand how do you find it?**

**Answer:** _[To be filled]_

---

## Question 14

**Explain the concept of anorthogonal matrix.**

**Answer:** _[To be filled]_

---

## Question 15

**What is therankof amatrixand why is it important?**

**Answer:** _[To be filled]_

---

## Question 16

**What is the method ofGaussian elimination?**

**Answer:** _[To be filled]_

---

## Question 17

**Explain the concept oflinear dependenceandindependence.**

**Answer:** _[To be filled]_

---

## Question 18

**What is the meaning of thesolution spaceof asystem of linear equations?**

**Answer:** _[To be filled]_

---

## Question 19

**Describe the conditions forconsistencyinlinear equations.**

**Answer:** _[To be filled]_

---

## Question 20

**Explain theLU decompositionof amatrix.**

**Answer:** _[To be filled]_

---

## Question 21

**What aresingularorill-conditioned matrices?**

**Answer:** _[To be filled]_

---

## Question 22

**What is theSingular Value Decomposition (SVD)and its applications inmachine learning?**

**Answer:** _[To be filled]_

---

## Question 23

**Explain the concept ofmatrix factorization.**

**Answer:** _[To be filled]_

---

## Question 24

**What is alinear transformationinlinear algebra?**

**Answer:** _[To be filled]_

---

## Question 25

**Describe thekernelandimageof alinear transformation.**

**Answer:** _[To be filled]_

---

## Question 26

**How doeschange of basisaffectmatrix representationoflinear transformations?**

**Answer:** _[To be filled]_

---

## Question 27

**Describe the role oflinear algebrainneural network computations.**

**Answer:** _[To be filled]_

---

## Question 28

**Explain how theSVDis used inrecommendation systems.**

**Answer:** _[To be filled]_

---

## Question 29

**Explain how you would preprocess data to be used inlinear algebracomputations.**

**Answer:** _[To be filled]_

---

## Question 30

**Describe ways to find therankof amatrixeffectively.**

**Answer:** _[To be filled]_

---

## Question 31

**Explain how you would uselinear algebrato clean and preprocess adataset.**

**Answer:** _[To be filled]_

---

## Question 32

**Describe a scenario where linear algebra could be used to improvemodel accuracy.**

**Answer:** _[To be filled]_

---

## Question 33

**What aresparse matricesand how are they efficiently represented and used?**

**Answer:** _[To be filled]_

---

## Question 34

**Explain howtensor operationsare vital in algorithms working with higher-dimensional data.**

**Answer:** _[To be filled]_

---

## Question 35

**What is the role oflinear algebraintime series analysis?**

**Answer:** _[To be filled]_

---

