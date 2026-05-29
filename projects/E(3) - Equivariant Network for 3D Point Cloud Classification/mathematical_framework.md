# Mathematical Framework for Project 3 - E(3) - Equivariant Network for 3D Point Cloud Classification

---

## Context

The project is an extension of Project: Graph Neural Network for Molecular Property Prediction (QM9). It implements EGNN (Equivariant Graph Neural Network) from scratch on the ModelNet40 dataset (3D object classification: chairs, tables, planes, etc.) without using PyG's built-in equivariant layers.

**THE MAIN GOAL:** The goal is to train model that achieves competitive accuracy on ModelNet40 showing the accuracy delta between equivariant and non-equivariant architectures under random rotations at test time.

---

## Concepts

* **Point Cloud:** Formally, a set of 3D coordinates $\mathcal{P} = \{x_1, \ldots, x_N\} \subset \mathbb{R}^3$. In practice, it includes features per point, represented as a matrix $P \in \mathbb{R}^{N \times (3 + D)}$ containing 3D positions and $D$-dimensional attributes (e.g., ModelNet40 stores $(x, y, z)$ coordinates alongside surface normals $(n_x, n_y, n_z)$).

* **Message Passing:** A process where graph nodes update their states by aggregating features from their neighbors. *Example: Node A updates its value by averaging the values of connected nodes B and C.*

* **Equivariance:** A property where transforming the input transforms the output in a predictable, matching way ($f(g(x)) = g(f(x))$). *Example: Rotating a 3D chair model rotates its predicted 3D bounding box by the exact same amount.*

* **Rotation Matrix:** A square matrix used to perform a rotation in Euclidean space. *Example: Multiplying a 2D vector by a 2D rotation matrix spins the vector around the origin by a specific angle.*

* **$SO(3)$ (Special Orthogonal Group):** The mathematical group representing all possible continuous rotations in 3D space.

* **$O(3)$ (Orthogonal Group):** The group of all linear transformations in 3D space that preserve distances, including both rotations and reflections.

* **$E(3)$ (Euclidean Group):** The group of all transformations in 3D space that preserve distances, combining rotations, reflections, and translations.

* **Group Action:** A formal way of using a group element to transform elements of a specific space. *Example: Applying a rotation matrix (group element) to a 3D coordinate vector (space element).*

* **$G$-Equivariant:** A function where transforming the input by a group element results in a predictably transformed output.

* **$G$-Invariant:** A function where transforming the input by a group element leaves the output completely unchanged. *Example: Rotating a house does not change its total square footage.*

* **Edge Vector ($r_{ij}$):** The directional vector pointing from node $i$ to node $j$ in 3D space. *Example: If atom $i$ is at $(0,0,0)$ and atom $j$ is at $(0,0,2)$, the edge vector is $(0,0,2)$.*

* **Edge Length ($d_{ij}$):** The straight-line Euclidean distance between node $i$ and node $j$. *Example: For the edge vector $(0,0,2)$, the edge length is $2$.*

* **EGNN (Equivariant Graph Neural Network):** A type of graph neural network designed to process 3D spatial data while guaranteeing that the outputs rotate and translate predictably with the inputs.

* **Coordinate Update:** The layer or mechanism within a spatial GNN that shifts the physical 3D positions ($x_i$) of nodes based on message passing.

* **Linearity of Matrix Multiplication:** The algebraic property where a matrix distributes over addition ($R(a + b) = Ra + Rb$) and factors out of scalar multiplication ($R(ca) = c(Ra)$).

* **Global Mean Pooling:** An operation that takes the average value of features across all nodes in a graph. *Example: For a point cloud with four points, mean pooling coordinates calculates the exact middle point of those four positions.*

* **Permutation Invariance:** A property where changing the ordering or indexing of the inputs does not change the final output. *Example: Shuffling the rows of a point cloud matrix yields the exact same average coordinate.*

* **Center of Mass (Geometric Centroid):** The average position of all points in a shape or point cloud, representing its geometric center.

* **Smooth Manifold ($M$):** A mathematical space that looks globally curved but behaves locally exactly like flat Euclidean space. *Example: The Earth is a 2D spherical manifold; it looks curved from space, but looks like a flat 2D plane to an observer on the ground.*

* **Tangent Space ($T_pM$):** The vector space containing all possible velocity vectors of curves passing through a specific point $p$ on a manifold.

* **Local PCA (Principal Component Analysis):** A data analysis technique applied to a small localized cluster of data points to find the directions of maximum variation. *Example: Fitting a flat cardboard disc to a cluster of points on a basketball surface to capture its local slope.*

* **Surface Normal:** A vector perpendicular to a surface at a specific point, pointing directly away from it.

* **PointNet:** A pioneer deep learning architecture that processes point clouds directly by mapping each point individually using a shared Multi-Layer Perceptron (MLP) and aggregating features globally.

* **Shared MLP:** A network structure where the exact same weights and biases are applied independently to the coordinates of every single point in the cloud.

* **Global Max-Pooling:** A pooling operation that takes a set of point features and extracts the maximum value across each feature channel, collapsing the point cloud into a single global vector.

---

## Underlying Mathematics

---

### Stage 1 - Environment & Dataset Inspection

We're building the foundation: loading ModelNet40 and understanding what we're actually working with before writing a single line of model code. This matters because the entire architectural motivation for EGNN flows from one fact — our data lives in 3D Euclidean space and carries geometric symmetry that naive GNNs throw away.

<br>

### Questions [1]

A point cloud is often described as "a set of 3D points." But sets have no structure beyond membership — and GNNs operate on graphs.

1. **Mathematically**, what is a point cloud as a formal object? Is it just $\mathcal{P} = \{x_1, \ldots, x_N\} \subset \mathbb{R}^3$, or does it carry more structure than that in practice (think: what ModelNet40 actually stores per point)?

2. **Structurally**, what is the minimal additional structure you must impose on $\mathcal{P}$ to run message passing on it — and what choice of that structure will turn out to matter enormously for equivariance?

<br>

### Answers[1]

**1. Mathematical Object vs. Practical Structure**

Mathematically, a raw point cloud is just an unordered set of coordinates. In practice (like ModelNet40), it carries geometric attributes:
$$\mathcal{P} = \{(x_i, f_i) \mid x_i \in \mathbb{R}^3, f_i \in \mathbb{R}^D\}$$
where $x_i$ is the position and $f_i$ is a feature vector (e.g., normals, RGB intensity).

<br>

**2. Minimal Structure for Message Passing & Equivariance**

* **Minimal Structure:** An adjacency relation (a graph topology) defining neighbors, typically constructed via a $k$-nearest neighbors ($k$-NN) graph or an $\epsilon$-radius ball graph.

* **The Equivariance Choice:** The method used to compute relative positions ($x_i - x_j$) or distances ($\|x_i - x_j\|$) within that neighborhood determines equivariance. Using absolute coordinates breaks translation invariance, whereas using relative coordinates and invariant features (like Euclidean distances) allows the network to respect $E(3)$ or $SE(3)$ symmetries.

<br>

The deeper point is why the choice of graph construction is itself a design decision with equivariance consequences. Specifically: the k-NN graph built from Euclidean distances is itself rotation-equivariant — rotating the point cloud permutes the graph edges but doesn't change which points are neighbors. That's a non-trivial property we get for free.

---

In Stage 1, the initial setup and data inspection phase, we have verified and established the following core components for our 3D deep learning pipeline:

* **Environment & Hardware Integration:** The deep learning environment is correctly configured with PyTorch and PyTorch Geometric. It has full access to an enterprise-grade NVIDIA Tesla P100 GPU (16GB), ensuring that our model training will be hardware-accelerated.
* **Dataset Structure:** The ModelNet40 dataset was successfully downloaded and processed from scratch. It contains a total of **12,311 3D CAD models**, cleanly split into:
* **9,843 training samples** to train our neural network.
* **2,468 test samples** to evaluate final model performance.
* **40 distinct object classes** (such as airplanes, bathtubs, and beds).

<br>

* **Data Representation:** Raw 3D meshes are converted into uniform, normalized point clouds. Each individual object in the dataset is structured as a PyTorch Geometric `Data` object containing two critical arrays:
* `pos`: A matrix of shape `[1024, 3]` tracking the precise $(X, Y, Z)$ coordinates of 1,024 points sampled from the object's surface, automatically centered and scaled.
* `normal`: A matrix of shape `[1024, 3]` tracking the surface normal vectors, providing the network with crucial local geometric context about surface orientation.



With the environment verified, data pipeline structured, and spatial dimensions audited via 3D visualization, the data is officially ready to be fed into a geometric deep learning model.

---

### Stage 2 - The E(3) Group: Group: Rotations, Reflections, Translations

Here, we define the three generators of E(3) explicitly as transformation matrices and functions. Implement `rotate(points, R)`, `reflect(points, axis)`, and `translate(points, t)` as explicit, testable Python functions. Then, we apply each to a single point cloud and visualize the result before and after.

<br>

### Questions [2]

1. Write the 3×3 rotation matrix $R_z(\theta)$ for a rotation by angle $\theta$ around the z-axis.

2. State the two conditions a matrix $R \in \mathbb{R}^{3\times3}$ must satisfy to be a valid element of $SO(3)$.

3. In one sentence each: how does $O(3)$ extend $SO(3)$, and how does $E(3)$ extend $O(3)$?

<br>

### Answers [2]

**1. Rotation Matrix around the Z-axis**

The $3\times3$ rotation matrix $R_z(\theta)$ is:

$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\\\ \sin\theta & \cos\theta & 0 \\\\ 0 & 0 & 1 \end{bmatrix}$$

<br>

**2. $SO(3)$ Conditions**

A matrix $R \in \mathbb{R}^{3\times3}$ must satisfy:

1. **Orthogonality:** $R^T R = I$ (the columns are orthonormal and preserve lengths).
2. **Orientation-Preserving:** $\det(R) = +1$ (prevents reflections).

<br>

**3. Group Extensions**

* **$O(3)$ extension of $SO(3)$:** $O(3)$ adds reflection transformations to $SO(3)$, allowing the matrix determinant to be $-1$ instead of strictly $+1$.
* **$E(3)$ extension of $O(3)$:** $E(3)$ adds translation transformations to $O(3)$, shifting objects in space rather than just rotating or reflecting them around the origin.

One precision note: $O(3)$ doesn't just "add" reflections — it's the full group of distance-preserving linear maps, with $SO(3)$ as the index-2 subgroup of orientation-preserving elements. And $E(3)$ is a semidirect product $\mathbb{R}^3 \rtimes O(3)$, not a direct product, because translations and rotations don't commute.

---

In Stage 2, from the geometric manipulation and verification phase, we have established and proven the fundamental properties of 3D spatial transformations required to analyze model invariance and equivariance:

* **Euclidean Transformation Engine:** We implemented the three foundational operations of the Euclidean group ($E(3)$) using vectorized PyTorch operations:
	* **Rotation:** Modeled via matrix multiplication ($X \cdot R^T$) using an orthogonal matrix $R \in \text{SO}(3)$.
	* **Reflection:** Handled by inverting the sign of a specified coordinate hyperplane axis.
	* **Translation:** Executed via PyTorch tensor broadcasting ($X + t$) to shift global positioning.

* **Rigid Rotation Generation:** We built a robust random rotation generator utilizing **QR matrix decomposition** on a random normal matrix. By explicitly checking and correcting the sign of the determinant ($\det(Q) = 1$), we ensured that the matrix yields a true rigid rotation rather than an accidental mirror reflection.

* **Geometric Invariance Proof:** Through visual plotting and mathematical assertion, we verified that these transformations preserve local structure. Specifically, calculating the pairwise distance matrices ($\text{CDist}$) before and after translation returned an exact match (`True`). This mathematically guarantees that the operations are isometric—shifting, rotating, or flipping the point clouds without introducing geometric warping, stretching, or structural distortion.

---

### Stage 3 - Equivariance vs. Invariance Checkers

Here, we implement a formal numerical equivariance checker: a function `check_equivariance(f, transform, x)` that verifies whether f(transform(x)) ≈ transform(f(x)) up to tolerance. Implement a formal invariance checker: `check_invariance(f, transform, x)` that verifies f(transform(x)) ≈ f(x). These two functions will be reused throughout the project as the ground-truth test for every architectural component.

<br>

### Questions [3]

1. State the formal definition of equivariance using group action notation. If $G$ acts on input space $X$ via $\rho_X$ and on output space $Y$ via $\rho_Y$, what equation must hold for $f: X \to Y$ to be $G$-equivariant?

2. For molecular property prediction (e.g., predicting the energy of a molecule), should the output head be equivariant or invariant under $E(3)$ — and why? What breaks if you get this wrong?

<br>

### Answers [3]

**1. Formal Definition of Equivariance**

For a function $f: X \to Y$ to be $G$-equivariant, the following equation must hold for all $g \in G$ and all $x \in X$:

$$f(\rho_X(g) \cdot x) = \rho_Y(g) \cdot f(x)$$

<br>

**2. Molecular Property Prediction Symmetries**

* **The Correct Choice:** The output head must be **invariant** under $E(3)$.
* **Why:** Molecular energy is a scalar quantity that depends only on the relative spatial configuration of the atoms, not on where the molecule is located or how it is rotated in space.
* **What Breaks:** If you use an equivariant head (or a non-symmetric model), translating or rotating the molecule changes its predicted energy, violating the physical laws of conservation of energy.

---

In Stage 3, from the design and verification of the symmetry-testing framework, we have established rigorous methods to evaluate how models handle geometric transformations:

* **Symmetry Testing Suite:** We designed and implemented two algorithmic checkers to mathematically evaluate any function $f$ under spatial transformations $T$:
	* **Equivariance ($f(T(x)) \approx T(f(x))$):** Verifies that the output transforms in exact alignment with the input.
	* **Invariance ($f(T(x)) \approx f(x)$):** Verifies that the output remains entirely unaffected by spatial modifications.

* **Baseline Mathematical Verification:** We confirmed the precision of our testing suite against known geometric operations using a floating-point tolerance threshold ($atol$):
* The **pairwise distance matrix** calculation ($\text{CDist}$) was proven to be strictly **invariant** under random rotation (`True`), as rigid transformations preserve local lengths.
* The **identity function** was proven to be strictly **equivariant** under rotation (`True`), showing an absolute error of exactly zero.


* **Data Edge-Cases and Validation Rigor:** Testing intentionally challenging counter-examples revealed a critical nuance in geometric deep learning:
	* The global centroid function (`mean_fn`) unexpectedly passed *both* equivariance and invariance tests. It was invariant because our dataset was already perfectly pre-centered at $(0,0,0)$ in Stage 1, highlighting that data normalization choices can introduce false positives in symmetry testing.
	* The distance-from-origin function (`broken_fn`) successfully failed the equivariance test (`False` with a massive error of $2.72$), mathematically confirming that discarding directional information while retaining spatial coordinate shapes destroys geometric equivariance.

Cell 12 Code:

```
# cell 12 — deliberately break equivariance to confirm the checker catches failures
# global mean is invariant, not equivariant — checker must return False
mean_fn = lambda p: p.mean(dim=0, keepdim=True).expand_as(p)

passed, err = check_equivariance(mean_fn, rotate_fn, pts)
print(f"constant-mean equivariant (should be False): {passed} (max err {err:.2e})")

passed, err = check_invariance(mean_fn, rotate_fn, pts)
print(f"constant-mean invariant (should be False): {passed} (max err {err:.2e})")
```

Cell 12 Output:

```
constant-mean equivariant (should be False): True  (max err 9.52e-10)
constant-mean invariant (should be False):   True  (max err 6.05e-08)
```

Cell 12.5 Fix:

Think about what `mean_fn` actually computes:

```
mean_fn = lambda p: p.mean(dim=0, keepdim=True).expand_as(p)
```

This returns a tensor where *every row is the global mean* $\bar{x} = \frac{1}{N}\sum_i x_i$.

Is $R\bar{x} = \overline{Rx}$? Work it out — that's exactly why both tests pass and why this is actually a *correct* equivariant/invariant function, not a broken one.

We need a function that genuinely breaks equivariance. A classic example: take the norm of each point independently.

```
# ‖x_i‖ is invariant per point, so the output doesn't transform with R
broken_fn = lambda p: p.norm(dim=1, keepdim=True).expand_as(p)

passed, err = check_equivariance(broken_fn, rotate_fn, pts)
print(f"norm-expand equivariant (should be False): {passed}  (max err {err:.2e})")
```

**Why this `broken_fn` is *not* equivariant and what does equivariance require the output to do under rotation, and what does this function's output actually do?**

Equivariance requires: if you rotate the input, the output rotates by the same $R$. Concretely, $f(Rp)$ must equal $Rf(p)$.

`broken_fn` outputs $\|x_i\|$ broadcast across all 3 coordinates. Since $\|Rx_i\| = \|x_i\|$ (rotations are isometries), rotating the input changes nothing — $f(Rp) = f(p)$. But $Rf(p)$ rotates those norm values as if they were 3D vectors, producing something different. The function is actually *invariant per point*, which is strictly weaker than equivariant. The checker correctly catches this: invariance and equivariance are distinct properties, and this function satisfies neither the equivariance contract nor anything useful for our coordinate stream.

**THE KEY POINT:** the original `mean_fn` passed both tests because linear maps commute with rotation — $R\bar{x} = \overline{Rx}$ exactly. It was never broken; it was a bad choice of counterexample. A genuine failure requires a function that discards directional information, which `norm-expand` does.

---

### Stage 4 - Graph Construction from Point Cloud (k-NN Graph)

We build a k-nearest-neighbor graph over each point cloud: for each point, connect it to its k=20 nearest neighbors by Euclidean distance. Compute edge vectors (relative positions) $r_{ij} = x_j - x_i \quad \text{and edge lengths} \quad d_{ij} = \lVert r_{ij} \rVert$

We inspect the resulting graph structure: node count, edge count, degree distribution. We're imposing the minimal structure needed for message passing: a k-NN graph over each point cloud, plus edge vectors and edge lengths.

<br>

### Questions [4]

1. Of these two edge features — the edge vector $r_{ij} = x_j - x_i \in \mathbb{R}^3$ and the edge length $d_{ij} = \|r_{ij}\| \in \mathbb{R}$ — which is invariant under rotation and which is equivariant?

2. If we discard $r_{ij}$ and use only $d_{ij}$ as the edge feature, what geometric information is lost — and does that loss break the equivariance of the network?

<br>

### Answers [4]

**1. Invariance vs. Equivariance of Edge Features**

* **$d_{ij}$ (Edge Length) is invariant:** Rotating the points does not change the distance between them ($\|R x_j - R x_i\| = \|R(x_j - x_i)\| = \|x_j - x_i\|$).

* **$r_{ij}$ (Edge Vector) is equivariant:** Rotating the points rotates the resulting vector in an identical manner ($R x_j - R x_i = R(x_j - x_i) = R r_{ij}$).

<br>

**2. Information Loss from Discarding $r_{ij}$**

* **What is lost:** You lose all **angular/directional orientation** information between edges. The network can no longer resolve the relative angles between different neighbor points (e.g., it cannot distinguish between a flat, planar molecule and a puckered, non-planar molecule if all pairwise distances happen to match).

* **Impact on Equivariance:** No, it does not break equivariance. Because $d_{ij}$ is strictly invariant, any network processing it remains trivially equivariant (specifically, invariant) because transforming the input yields a predictably unchanged output. However, it severely limits the network's expressive power to capture 3D geometric shapes.

---

In Stage 4, from the graph topological construction and validation phase, we have established how to transform raw unstructured 3D points into structured, localized networks suitable for Geometric Deep Learning:

* **Graph Topological Construction:** We successfully generated a local spatial neighborhood structure using a **$k$-Nearest Neighbors ($k$-NN)** algorithm ($k=20$). This converted an isolated 1,024-point cloud into a connected graph mapped by an `edge_index` matrix tracking exactly **20,480 directed connections**.

* **Decoupled Geometric Features:** We extracted two fundamental local edge properties required for graph message passing, isolating directional and scaling attributes:
	* **Relative Edge Vectors ($r_{ij} = x_j - x_i$):** A matrix of shape `[20480, 3]` capturing local spatial orientation. It is strictly **rotation-equivariant**.
	* **Absolute Edge Lengths ($d_{ij} = \|r_{ij}\|$):** A matrix of shape `[20480, 1]` capturing local scale. It is strictly **rotation-invariant**.

* **Graph Degree Heterogeneity:** Analyzing the network's degree distribution showed an exact mathematical average of 20.0 connections per node. However, we discovered structural variation ranging from an absolute minimum degree of 5 (isolated points on sparse edges like wingtips) to a maximum degree of 41 (dense structural hubs like the fuselage junction), proving that spatial graphs natively adapt to varying local densities.

* **Topological and Vectorial Equivariance Proof:** Using our Stage 3 verification suite, we mathematically validated the structural stability of the graph under random 3D rotations:
	* **Topology Preservation:** The $k$-NN connectivity structure is strictly **invariant** under rigid rotation (`True`), meaning the network map remains identical regardless of how the object is spun.
	* **Vector Equivariance:** The underlying relative edge vectors are strictly **equivariant** (`True`, with a negligible floating-point error of $1.38 \times 10^{-7}$), confirming that local geometric directions transform in perfect mathematical synchronization with global coordinates.

---

### Stage 5 - EGNN Message Passing from Scratch

We now implement the EGNN message passing layer from scratch (no PyG MessagePassing base class, no equivariant layer libraries). The update rule has two coupled streams: a scalar feature stream h and a coordinate stream x. Each layer updates both:

$$
m_{ij} = \phi_e(h_i, h_j, d_{ij}^2)
$$

$$
x_i \leftarrow x_i + \sum_j (x_i - x_j) \cdot \phi_x(m_{ij})
$$

$$
h_i \leftarrow \phi_h\left(h_i, \sum_j m_{ij}\right)
$$

where $\phi_e$, $\phi_x$, and $\phi_h$ are learned MLPs.

<br>

### Questions [5]

Derive the equivariance of the coordinate update algebraically.

The EGNN coordinate update is:

$$x_i \leftarrow x_i + \sum_j (x_i - x_j) \cdot \phi_x(m_{ij})$$

where $m_{ij} = \phi_e(h_i, h_j, d_{ij}^2)$ depends only on scalar features and the invariant distance $d_{ij}^2$.

Now suppose we rotate all input coordinates by $R$, replacing every $x_k$ with $Rx_k$. Carry out the substitution on the right-hand side and show that the result equals $R$ applied to the original update. Write out the algebra step by step — don't skip the distribution over the sum.

<br>

### Answers [5]

Let the original update function be $f(x_i) = x_i + \sum_j (x_i - x_j) \cdot \phi_x(m_{ij})$.

We apply a 3D rotation matrix $R \in SO(3)$ to all coordinates, substituting $x_k \to R x_k$. Because the message scalar $m_{ij}$ relies only on invariant node features and the distance $d_{ij}^2 = \|R x_i - R x_j\|^2 = \|x_i - x_j\|^2$, the scalar output $\phi_x(m_{ij})$ remains completely unchanged.

Substituting the rotated coordinates into the right-hand side of the update equation:

$$\text{RHS} = (R x_i) + \sum_j \left( (R x_i) - (R x_j) \right) \cdot \phi_x(m_{ij})$$

### Step-by-Step Algebra

1. **Factor out $R$ inside the summation:**

$$\text{RHS} = R x_i + \sum_j \left( R(x_i - x_j) \right) \cdot \phi_x(m_{ij})$$


2. **Commute the scalar $\phi_x(m_{ij})$ with the matrix $R$:**

$$\text{RHS} = R x_i + \sum_j R \left( (x_i - x_j) \cdot \phi_x(m_{ij}) \right)$$


3. **Distribute the matrix $R$ out of the summation (by linearity):**

$$\text{RHS} = R x_i + R \left( \sum_j (x_i - x_j) \cdot \phi_x(m_{ij}) \right)$$


4. **Factor out $R$ from the entire expression:**

$$\text{RHS} = R \left( x_i + \sum_j (x_i - x_j) \cdot \phi_x(m_{ij}) \right)$$



The term inside the parentheses is the original unrotated coordinate update. Therefore:

$$\text{RHS} = R \cdot f(x_i)$$

This proves that transforming the input coordinates by $R$ applies the exact same transformation $R$ to the updated coordinates, satisfying the definition of equivariance.

---

In Stage 5, from the architectural design, implementation, and numerical verification of the Equivariant Graph Neural Network (EGNN), we have established how to construct a deep learning system that preserves spatial symmetries:

* **Dual-Stream Layer Architecture:** We implemented a custom `EGNNLayer` based on Satorras et al. (2021) that successfully decouples point cloud updates into two distinct computational pathways:
	* **An Invariant Scalar Stream ($h$):** Updated using a Multi-Layer Perceptron ($\phi_e$) parameterized by squared local distances ($d_{ij}^2$), which are strictly unaffected by rotation or translation.
	* **An Equivariant Coordinate Stream ($x$):** Shifted dynamically along relative displacement vectors ($r_{ij} = x_j - x_i$) scaled by a learned scalar gate ($\phi_x$), forcing coordinates to transform predictably in 3D space.

* **Purely Geometric Feature Learning:** We constructed an end-to-end `EGNN` backbone model that stacked these layers sequentially. By initializing the scalar stream with uninformative constant features ($\mathbf{1}$), we forced the network to learn rich structural shape descriptors exclusively from raw 3D geometry ($x$) during graph message passing.

* **Deterministic Symmetry Validation:** Using our Stage 3 verification framework, we subjected the complete network to random 3D rotations, mathematically proving its spatial correctness:
	* **Coordinate Equivariance:** The structural coordinate stream returned `True` ($max\ err = 1.19 \times 10^{-7}$), confirming that local geometric updates track global rotations perfectly.
	* **Classification Invariance:** The final classification head returned `True` ($max\ err = 8.94 \times 10^{-8}$), mathematically proving that the model outputs identical classification logits regardless of how the 3D shape is oriented in space.

---

### Stage 6 - Invariant and Equivariant Output Heads

We now implement two separate output heads on top of the EGNN backbone:

- An invariant head: global mean-pool the scalar features $h_i$ $\to$ predict class label (for classification).
- An equivariant head: aggregate the coordinate stream $x_i$ $\to$ predict a 3D vector output (e.g., center of mass displacement).

We run both through our `check_invariance` and `check_equivariance` functions respectively and confirm they pass.

<br>

### Questions [6]

Global mean pooling of scalar features $h_i$ is used for classification. Global mean pooling of coordinate features $x_i$ would give a 3D vector output.
Explain why:

1. Mean pooling $h_i$ is invariant to both permutation and rotation
2. Mean pooling $x_i$ is equivariant to rotation but not invariant

Be precise — what does the pooled coordinate vector actually represent geometrically, and why does rotating the point cloud rotate that vector?

<br>

### Answers [6]

**1. Invariance of Mean Pooled Scalar Features ($h_i$)**

* **Permutation Invariance:** The average operation ($\frac{1}{N}\sum_{i=1}^N h_i$) is commutative and associative. Changing the order or indexing of the nodes does not change the sum or the final average.

* **Rotation Invariance:** The scalar features $h_i$ are internal node properties (like chemical element type or local density) that do not depend on spatial coordinates. Rotating the point cloud leaves every $h_i$ completely unchanged, making their mean inherently invariant.

<br>

**2. Equivariance of Mean Pooled Coordinates ($x_i$)**

**Geometric Representation:** The pooled coordinate vector $\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i$ represents the **geometric centroid** (center of mass) of the point cloud.

**Why Rotation Rotates the Vector?**

If you rotate every point in the cloud by a rotation matrix $R$, the new pooled vector becomes:

$$\bar{x}_{\text{rotated}} = \frac{1}{N}\sum_{i=1}^N (R x_i) = R \left( \frac{1}{N}\sum_{i=1}^N x_i \right) = R \bar{x}$$

Because the rotation matrix factors out of the summation due to linearity, rotating the individual points shifts the center of mass by that exact same rotation. It is not invariant because the output vector physically changes position, but it is strictly equivariant because the output changes in lockstep with the input.

---

Stage 6 concluded the architectural blueprint of our Equivariant Graph Neural Network (EGNN) by designing, wiring, and verifying specialized pooling and projection layers. These layers map internal graph features to downstream tasks without disrupting the geometric symmetries maintained by the upstream backbone.

<br>

**1. Invariant Downstream Reasoning (Classification)**

We implemented an `InvariantHead` to compress local, point-level structural embeddings into global shape descriptors:

* **Mathematical Invariance via Permutation Symmetry:** The head utilizes a global mean pooling operator over the scalar feature stream:

$$\text{global\_mean\_pool}(h) = \frac{1}{N} \sum_{i=1}^{N} h_i$$

Because addition is commutative, this averaging step is inherently invariant to node ordering and permutations.

* **Preservation of Rotation Invariance:** Since the incoming hidden scalar attributes $h$ are explicitly decoupled from coordinate orientations within the `EGNNLayer` message-passing loop, their global average is completely static under spatial transformations.

* **Logit Mapping:** Passing this invariant descriptor through a Multi-Layer Perceptron (MLP) with SiLU activations computes classification confidence scores (logits) that depend strictly on structural topology, not physical orientation.

<br>

**2. Equivariant Spatial Tracking (Regression)**

We designed an `EquivariantHead` to handle physical tasks such as coordinate tracking, trajectory estimation, or center-of-mass localization:

* **Vector Equivariance via Linear Pooling:** The head applies global mean pooling directly over the transformed 3D coordinate stream $x$:

$$\vec{x}_{\text{centroid}} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

* **Rigid Transformation Scaling:** Because a rigid 3D rotation matrix $R$ distributes linearly over addition, calculating the centroid of a rotated point cloud yields the exact same vector as rotating the original centroid:

$$\frac{1}{N} \sum_{i=1}^{N} R x_i = R \left( \frac{1}{N} \sum_{i=1}^{N} x_i \right)$$

This enables the network to output spatial coordinate predictions that rotate and translate in perfect harmony with the input space without requiring any learnable parameters.

<br>

**3. Verification Suite Proofs**

Using the symmetry-testing suite established in Stage 3, we subjected the fully integrated multi-task model to random 3D rotations, yielding deterministic mathematical verification of both paths:

| Model Component | Target Symmetry Property | Pass Status | Maximum Numerical Deviation |
| --- | --- | --- | --- |
| **`InvariantHead`** | Invariance ($f(R \cdot p) \approx f(p)$) | **`True`** | $1.19 \times 10^{-7}$ |
| **`EquivariantHead`** | Equivariance ($f(R \cdot p) \approx R \cdot f(p)$) | **`True`** | $3.43 \times 10^{-9}$ |

The minuscule absolute errors confirm that our model can simultaneously extract invariant categorical labels and equivariant spatial vectors from a unified geometric graph representation, bounded only by standard 32-bit floating-point rounding limits ($10^{-7}$).

---

### Stage 7 - Tangent Bundle Interpretation

For one point cloud, we treat each point's k-NN neighborhood as a local coordinate chart. We compute the local PCA frame (3 principal axes) for that neighborhood. Then, we visualize the local frame as arrows overlaid on the point cloud. This is the tangent bundle made concrete: each point has a local "coordinate system" defined by its neighborhood geometry. We're grounding the architecture in differential geometry: making the connection between the abstract notion of a tangent bundle and what we're actually computing on point clouds.

<br>

### Questions [7]

For a smooth manifold $M$, the tangent bundle $TM = \bigsqcup_{p \in M} T_pM$ assigns a tangent space to every point.

1. What lives at each point $p \in M$ in the tangent bundle — what is $T_pM$ geometrically, and what does a basis for $T_pM$ represent?

2. A point cloud has no smooth structure. Explain how the local PCA frame of a k-NN neighborhood around point $x_i$ serves as a discrete approximation of $T_{x_i}M$ — specifically, what do the three principal axes correspond to geometrically, and why is the smallest-variance axis particularly meaningful for surface data?

<br>

### Answers [7]

**1. Geometrical Nature of the Tangent Space**

* **What $T_pM$ is geometrically:** For a 2D surface manifold embedded in 3D space, $T_pM$ is a flat 2D plane that kisses the surface perfectly at point $p$. It contains the velocity vectors of all smooth paths moving through $p$.

* **What a basis represents:** A basis for $T_pM$ represents a coordinate frame for local directions along the surface. For example, on a globe, a standard basis at your position could be a unit vector pointing directly North and a unit vector pointing directly East.

<br>

**2. Discrete Approximation via Local PCA**

When you run PCA on the $k$-nearest neighbors of a point $x_i$, you calculate three orthogonal principal axes (eigenvectors of the local covariance matrix):

* **First two principal axes (Highest Variance):** These two axes span the directions where the neighboring points scatter the most. They approximate the flat 2D tangent plane ($T_{x_i}M$) across the underlying surface.

* **Third principal axis (Smallest Variance):** This axis captures the direction of minimum data scatter. For a clean, non-noisy 2D surface, the points do not deviate significantly out of their plane, meaning the variation along this axis is near zero.

* **Meaning of the smallest-variance axis:** This axis corresponds directly to the **surface normal vector** at $x_i$. It is highly meaningful because estimating this normal allows algorithms to reconstruct surface orientation, calculate lighting, and compute geometric curvatures without an explicit mesh structure.

---

Stage 7 introduced an alternative paradigm for achieving geometric robustness in 3D deep learning: **Local Reference Frames (LRFs)**. Rather than relying on architectural constraints to maintain symmetry (as done with the EGNN dual-stream architecture), this approach uses localized unsupervised geometry to construct invariant surface representations.

<br>

**1. Mathematical Formulation of Local PCA Frames**

We developed a deterministic method (`local_pca_frame`) to map raw spatial coordinates to an anchored, localized 3D coordinate system for any target node $i$:

* **Neighborhood Isolation:** For a target point $x_i$, a local spatial context is isolated using a $k$-NN graph ($k=20$), yielding a neighborhood matrix $X_i \in \mathbb{R}^{k \times 3}$.

* **Centroid Alignment (Translation Invariance):** The neighborhood is centered around its local mean vector to eliminate absolute positional dependencies:

$$X_{i,\text{centered}} = X_i - \frac{1}{k}\sum_{j \in \mathcal{N}(i)} x_j$$


* **Covariance Computation:** A structural $3 \times 3$ covariance matrix $\Sigma_i$ is constructed to measure spatial dispersion:

$$\Sigma_i = \frac{1}{k-1} X_{i,\text{centered}}^T X_{i,\text{centered}}$$


* **Eigendecomposition:** Performing an eigenvalue decomposition yields the local frame:

$$\Sigma_i V_i = V_i \Lambda_i$$

Where $V_i = [v_1, v_2, v_3] \in \mathbb{R}^{3 \times 3}$ represents the orthogonal eigenvectors (principal axes), and $\Lambda_i = \text{diag}(\lambda_1, \lambda_2, \lambda_3)$ contains the eigenvalues sorted in ascending order ($\lambda_1 \le \lambda_2 \le \lambda_3$).

<br>

**2. Geometric Interpretation and Properties**

Inspecting and visualizing the extracted frame at node index 0 revealed distinct geometric attributes:

* **Surface Normal Approximation:** The first principal component ($v_1$), corresponding to the minimum eigenvalue ($\lambda_1 = 0.00114$), defines the axis of least variance. On dense 3D manifolds, this vector acts as an unoriented estimate of the **surface normal**.

* **Flatness Quantified:** The variance ratio along this normal axis was calculated as:

$$\frac{\lambda_1}{\sum \lambda_m} = 0.2455$$

Since this value is strictly below $\frac{1}{3}$ ($0.3333$), it proves the local neighborhood forms a valid, relatively flat localized tangent plane rather than a spherical volume.
* **Orthonormality Verification:** The orthogonality and unit scaling of the coordinate axes were confirmed via:

$$V_i^T V_i = I_3$$

This returns `True`, proving that the frame is a valid metric space that will not distort spatial features when used for coordinate projections.

<br>

**3. Subspace Equivariance Under Rotation**

Eigenvector computation introduces an inherent sign-flip ambiguity: both $v$ and $-v$ satisfy the directional requirements of the covariance matrix. Consequently, standard rigid rotation verification using direct element-wise equality will fail if a sign inversion occurs during numerical optimization.

To robustly validate the rotation-equivariance of the LRF, we implemented a **subspace agreement test** utilizing the determinant of the overlap matrix between the explicitly rotated frame ($V_{\text{rot}}$) and the globally transformed original frame ($R \cdot V_{\text{orig}}$):

$$\left| \det\left( V_{\text{rot}}^T \cdot (R \cdot V_{\text{orig}}) \right) \right| = 1.0$$

| Metric / Check | Value / Status | Geometric Implication |
| :--- | :--- | :--- |
| Subspace Agreement $\lvert\det\rvert$ | **`1.000000`** | The frames span the exact same oriented physical subspace. |
| PCA Frame Equivariance | **`True`** | Constructing an LRF after rotation is equivalent to rotating the original LRF. |

This mathematically proves that the local PCA frame functions as a perfectly **rotation-equivariant** base platform. By projecting surrounding relative coordinates onto these localized axes ($V_i^T \cdot r_{ij}$), a neural network can extract deep 3D structural representations that are entirely **invariant** to global rigid transformations.

---

### Stage 8 - PointNet Baseline (Non-Equivariant)

We implement a PointNet baseline from scratch: shared MLP applied independently to each point, followed by global max-pooling, followed by a classification MLP. This architecture is permutation-invariant but NOT rotation-equivariant. We're building the foil to EGNN. PointNet's architecture is elegant and historically important — but it has a specific, diagnosable failure mode under rotation that the Stage 9 experiment will quantify.

<br>

### Questions [8]

PointNet applies a shared MLP independently to each point, then aggregates with global max-pooling.

Predict what happens to PointNet's classification accuracy when test-time point clouds are randomly rotated by arbitrary $SO(3)$ rotations — and give a geometric argument for *why*. Specifically:

- Global max-pooling gives permutation invariance. Does it give any protection against rotation? Why or why not?
- During training on aligned data, what does the shared MLP effectively learn about each point's coordinates — and why does that learned representation fail when the object arrives rotated at test time?

<br>

### Answers [8]

**What happens when PointNet encounter arbitrary rotations?**

If PointNet is trained on aligned data (e.g., all chairs facing forward) and tested on randomly rotated data, its classification accuracy **drops drastically**.

<br>

**1. Does Global Max-Pooling protect against rotation?**

* **The Answer:** No, it offers absolutely zero protection against rotations.

* **Geometric Argument:** Global max-pooling only ensures **permutation invariance** (the order you list the points does not matter). It calculates the maximum feature value across all points *independently for each feature channel*. It does not look at spatial relationships between points. If you rotate the object, the underlying spatial coordinates ($x, y, z$) feed completely different values into the shared MLP. The max-pooling operation simply takes the maximum values of these corrupted features, resulting in an entirely different, unpredictable global feature vector.

**2. What the shared MLP learns and why it fails?**

* **What it learns during training:** When trained on perfectly aligned objects, the shared MLP effectively learns to partition the absolute 3D bounding box of the dataset. It creates axis-aligned bounding boundaries in coordinate space. For example, it might learn a rule like: *"If a point has a very high $z$ coordinate and centered $x,y$ coordinates, it represents the top headrest of a chair."*

* **Why it fails at test time:** Absolute coordinates are not invariant. If the chair is rotated upside down, the points forming the headrest now have a very low $z$ coordinate. The shared MLP interprets these points based on their new absolute locations in space, mistaking the headrest for the legs of the chair. Because PointNet lacks structural equivariance or relative coordinate tracking, it completely misinterprets the geometry when the absolute coordinate frames no longer align with training.

---

Stage 8 established an empirical baseline comparison between a standard 3D deep learning model (**PointNet**) and our geometric architecture (**EGNN**). By subjecting both networks to deterministic permutation and rigid rotation transformations, we demonstrated the practical differences between structural graph processing and raw coordinate modeling.

<br>

**1. PointNet Structural Properties and Permutation Invariance**

We implemented a standard `PointNet` baseline architecture designed to process unordered point sets directly without constructing a spatial adjacency graph:

* **Per-Point Feature Lifting:** Absolute spatial coordinates $x \in \mathbb{R}^{N \times 3}$ are projected independently into a high-dimensional latent space via a shared Multi-Layer Perceptron (MLP):

$$h_i = \text{SharedMLP}(x_i)$$

* **The Symmetric Pooling Shield:** To collapse the point dimensions into a unified global shape descriptor, the model applies a global maximum pooling operator:

$$h_{\text{graph}} = \max_{i \in \{1, \dots, N\}} (h_i)$$

* **Permutation Verification:** Shuffling the row indices of the input matrix ($x_{\text{perm}} = P \cdot x$, where $P$ is a permutation matrix) yielded an absolute error of exactly **`0.00e+00`**, returning **`True`** for permutation invariance. Because the max operator is associative and commutative, the global feature vector remains mathematically identical regardless of the order in which nodes are fed to the GPU.

<br>

**2. The Failure Mode of Absolute Coordinate Modeling**

While PointNet successfully handles unordered data, it processes raw, absolute spatial coordinates directly. This design choice creates a significant vulnerability to rigid spatial transformations:

* **Rotation Sensitivity:** When an object is rotated ($x_{\text{rot}} = x \cdot R^T$), the absolute values in the coordinate matrix shift completely. Because the linear projection weights within the shared MLP are static, they cannot adapt to global coordinate rotations:

$$\text{SharedMLP}(x_i \cdot R^T) \neq \text{SharedMLP}(x_i)$$

* **Empirical Verification:** Subjecting PointNet to our rotation verification suite returned **`False`**, yielding a significant output logit deviation ($max\ err = 1.20 \times 10^{-2}$). This proves that standard PointNet cannot generalize across orientations; spinning an object changes the model's internal representations and shifts its final prediction confidence scores.

<br>

**3. Empirical Comparison Suite**

To confirm the advantages of geometric deep learning, we evaluated our `EGNN` architecture under identical conditions. Crucially, the EGNN test dynamically re-computed the underlying $k$-NN graph structure on the transformed coordinates to ensure that graph reconstruction did not introduce geometric instability.

The final comparative results are summarized in the table below:

| Model Architecture | Evaluated Transformation | Target Symmetry Property | Pass Status | Maximum Numerical Deviation |
| --- | --- | --- | --- | --- |
| **`PointNet` (Baseline)** | Row Index Shuffling ($P \cdot x$) | Permutation Invariance | **`True`** | $0.00 \times 10^{00}$ (Exact) |
| **`PointNet` (Baseline)** | 3D Rotation Matrix ($x \cdot R^T$) | Rotation Invariance | **`False`** | $1.20 \times 10^{-2}$ (Failed) |
| **`EGNN` (Geometric)** | 3D Rotation Matrix ($x \cdot R^T$) | Rotation Invariance | **`True`** | $5.96 \times 10^{-8}$ (Passed) |

<br>

**4. Core Takeaway**

Stage 8 demonstrated that **permutation invariance does not imply geometric invariance**. While symmetric pooling layers (like `global_max_pool` or `global_mean_pool`) protect a network from variations in data ordering, they offer no protection against spatial rotations.

By grounding its message-passing equations in relative distances ($d_{ij}^2$) rather than absolute spatial points ($x_i$), the **EGNN** maintains an invariant scalar stream across all internal layers. This allows the network to maintain consistent classification outputs under rigid transformations down to floating-point precision limits ($10^{-8}$), eliminating the need for computationally heavy data augmentation.

---

### Stage 9 - PointNet Baseline (Non-Equivariant): Empirical Training and Rigorous Symmetry Benchmarking

We now train both models on ModelNet40 for 3D object classification. We use a standard 80/10/10 split. Evaluate under two test conditions:

- Condition A: no rotation augmentation (aligned test set).
- Condition B: random SO(3) rotation applied to every test sample.

---

Stage 9 transitioned our theoretical models into full-scale empirical training and validation. By training both the Equivariant Graph Neural Network (**EGNN**) and **PointNet** under identical, controlled conditions on the ModelNet40 dataset, we exposed the stark operational trade-offs between absolute coordinate modeling and geometry-grounded message passing.

<br>

**1. High-Efficiency Data Ingestion Pipeline**

To feed our models efficiently during training, we optimized data ingestion by moving geometric operations out of the active GPU execution loop:

* **Precomputed Spatial Topology:** Calculating a $k$-Nearest Neighbors graph scales quadratically with point density ($O(N^2)$). Computing this dynamically on the GPU at every epoch creates severe CPU-GPU synchronization bottlenecks.

* **Transform Serialization:** By constructing an asset pipeline using `torch_geometric.transforms` (`SamplePoints` $\rightarrow$ `NormalizeScale` $\rightarrow$ `KNNGraph`), we precomputed local topology once upfront, serializing the structural edges directly into the dataset tensors.

* **Collated Multi-Object Graph Batching:** PyTorch Geometric aggregates a mini-batch of size 32 by flattening separate entities into a unified, giant disjoint graph structure:

$$\mathbf{X}_{\text{batch}} \in \mathbb{R}^{32768 \times 3}, \quad \mathbf{E}_{\text{batch}} \in \mathbb{R}^{2 \times 655360}, \quad \mathbf{Y}_{\text{batch}} \in \mathbb{R}^{32}$$

This avoids padding matrices with useless zeros, keeping GPU tensor operations highly parallelized and efficient.

<br>

**2. Controlled Optimization Strategy**

To isolate architectural differences as the sole independent variable, we synchronized both training runs across a shared hyperparameter space:

* **Model Dimensions:** Capacity was pinned at $\text{Hidden Dimension} = 64$, $\text{Message Dimension} = 64$, and $\text{Layer Depth} = 4$.
* **Learning Rate Schedule:** Both models utilized an initial learning rate of $\eta = 10^{-3}$ managed by a `CosineAnnealingLR` scheduler over a 64-epoch horizon:

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)$$

This smoothly decayed the optimization step sizes down to narrow local minima by epoch 64.

<br>

**3. Empirical Training Profiles**

The two models exhibited drastically different behaviors during their 64-epoch training runs:

#### PointNet Profiles

* **Computational Footprint:** Concluded training in **41 minutes and 23 seconds** ($\approx 38.80\text{s/epoch}$).

* **Convergence Behavior:** Fitted the training data rapidly, reaching a low training loss of $0.3040$ and achieving a standard validation accuracy of **$83.23\%$**. Because it ignores spatial topology and updates each point independently through a shared MLP, PointNet has a massive advantage in floating-point operation (FLOP) efficiency, allowing it to easily map global spatial patterns.

<br>

#### EGNN Profiles

* **Computational Footprint:** Concluded training in **1 hour, 16 minutes, and 20 seconds** ($\approx 71.57\text{s/epoch}$).

* **Convergence Behavior:** Exhibited slower, more constrained learning, plateauing at a training loss of $1.5589$ and a baseline test accuracy of **$49.47\%$**. This behavior occurs because the network is strictly restricted to learning from relative distance scalars ($d_{ij}^2$), making it more difficult to memorize the raw spatial coordinates of the training objects.

<br>

**4. Rigorous Stress-Testing: The $2 \times 2$ Symmetry Matrix**

The true value of built-in symmetry guarantees became obvious when we evaluated both models against unexpected rigid SO(3) coordinate rotations ($\mathbf{X}_{\text{rot}} = \mathbf{X} \cdot \mathbf{R}^T$).

The final evaluation results across our 4-way testing suite are detailed below:

| Model Architecture | Aligned Test Accuracy | Rotated Test Accuracy | Performance Degradation ($\Delta$) | Empirical Vulnerability Status |
| --- | --- | --- | --- | --- |
| **EGNN** (Geometric) | $0.4935$ | $0.4931$ | **`0.0004`** | **Perfect Rotation Invariance** |
| **PointNet** (Baseline) | $0.8310$ | $0.0778$ | **`0.7532`** | **Catastrophic Structural Collapse** |

<br>

**5. Final Synthesis of Learned Concepts**

* **The Illusion of Aligned Metrics:** PointNet's high accuracy on standard data ($83.10\%$) is a product of dataset alignment bias. In datasets like ModelNet40, objects share a uniform orientation (e.g., chairs always stand upright along the Z-axis). PointNet capitalizes on this bias by memorizing absolute spatial positions.

* **The Catastrophic Breakdown:** The moment raw point coordinates are rotated, PointNet's spatial representations shatter. Its accuracy collapses to **$7.78\%$**—close to random guessing on a 40-class dataset.

* **The Invariance Guarantee:** In contrast, our **EGNN** tracks a flat performance degradation line ($\Delta = 0.0004$), maintaining identical predictions down to floating-point precision limits. Because its internal message equations rely entirely on relative distance vectors, the network's latent features do not shift when an object spins.

Ultimately, **PointNet proved to be $1,859.0\times$ more vulnerable to rotation than the EGNN**. This outcome demonstrates that while geometric deep learning architectures are harder to optimize initially, they provide absolute generalization guarantees that standard, coordinate-bound networks cannot achieve without massive data augmentation.

---
---
