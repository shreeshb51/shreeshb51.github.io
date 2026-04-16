## Mathematical Framework for Project 1 - Wasserstein Distance as Loss Function for Image Morphing

Before any optimal transport can happen, we need two objects to *transport between*. This code is the moment we reach into a dataset of 60,000 handwritten digits and pull out exactly two images — one "3" and one "8" — that will become the **source and target measures** of our Wasserstein geodesic. Every subsequent computation flows from these two arrays.

---

**We take anchor example and use it consistently and cumulatively throughout all subsequent explanations.**

We pretend MNIST contains six toy samples, each a 2×2 pixel image. The six images have labels:

$$\mathbf{y} = (3,\ 7,\ 8,\ 3,\ 1,\ 8)$$

The two images we care about are:

$$A^{(3)} = \begin{pmatrix} 0 & 180 \\ 200 & 255 \end{pmatrix}, \qquad A^{(8)} = \begin{pmatrix} 210 & 255 \\ 180 & 240 \end{pmatrix}$$

Each entry is an integer in $\{0, \ldots, 255\}$, where 0 is black and 255 is white.

---

**`labels = np.array([label for _, label in mnist])`**

The MNIST dataset returns pairs $(\text{image},\ \text{label})$. This line **throws away every image** and collects only the integer class labels into a single vector. The underscore `_` is a Python convention for "I am deliberately ignoring this value."

Mathematically, we are constructing the label sequence:

$$\mathbf{y} = (y_0,\ y_1,\ \ldots,\ y_{N-1}), \qquad y_i \in \{0, 1, \ldots, 9\}, \qquad N = 60000$$

Think of $\mathbf{y}$ as a function $\mathbf{y}: \{0,\ldots,N-1\} \to \{0,\ldots,9\}$ — it maps a dataset index to a digit class.

On the anchor: $\mathbf{y} = (3,\ 7,\ 8,\ 3,\ 1,\ 8)$, so $N = 6$.

---

**`idx_3 = np.where(labels == 3)[0][0]`**

`np.where(labels == 3)` returns every index $i$ where $y_i = 3$. We then take `[0][0]` — the first element of that list — giving us the **earliest occurrence** of a "3" in the dataset.

Mathematically:

$$\mathcal{I}_3 = \{i \in \{0,\ldots,N-1\} : y_i = 3\}, \qquad i_3 = \min(\mathcal{I}_3)$$

The set $\mathcal{I}_3$ might contain thousands of elements — there are roughly 6,000 threes in MNIST. We take the minimum purely for **reproducibility**: given the same dataset, we always get the same image. No statistical significance is claimed for this particular "3."

On the anchor: $\mathcal{I}_3 = \{0, 3\}$, so $i_3 = \min\{0,3\} = 0$.
Similarly: $\mathcal{I}_8 = \{2, 5\}$, so $i_8 = \min\{2,5\} = 2$.

---

**`img_3 = np.array(mnist[idx_3][0], dtype=np.float64)`**

`mnist[idx_3]` retrieves the pair $(\text{image},\ \text{label})$ at position $i_3$. The `[0]` selects just the image (a PIL Image object).

`np.array(..., dtype=np.float64)` converts it into a 2D NumPy array of **64-bit floating-point numbers**.

This is a **type cast**, not a value change:

$$A^{(3)} \in \{0,\ldots,255\}^{28\times28} \xrightarrow{\ \text{cast}\ } A^{(3)} \in \mathbb{R}^{28\times28}$$

The value 180 becomes 180.0 — identical numerically, but now living in $\mathbb{R}$ rather than $\mathbb{Z}$.

**Why does this matter so much?** Consider what is coming:

1. **Normalization** requires dividing by the pixel sum. In integer arithmetic, $1 / 635 = 0$ (truncated). In float64, $1.0 / 635.0 = 0.001574\ldots$ — correct.
2. **Sinkhorn iterations** require computing $\log$ of pixel values. $\log$ is not defined on integers in NumPy without a cast.
3. **Barycentric interpolation** produces values like $0.3 \cdot 180.0 + 0.7 \cdot 210.0 = 201.0$ — fractional arithmetic is essential throughout.

The choice of **float64** (rather than float32) is deliberate: Sinkhorn iterations involve repeated multiplication of numbers close to zero (exponentials of negative costs scaled by small $\varepsilon$), and float32's ~7 decimal digits of precision can cause **numerical underflow** that corrupts the transport plan. float64 gives ~15 digits.

On the anchor, after the cast:

$$A^{(3)} = \begin{pmatrix} 0.0 & 180.0 \\ 200.0 & 255.0 \end{pmatrix}, \qquad A^{(8)} = \begin{pmatrix} 210.0 & 255.0 \\ 180.0 & 240.0 \end{pmatrix}$$

---

```
pixel range before normalization: [0.0, 255.0]
```

This confirms two facts that will drive the process fowrward:

- The **minimum is 0.0** — the background pixels. These are *problematic* for optimal transport, because a pixel with value 0 carries zero mass, and we need to decide how to handle that (you'll see a small $\epsilon$ regularization added during normalization).
- The **maximum is 255.0** — the brightest ink pixel. The *ratio* between these extremes spans three orders of magnitude, which is why naive normalization (divide by 255) would leave most of the image near zero mass.

---

We hold two raw pixel matrices $A^{(3)}, A^{(8)} \in [0.0,\ 255.0]^{28 \times 28}$ as float64 arrays. They are **not yet valid probability measures**: their entries are not normalized (they don't sum to 1), some entries are exactly zero (which causes $\log(0) = -\infty$ in Sinkhorn), and they are still 2D grids rather than distributions over a metric space. All three of these issues must be resolved before optimal transport is computable.

Right now, the sums of pixels of two selected images: $35867.0$ and $27106.0$ are **unequal**. This is the first concrete signal that the two images, when normalized to probability measures, will have different "mass distributions" — and is precisely why naive pixel blending (which ignores mass geometry entirely) produces different results from Wasserstein interpolation.

---

The raw pixel arrays are just grids of brightness values — they carry no probabilistic structure. Optimal transport is a theory about *moving mass between probability measures*, so before we can apply a single result from OT theory, we must convert our images into valid probability measures.

---

A **discrete probability measure** on a finite set $\Omega$ is a function $\mu: \Omega \to [0,1]$ satisfying:

$$\mu(i) \geq 0 \quad \forall i \in \Omega, \qquad \sum_{i \in \Omega} \mu(i) = 1$$

Here $\Omega = \{0, 1, \ldots, n^2 - 1\}$ is the set of **pixel indices** after flattening the $n \times n$ image into a 1D vector. For MNIST, $n = 28$, so $|\Omega| = 784$.

The interpretation is direct: $\mu(i)$ is the **fraction of total ink mass** sitting at pixel $i$. A bright pixel carries a lot of mass; a background pixel carries none.

---

**`flat = img.flatten().astype(np.float64)`**

Reshapes the 2D pixel grid into a 1D vector, destroying spatial structure in a controlled way:

$$A \in \mathbb{R}^{n \times n} \xrightarrow{\ \text{flatten}\ } \mathbf{a} \in \mathbb{R}^{n^2}$$

The mapping is row-major (C-order): row 0 is laid down first, then row 1, etc. Pixel at grid position $(r, c)$ lands at index $i = rn + c$.

On the anchor ($n=2$):

$$A^{(3)} = \begin{pmatrix} 0.0 & 180.0 \\ 200.0 & 255.0 \end{pmatrix} \xrightarrow{\ \text{flatten}\ } \mathbf{a}^{(3)} = (0.0,\ 180.0,\ 200.0,\ 255.0)$$

$$A^{(8)} = \begin{pmatrix} 210.0 & 255.0 \\ 180.0 & 240.0 \end{pmatrix} \xrightarrow{\ \text{flatten}\ } \mathbf{a}^{(8)} = (210.0,\ 255.0,\ 180.0,\ 240.0)$$

**Why flatten?** Optimal transport operates on measures over a *metric space of points*. Each pixel index $i$ will later be assigned a 2D coordinate $\mathbf{x}_i = (r_i, c_i) \in \mathbb{R}^2$, and the cost of transporting mass from $i$ to $j$ will be $\|\mathbf{x}_i - \mathbf{x}_j\|^2$. The 2D grid structure is preserved through this coordinate system — flattening just gives each pixel a single canonical index.

---

**`total = flat.sum()`**

Computes the total brightness — the sum of all pixel values:

$$Z = \sum_{i=0}^{n^2-1} a_i$$

On the code:

$Z^{(3)} = 0.0 + 180.0 + 200.0 + 255.0 = 635.0$, and 

$Z^{(8)} = 210.0 + 255.0 + 180.0 + 240.0 = 885.0$.

$Z$ is the **normalizing constant** — the total mass before we impose the probability constraint. Notice $Z^{(3)} \neq Z^{(8)}$: the two images have different total brightness. This is exactly what the print statement in previous output flagged ($35867 \neq 27106$). After normalization, both measures will sum to 1, so this raw difference is absorbed — each measure describes a *shape of mass distribution*, not an absolute brightness.

---

**`return flat / total`**

Divides every pixel value by the total, producing the normalized measure:

$$\mu(i) = \frac{a_i^{(3)}}{Z^{(3)}}, \qquad \nu(i) = \frac{a_i^{(8)}}{Z^{(8)}}$$

This guarantees $\mu(i) \geq 0$ for all $i$ (since pixel values are non-negative) and $\sum_i \mu(i) = 1$ (by construction).

On the code:

$$\mathbf{m} = \frac{1}{635}(0.0,\ 180.0,\ 200.0,\ 255.0) = (0.0,\ 0.2835,\ 0.3150,\ 0.4016)$$

$$\mathbf{v} = \frac{1}{885}(210.0,\ 255.0,\ 180.0,\ 240.0) = (0.2373,\ 0.2881,\ 0.2034,\ 0.2712)$$

Verify:

$0.0 + 0.2835 + 0.3150 + 0.4016 = 1.0$, and

$0.2373 + 0.2881 + 0.2034 + 0.2712 = 1.0$

---

### THE ZERO MASS PROBLEM

The output shows `m min: 0.000000`. This means **background pixels carry exactly zero mass**: $\mu(i) = 0$, for every pixel that was black in the original image. This is mathematically valid for a probability measure, but it creates a serious numerical problem in the next stage.

Sinkhorn iterations require computing $\log \mu(i)$ for every $i$. But:

$$\log(0) = -\infty$$

This would immediately produce `NaN` values and crash the entire computation. Handling this is one of the most important numerical decisions in the pipeline.

---

We now hold two valid discrete probability measures:

$$\mathbf{m} \in \Delta^{783}, \qquad \mathbf{v} \in \Delta^{783}$$

where $\Delta^{783} = \{\mathbf{p} \in \mathbb{R}^{784} : p_i \geq 0,\ \sum_i p_i = 1\}$ is the **783-dimensional probability simplex**. The geometric intuition: each measure is a *point* on this simplex. The Wasserstein geodesic we want to compute is a *path* connecting these two points, measured with a distance that accounts for the underlying pixel geometry.

---

The probability measures $\mathbf{m}$ and $\mathbf{v}$ currently live on an **abstract index set** $\{0, 1, \ldots, 783\}$ — just a list of 784 slots with no geometry. Optimal transport requires a *metric space*: we need to know the cost of moving mass from pixel $i$ to pixel $j$, which requires knowing where $i$ and $j$ actually *are* in space. Thus, we **embed the abstract index set into $\mathbb{R}^2$** by assigning each pixel index its grid coordinates.

---

**`coords = np.array([[i // 28, i % 28] for i in range(784)], dtype=np.float64)`**

For each flat index $i \in \{0, \ldots, 783\}$, this recovers the corresponding 2D grid position via integer division and remainder — the exact inverse of the row-major flattening:

$$\mathbf{x}_i = \left(\left\lfloor \frac{i}{28} \right\rfloor,\ i \bmod 28\right) \in \{0,\ldots,27\}^2$$

The result is a matrix of pixel coordinates:

$$X = \begin{pmatrix} \mathbf{x}_0 \\ \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_{783} \end{pmatrix} \in \mathbb{R}^{784 \times 2}$$

where row $i$ of $X$ is the spatial location of pixel $i$.

On the anchor example ($n=2$, so we divide by 2 instead of 28):

| $i$ | $\lfloor i/2 \rfloor$ | $i \bmod 2$ | $\mathbf{x}_i$ |
|---|---|---|---|
| 0 | 0 | 0 | $(0,\ 0)$ |
| 1 | 0 | 1 | $(0,\ 1)$ |
| 2 | 1 | 0 | $(1,\ 0)$ |
| 3 | 1 | 1 | $(1,\ 1)$ |

So $X = \begin{pmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{pmatrix}$, recovering exactly the four corners of the 2×2 grid. The flattening from code and the coordinate recovery here are **consistent inverses** of each other: pixel $(r, c)$ flattens to index $i = 2r + c$, and index $i$ recovers to $\mathbf{x}_i = (\lfloor i/2 \rfloor, i \bmod 2) = (r, c)$.

---

With coordinates in hand, we can now define the **ground cost** between any two pixels $i$ and $j$ as squared Euclidean distance:

$$c(i, j) = \|\mathbf{x}_i - \mathbf{x}_j\|^2 = \left(\left\lfloor\frac{i}{n}\right\rfloor - \left\lfloor\frac{j}{n}\right\rfloor\right)^2 + (i \bmod n - j \bmod n)^2$$

This is the cost a unit of mass *pays* to travel from pixel $i$ to pixel $j$. It encodes the crucial geometric intuition: moving mass between nearby pixels is cheap; moving it across the image is expensive. This is what makes Wasserstein interpolation **geometrically aware** in a way that pixel blending is not — pixel blending has no notion of distance between pixel locations whatsoever.

On the anchor, the cost between pixel 0 (top-left, $(0,0)$) and pixel 3 (bottom-right, $(1,1)$) is:

$$c(0, 3) = (0-1)^2 + (0-1)^2 = 2$$

while the cost between pixel 0 and pixel 1 (its immediate neighbor) is:

$$c(0, 1) = (0-0)^2 + (0-1)^2 = 1$$

Transporting mass diagonally costs twice as much as transporting it one step horizontally or vertically.

---

We now have three objects: measures $\mathbf{m}, \mathbf{v} \in \Delta^{783}$ and a coordinate matrix $X \in \mathbb{R}^{784 \times 2}$.

---

The cost matrix $C$ will have $784^2 = 614{,}656$ entries. This part commits to storing that full matrix — a deliberate trade-off we'll examine. More importantly, it constructs the object that gives Wasserstein distance its geometric meaning: the **ground cost matrix** $C$, where $C_{ij}$ encodes how expensive it is to move a unit of mass from pixel $i$ to pixel $j$.

Without $C$, optimal transport is blind to geometry. With it, the transport plan is forced to respect the spatial structure of the image.

---

The ground cost is the squared Euclidean distance between pixel coordinates:

$$C_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|^2 = (r_i - r_j)^2 + (c_i - c_j)^2$$

for all $i, j \in \{0, \ldots, n^2-1\}$, where $\mathbf{x}_i = (r_i, c_i)$ are the coordinates obtained earlier. This defines a $784 \times 784$ matrix:

$$C \in \mathbb{R}^{784 \times 784}_{\geq 0}, \qquad C_{ij} = C_{ji}, \qquad C_{ii} = 0$$

The symmetry $C_{ij} = C_{ji}$ reflects the fact that distance is symmetric — moving mass from $i$ to $j$ costs the same as the reverse. The zero diagonal $C_{ii} = 0$ reflects the fact that keeping mass in place is free.

---

**`diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]`**

This is the key broadcasting step. Let's unpack it carefully.

`coords` has shape $(784, 2)$. The two index manipulations reshape it for broadcasting:

$$\texttt{coords[:, np.newaxis, :]} \quad \text{has shape } (784, 1, 2) \quad \longleftrightarrow \quad \mathbf{x}_i \text{ varies along axis 0}$$

$$\texttt{coords[np.newaxis, :, :]} \quad \text{has shape } (1, 784, 2) \quad \longleftrightarrow \quad \mathbf{x}_j \text{ varies along axis 1}$$

NumPy broadcasts these to shape $(784, 784, 2)$, computing:

$$\texttt{diff}[i, j, :] = \mathbf{x}_i - \mathbf{x}_j = \begin{pmatrix} r_i - r_j \\ c_i - c_j \end{pmatrix}$$

simultaneously for all $784 \times 784$ pairs $(i,j)$ — no Python loop required.

On the anchor example ($n=2$, four pixels with coordinates $(0,0),(0,1),(1,0),(1,1)$):

$$\texttt{diff}[0, 3, :] = (0,0) - (1,1) = (-1, -1)$$
$$\texttt{diff}[0, 1, :] = (0,0) - (0,1) = (0, -1)$$
$$\texttt{diff}[2, 1, :] = (1,0) - (0,1) = (1, -1)$$

---

**`C = np.sum(diff ** 2, axis=-1)`**

Squares each component and sums along the last axis (the 2D coordinate axis), collapsing shape $(784, 784, 2) \to (784, 784)$:

$$C_{ij} = \sum_{k \in \{0,1\}} \texttt{diff}[i,j,k]^2 = (r_i - r_j)^2 + (c_i - c_j)^2$$


On the anchor, we have four pixels with coordinates:

$$\mathbf{x}_0 = (0,0), \quad \mathbf{x}_1 = (0,1), \quad \mathbf{x}_2 = (1,0), \quad \mathbf{x}_3 = (1,1)$$

Each entry $C_{ij} = (r_i - r_j)^2 + (c_i - c_j)^2$. Work through row 0 explicitly:

$$C_{00} = (0-0)^2 + (0-0)^2 = 0$$
$$C_{01} = (0-0)^2 + (0-1)^2 = 0 + 1 = 1$$
$$C_{02} = (0-1)^2 + (0-0)^2 = 1 + 0 = 1$$
$$C_{03} = (0-1)^2 + (0-1)^2 = 1 + 1 = 2$$

All obtained through broadcasting step stated earlier.
The full $4\times 4$ cost matrix is:

$$C = \begin{pmatrix} 0 & 1 & 1 & 2 \\ 1 & 0 & 2 & 1 \\ 1 & 2 & 0 & 1 \\ 2 & 1 & 1 & 0 \end{pmatrix}$$

Read this as: moving mass from pixel 0 (top-left) to pixel 3 (bottom-right) costs 2; moving it one step right or down costs 1; staying put costs 0. The maximum cost is 2, achieved by the diagonal traversal — consistent with $c(0,3) = (-1)^2 + (-1)^2 = 2$.

The print statement verifies this on the real grid: $C_{0,27} = 27^2 = 729$ (moving across an entire row) and $C_{0,783} = 27^2 + 27^2 = 1458$ (moving corner to corner).

---

Using $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ rather than $\|\mathbf{x}_i - \mathbf{x}_j\|$ gives us the **$W_2$ Wasserstein distance** (the "2" refers to this exponent). This choice has deep consequences:

- It makes the optimization problem strongly convex (a line connecting any two points on the graph always sits above or on the graph), which improves numerical behavior.
- The $W_2$ geodesic between two measures has a clean characterization via **displacement interpolation** — mass moves in straight lines in $\mathbb{R}^2$, which is precisely what will give us the morphing effect.
- It connects to a rich mathematical theory (Brenier's theorem, McCann's interpolation) that $W_1$ (with linear cost) does not enjoy in the same form.

---

The $784^2 \approx 614\text{K}$ entries as a concern. At float64, this is $614656 \times 8 \approx 4.9\ \text{MB}$ — large but manageable on modern hardware for a one-time computation. The alternative (computing $C_{ij}$ on the fly during each Sinkhorn iteration) would be slower. This is a classic **memory-vs-compute trade-off**, and here memory wins because $C$ is used repeatedly across many iterations.

---

We now hold the complete problem specification for discrete optimal transport:

- Source measure: $\mathbf{m} \in \Delta^{783}$
- Target measure: $\mathbf{v} \in \Delta^{783}$  
- Ground cost: $C \in \mathbb{R}^{784 \times 784}_{\geq 0}$

---

The unregularized OT problem is:

$$\min_{\gamma \geq 0} \sum_{ij} C_{ij} \gamma_{ij} \qquad \text{subject to} \qquad \gamma \mathbf{1} = \mathbf{m}, \quad \gamma^\top \mathbf{1} = \mathbf{v}$$

The entropy-regularized version adds a penalty that discourages $\gamma$ from being too "concentrated":

$$\min_{\gamma \geq 0} \sum_{ij} C_{ij} \gamma_{ij} - \varepsilon \underbrace{\sum_{ij} \gamma_{ij} \log \gamma_{ij}}_{-H(\gamma)} \qquad \text{subject to} \qquad \gamma \mathbf{1} = \mathbf{m}, \quad \gamma^\top \mathbf{1} = \mathbf{v}$$

where $H(\gamma) = -\sum_{ij} \gamma_{ij} \log \gamma_{ij}$ is the **entropy** of $\gamma$. Maximizing entropy means spreading mass as diffusely as possible. The $-\varepsilon H(\gamma)$ term therefore fights against concentration — and $\varepsilon > 0$ controls how hard it fights.

Two effects follow immediately:

1. **Convexity:** $-H(\gamma)$ is strictly convex, so the full objective is strictly convex — unique minimizer guaranteed.
2. **Smoothness:** The entropic term is differentiable everywhere, unlike the linear program's sharp corners. Iterative algorithms converge smoothly rather than bouncing between vertices of a polytope.

The trade-off: the solution $\gamma_\varepsilon$ is a **blurred approximation** of the true OT plan. As $\varepsilon \to 0$, $\gamma_\varepsilon \to \gamma^*$. As $\varepsilon \to \infty$, $\gamma_\varepsilon \to \mathbf{m}\mathbf{v}^\top$ — the independent coupling, which ignores cost entirely.

---

The key insight that makes Sinkhorn work is that this regularized problem has an **analytical solution** in terms of its Lagrange multipliers. Setting up the Lagrangian for the marginal constraints and solving gives:

$$\gamma_{ij} = \exp\!\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)$$

where $\mathbf{f} \in \mathbb{R}^{784}$ and $\mathbf{g} \in \mathbb{R}^{784}$ are the **dual potentials** — one scalar per source pixel, one per target pixel. They are the Lagrange multipliers for the marginal constraints, and they play the role of "prices": $f_i$ is the price of shipping mass *out of* pixel $i$, $g_j$ is the price of receiving mass *into* pixel $j$.

The marginal constraints then become:

$$\sum_j \exp\!\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right) = m_i \qquad \forall i$$

$$\sum_i \exp\!\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right) = v_j \qquad \forall j$$

Sinkhorn solves these two equations by **alternating**: fix $\mathbf{g}$, solve for $\mathbf{f}$; fix $\mathbf{f}$, solve for $\mathbf{g}$; repeat until convergence. Each step has a closed form — that's why the algorithm is so efficient.

---

**`log_mu = np.log(m + 1e-300)`**

Computes $\log m_i$ for every pixel, with a numerical guard:

$$\tilde{\ell}_i = \log(m_i + \epsilon_{\text{guard}}), \qquad \epsilon_{\text{guard}} = 10^{-300}$$

$10^{-300}$ is chosen to be smaller than any meaningful pixel mass (the smallest nonzero value in $\mathbf{m}$ is $1/35867 \approx 2.8 \times 10^{-5}$) but large enough to keep $\log$ finite. It doesn't alter any nonzero value — $\log(2.8\times10^{-5} + 10^{-300}) \approx \log(2.8\times10^{-5})$ to full float64 precision. It only saves the zero-mass background pixels from producing $-\infty$.

On the anchor, $\mathbf{m} = (0.0,\ 0.2835,\ 0.3150,\ 0.4016)$:

$$\log_{mu} = (\log(10^{-300}),\ \log(0.2835),\ \log(0.3150),\ \log(0.4016)) \approx (-690.8,\ -1.261,\ -1.155,\ -0.912)$$

---

**`f = np.zeros(len(m))`**  
**`g = np.zeros(len(v))`**

Initializes both dual potentials to zero:

$$\mathbf{f}^{(0)} = \mathbf{0} \in \mathbb{R}^{784}, \qquad \mathbf{g}^{(0)} = \mathbf{0} \in \mathbb{R}^{784}$$

With $\mathbf{f} = \mathbf{g} = \mathbf{0}$, the transport plan formula gives $\gamma_{ij} = \exp(-C_{ij}/\varepsilon)$ — mass is distributed according to cost alone, with no marginal constraints enforced yet. Sinkhorn iterations will progressively correct this.

---

**`f = epsilon * log_mu - epsilon * scipy_logsumexp((g[np.newaxis, :] - C) / epsilon, axis=1)`**

This is the Sinkhorn $\mathbf{f}$-update. To derive it: the marginal constraint $\sum_j \gamma_{ij} = m_i$ with $\gamma_{ij} = \exp((f_i + g_j - C_{ij})/\varepsilon)$ gives:

$$\exp\!\left(\frac{f_i}{\varepsilon}\right) \sum_j \exp\!\left(\frac{g_j - C_{ij}}{\varepsilon}\right) = m_i$$

Taking $\varepsilon \log$ of both sides:

$$f_i = \varepsilon \log m_i - \varepsilon \log \sum_j \exp\!\left(\frac{g_j - C_{ij}}{\varepsilon}\right)$$

The second term is a **log-sum-exp** (LSE) — numerically unstable if computed naively (exponentials of large numbers overflow). `scipy_logsumexp` computes $\log\sum_j \exp(z_j)$ by factoring out $\max_j z_j$, keeping everything finite. In compact notation:

$$f_i \leftarrow \varepsilon \log m_i - \varepsilon \operatorname{LSE}_j\!\left(\frac{g_j - C_{ij}}{\varepsilon}\right)$$

On the anchor at iteration 0 (with $\mathbf{g} = \mathbf{0}$, $\varepsilon = 0.01$, using $C$):

$$f_0 \leftarrow \varepsilon \log m_0 - \varepsilon \operatorname{LSE}_j\!\left(\frac{0 - C_{0j}}{\varepsilon}\right) = 0.01 \cdot (-690.8) - 0.01 \cdot \operatorname{LSE}(0,\ -100,\ -100,\ -200)$$

The LSE is dominated by its largest argument (0), so $\approx 0.01\cdot(-690.8) - 0.01\cdot 0 = -6.908$. The zero-mass pixel immediately gets a very negative potential — it will contribute negligible mass to the transport plan, as it should.

---

**`g = epsilon * log_nu - epsilon * scipy_logsumexp((f[:, np.newaxis] - C) / epsilon, axis=0)`**

Identical structure, now enforcing the *target* marginal constraint $\sum_i \gamma_{ij} = v_j$:

$$g_j \leftarrow \varepsilon \log \nu_j - \varepsilon \operatorname{LSE}_i\!\left(\frac{f_i - C_{ij}}{\varepsilon}\right)$$

Note `axis=0` — we sum over source pixels $i$ (rows) rather than target pixels $j$ (columns). The updated $\mathbf{f}$ from the previous line feeds directly into this computation, which is why the two updates are **sequential within each iteration**, not simultaneous.

---

**`if np.max(np.abs(f - f_prev)) < tol: break`**

Convergence check: if the largest change in any component of $\mathbf{f}$ between iterations is smaller than $10^{-9}$, the potentials have stabilized and further iterations won't change the transport plan meaningfully:

$$\|\mathbf{f}^{(k)} - \mathbf{f}^{(k-1)}\|_\infty < 10^{-9} \implies \text{stop}$$

Only $\mathbf{f}$ is checked — because the two updates are coupled, convergence of $\mathbf{f}$ implies convergence of $\mathbf{g}$.

---

**`log_pi = (f[:, np.newaxis] + g[np.newaxis, :] - C) / epsilon`**  
**`pi = np.exp(log_pi)`**

Once the potentials have converged, the transport plan is recovered via the analytical solution derived earlier:

$$\log \pi_{ij} = \frac{f_i + g_j - C_{ij}}{\varepsilon}, \qquad \pi_{ij} = \exp\!\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)$$

Computing in log-space first and exponentiating once at the end is critical — computing $\exp(f_i/\varepsilon)\cdot\exp(g_j/\varepsilon)\cdot\exp(-C_{ij}/\varepsilon)$ directly would overflow for small $\varepsilon$.

On the anchor, after convergence, $\pi$ is a $4\times4$ matrix where $\pi_{ij}$ tells us exactly how much mass flows from source pixel $i$ to target pixel $j$. Its row sums equal $\mathbf{m}$ and its column sums equal $\mathbf{v}$ — the marginal constraints are satisfied.

---

The function `sinkhorn_log` is defined but not yet called — $\varepsilon$ hasn't been chosen and no transport plan has been computed yet.

---

**`C_norm = C / C.max()`**

Divides every entry of $C$ by its maximum value, rescaling the cost matrix to $[0, 1]$:

$$\tilde{C}_{ij} = \frac{C_{ij}}{\max_{kl} C_{kl}} = \frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{1458}$$

On the anchor, $\max C = 2$ (top-left to bottom-right), so:

$$\tilde{C} = \frac{1}{2}\begin{pmatrix} 0 & 1 & 1 & 2 \\ 1 & 0 & 2 & 1 \\ 1 & 2 & 0 & 1 \\ 2 & 1 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0.5 & 0.5 & 1 \\ 0.5 & 0 & 1 & 0.5 \\ 0.5 & 1 & 0 & 0.5 \\ 1 & 0.5 & 0.5 & 0 \end{pmatrix}$$

**Why does this matter?** Recall the Sinkhorn update involves $\exp(g_j - C_{ij})/\varepsilon$. The argument of the exponential is $C_{ij}/\varepsilon$. Without normalization, the maximum cost is 1458 and $\varepsilon = 0.01$, giving exponent magnitudes up to $1458/0.01 = 145800$ — catastrophic overflow even in float64, whose maximum exponent is $\approx 709$. After normalization the maximum is $1/0.01 = 100$ — well within float64 range. Normalizing $C$ effectively makes $\varepsilon$ meaningful as a fraction of the total cost scale.

---

**`pi, f, g = sinkhorn_log(m, v, C_norm, epsilon=epsilon, n_iter=1000)`**

Executes the Sinkhorn iterations defined earlier with $\varepsilon = 0.01$. This $\varepsilon$ sits close to the "sharp" end of the spectrum — recall that $\varepsilon \to 0$ recovers the true OT plan while $\varepsilon \to \infty$ gives the independent coupling $\mathbf{m}\mathbf{v}^\top$. At $\varepsilon = 0.01$, the transport plan is a good approximation of the true $W_2$ plan with only mild entropic blur.

---

The print statements check the **defining properties** of a valid transport plan. Let's state what each one is actually asserting mathematically.

`pi row sums ≈ mu` checks the **source marginal constraint**:

$$\sum_{j=0}^{783} \pi_{ij} = m_i \quad \forall i$$

Every unit of mass that leaves source pixel $i$ must equal $m_i$ — nothing is created or destroyed.

`pi col sums ≈ nu` checks the **target marginal constraint**:

$$\sum_{i=0}^{783} \pi_{ij} = v_j \quad \forall j$$

Every unit of mass arriving at target pixel $j$ must equal $v_j$.

`pi total mass: 1.0` follows directly from either marginal: $\sum_{ij} \pi_{ij} = \sum_i m_i = 1$.

Together, these three constraints define the **transport polytope**:

$$\Pi(\mathbf{m}, \mathbf{v}) = \{\gamma \in \mathbb{R}^{784\times784}_{\geq 0} : \gamma\mathbf{1} = \mathbf{m},\ \gamma^\top\mathbf{1} = \mathbf{v}\}$$

The output confirms $\pi \in \Pi(\mathbf{m}, \mathbf{v})$ to tolerance $10^{-6}$ — Sinkhorn has found a valid coupling.

---

We now hold the **regularized optimal transport plan** $\pi \in \Pi(\mathbf{m}, \mathbf{v})$, a $784 \times 784$ matrix encoding how mass flows from every pixel of the "3" to every pixel of the "8". This is the core geometric object the entire project was building toward.

---

A **one-line verification** of solution quality outputs: $0.009719$. It computes the total transport cost paid by the plan $\pi$ — the inner product of the cost matrix and the transport plan.

---

**`primal_cost = np.sum(C_norm * pi)`**

Computes the elementwise product and sums all entries:

$$\langle C, \pi \rangle = \sum_{i=0}^{783} \sum_{j=0}^{783} \tilde{C}_{ij}\, \pi_{ij}$$

Each term $\tilde{C}_{ij}\, \pi_{ij}$ is the **cost contribution** of moving mass $\pi_{ij}$ from pixel $i$ to pixel $j$ at unit cost $\tilde{C}_{ij}$. Summing over all pairs gives the total cost paid by the entire transport plan.

On the anchor, after convergence:

$$\langle C, \pi \rangle = \sum_{i,j} \tilde{C}_{ij}\,\pi_{ij} = 0 \cdot \pi_{00} + 0.5 \cdot \pi_{01} + 0.5 \cdot \pi_{02} + 1 \cdot \pi_{03} + \cdots$$

Each nonzero $\tilde{C}_{ij}$ penalizes mass that travels — the further it travels, the more it contributes to this sum.

---

**What $0.009719$ means?**

This is the **regularized Wasserstein cost** — the objective value of the entropy-regularized OT problem evaluated at the solution $\pi$. It is *not* quite the true $W_2^2(\mu, \nu)$, because entropic regularization biases the cost downward (the entropy term encourages spreading, which moves some mass along cheaper-than-optimal paths). As $\varepsilon \to 0$, this value converges to the true $W_2^2$.

The value itself — $0.009719$ — is meaningful relative to the normalized cost scale $[0,1]$: it tells us the "3" and "8" are separated by a moderate transport distance, not trivially close and not maximally far.

---

$\pi$ is verified as both feasible and low-cost (primal cost is small relative to the cost scale). We now have full confidence in the transport plan. In figure, the log scale is necessary because most entries of $\pi$ are extremely close to zero — only pixels that are spatially near each other in the two images exchange significant mass. The sparsity structure of $\pi$ is itself a geometric statement: the transport plan concentrates mass along a narrow band near the diagonal, meaning the "3" and "8" are not wildly different in their spatial mass distributions.

---

An empirical study of the $\varepsilon$ trade-off described is presented. Three values are tested; the primal cost strictly decreases as $\varepsilon$ decreases. This output makes the theoretical claim concrete.

---

| $\varepsilon$ | Primal cost |
|---|---|
| $1.00$ | $0.063129$ |
| $0.10$ | $0.040568$ |
| $0.01$ | $0.009719$ |

Each tenfold decrease in $\varepsilon$ pulls the cost significantly downward. The interpretation: at $\varepsilon = 1.0$, the entropy term dominates and forces $\pi$ to spread mass diffusely — even expensive long-distance transport paths get used, inflating the cost. At $\varepsilon = 0.01$, the cost term dominates and $\pi$ concentrates mass on cheap short-distance paths, approaching the true $W_2$ plan.

Mathematically, the regularized objective is:

$$\mathcal{L}_\varepsilon(\gamma) = \underbrace{\langle C, \gamma \rangle}_{\text{transport cost}} - \varepsilon \underbrace{H(\gamma)}_{\text{entropy}}$$

As $\varepsilon$ grows, maximizing $H(\gamma)$ increasingly overrides minimizing $\langle C, \gamma \rangle$, pulling the solution toward the maximum-entropy coupling $\mathbf{m}\mathbf{v}^\top$. Its cost on the anchor would be:

$$\langle C, \mathbf{m}\mathbf{v}^\top \rangle = \sum_{ij} C_{ij} m_i v_j$$

— a weighted average of all pairwise costs, with no geometric preference whatsoever.

---

```
print(f"epsilon={eps:.2f} -> primal cost: {cost:.6f} | "
          f"marginals OK: {np.allclose(pi_eps.sum(axis=1), m, atol=1e-5)}")
```

Outputs `marginals OK: True` across all three values. This is not a coincidence — the Sinkhorn updates are derived by enforcing the marginal constraints exactly at each step, regardless of $\varepsilon$. The regularization affects *which* feasible point is selected, not whether the marginal constraints are satisfied.

---

The choice $\varepsilon = 0.01$ used is now justified empirically: it gives the sharpest transport plan while remaining numerically stable after $C$ normalization. 

---

We further alias `f` and `g` to `phi` and `psi` — no computation. However, this reveals something worth examining carefully before we reach the interpolation.

---

The dual potentials are the Lagrange multipliers from the regularized OT problem. Recall from previous statement that the transport plan is recovered as:

$$\pi_{ij} = \exp\!\left(\frac{\phi_i + \psi_j - \tilde{C}_{ij}}{\varepsilon}\right)$$

The range of $\phi$ is $[-6.9562,\ -0.0750]$ — entirely negative. Think about what that means for a pixel with very negative $\phi_i$, say $\phi_i = -6.9562$.

Plugging into the formula: the numerator $\phi_i + \psi_j - \tilde{C}_{ij}$ is pulled very negative, so $\pi_{ij} \approx \exp(-\text{large number}) \approx 0$ for all $j$.

Which pixels would have $\phi_i \approx -6.9562$? Recall that the first Sinkhorn update is:

$$\phi_i \leftarrow \varepsilon \log m_i - \varepsilon \operatorname{LSE}_j\!\left(\frac{\psi_j - \tilde{C}_{ij}}{\varepsilon}\right)$$

The $\varepsilon \log m_i$ term dominates the initialization. For zero-mass background pixels, $m_i \approx 10^{-300}$, giving $\varepsilon \log m_i \approx 0.01 \times (-690) = -6.9$. So the most negative values of $\phi$ correspond precisely to background pixels — they are priced out of the transport plan, carrying negligible mass exactly as intended.

---

$\phi$ and $\psi$ are extracted and ready. They are not used directly in the interpolation — $\pi$ carries all the information needed. Their role here is interpretive: they confirm the dual certificate of optimality.

---

We now compute the full regularized objective function — not just the transport cost, but the complete quantity that Sinkhorn actually minimized. It puts concrete numbers on the trade-off between cost and entropy that has been discussed theoretically.

---

**`H_pi = -np.sum(pi * np.log(pi + 1e-300))`**

Computes the **Shannon entropy** of the transport plan:

$$H(\pi) = -\sum_{i,j} \pi_{ij} \log \pi_{ij}$$

Entropy measures how *spread out* $\pi$ is. If all mass were concentrated on a single $(i,j)$ pair, $H(\pi) = 0$ — maximally concentrated. If mass were spread uniformly across all $784^2$ pairs, $H(\pi) = \log(784^2) \approx 13.4$ — maximally diffuse. The output $H(\pi) = 8.95$ sits between these extremes, reflecting a plan that is spread but not uniform — consistent with $\varepsilon = 0.01$ being small but nonzero.

---

**`primal_obj = np.sum(C_norm * pi) - epsilon * H_pi`**

Evaluates the full regularized objective at the solution:

$$\mathcal{L}_\varepsilon(\pi) = \langle \tilde{C}, \pi \rangle - \varepsilon H(\pi) = 0.00971915 - 0.01 \times 8.94968029 = 0.00971915 - 0.08949680 = -0.07977765$$

The objective is **negative** — which is initially surprising. How can a minimization problem have a negative optimal value?

The answer: $-\varepsilon H(\pi)$ is always negative (entropy is non-negative, multiplied by $-\varepsilon$), and for small $\varepsilon$ and a sufficiently diffuse $\pi$, this entropy bonus can outweigh the transport cost. The negativity is not pathological — it simply reflects that the entropy regularization has shifted the objective below zero. As $\varepsilon \to 0$, the entropy term vanishes and the objective returns to the non-negative transport cost $\langle \tilde{C}, \pi \rangle$.

---

| Quantity | Value | Meaning |
|---|---|---|
| $\langle \tilde{C}, \pi \rangle$ | $0.00972$ | Cost paid to move mass geometrically |
| $\varepsilon H(\pi)$ | $0.08950$ | Entropy bonus collected for spreading |
| $\mathcal{L}_\varepsilon(\pi)$ | $-0.07978$ | Net objective — what Sinkhorn minimized |

The entropy bonus is nearly **ten times larger** than the transport cost at $\varepsilon = 0.01$. This might seem to contradict the earlier claim that $\varepsilon = 0.01$ is "close to the sharp OT plan." It doesn't — the transport cost being small means the plan is geometrically efficient. The large entropy bonus is a separate statement: the plan is also quite spread, because even at $\varepsilon = 0.01$ there are $784^2 \approx 614\text{K}$ entries contributing to $H(\pi)$.

---

The solution is fully characterized: feasible (marginals match), geometrically efficient (low transport cost), and optimal for the regularized objective.

---

As $\varepsilon \to 0$, the plan $\pi$ sharpens toward a sparse deterministic map, $H(\pi)$ shrinks, and the entropy bonus collapses. The objective converges to the bare transport cost $\langle \tilde{C}, \pi \rangle$.

---

Now we verifiy one of the deepest theoretical guarantees in convex optimization: **strong duality**. The primal and dual objectives agree to within $1.79 \times 10^{-10}$ — essentially machine precision.

---

**`dual_obj = np.dot(phi, m) + np.dot(psi, v)`**

Computes the **dual objective**:

$$\mathcal{D}(\phi, \psi) = \langle \phi, \mathbf{m} \rangle + \langle \psi, \mathbf{v} \rangle = \sum_{i} \phi_i m_i + \sum_j \psi_j v_j$$

Each term has a clean economic interpretation: $\phi_i m_i$ is the "revenue" collected at source pixel $i$ — the price $\phi_i$ times the mass $m_i$ available there. $\psi_j v_j$ is the revenue collected at target pixel $j$. The dual objective is the total revenue across both sides of the transport market.

On the anchor:

$$\mathcal{D}(\phi, \psi) = \phi_0 \cdot 0.0 + \phi_1 \cdot 0.2835 + \phi_2 \cdot 0.3150 + \phi_3 \cdot 0.4016 + \psi_0 \cdot 0.2373 + \cdots$$

---

The duality gap is $|\mathcal{L}_\varepsilon(\pi) - \mathcal{D}(\phi,\psi)| = 1.79 \times 10^{-10} \approx 0$. This confirms:

$$\mathcal{L}_\varepsilon(\pi) = \mathcal{D}(\phi, \psi)$$

Why does this hold? The regularized OT problem is **strictly convex** with a compact feasible set — by strong duality theory (Fenchel-Rockafellar), the primal and dual optima coincide exactly. The tiny residual gap is purely numerical, not mathematical.

The practical significance: the potentials $\phi, \psi$ returned by Sinkhorn are not just algorithmic byproducts — they are the **true dual optimizers**, certifying that $\pi$ is the globally optimal primal solution. No other transport plan in $\Pi(\mathbf{m}, \mathbf{v})$ achieves a lower regularized cost.

---

The transport plan $\pi$ is fully certified from both primal and dual perspectives.

---

Now we verifies the **optimality conditions** of the regularized OT problem geometrically. Strong duality  ocd confirmed the objectives match. Complementary slackness goes further — it specifies *where* in the $(i,j)$ plane the transport plan is allowed to place mass.

---

**`slack = C_norm - (phi[:, np.newaxis] + psi[np.newaxis, :])`**

Computes the **slack** at every $(i,j)$ pair:

$$s_{ij} = \tilde{C}_{ij} - (\phi_i + \psi_j)$$

Recall that at optimality:

$$\pi_{ij} = \exp\!\left(\frac{\phi_i + \psi_j - \tilde{C}_{ij}}{\varepsilon}\right) = \exp\!\left(\frac{-s_{ij}}{\varepsilon}\right)$$

So $s_{ij}$ directly controls how much mass flows along path $(i,j)$: when $s_{ij} = 0$, the path is "free" and $\pi_{ij}$ is maximized; when $s_{ij}$ is large, $\pi_{ij} \approx 0$ and the path is effectively closed.

`slack.min() = 0.074001 > 0` confirms $s_{ij} \geq 0$ everywhere — the dual constraint $\phi_i + \psi_j \leq \tilde{C}_{ij}$ holds globally. This is the dual feasibility condition.

On the anchor: $s_{03} = \tilde{C}_{03} - (\phi_0 + \psi_3)$. If $\phi_0 + \psi_3 < \tilde{C}_{03} = 1$, then $s_{03} > 0$ and very little mass travels the expensive diagonal path.

---

**`weighted_slack = slack * pi`**

Computes $s_{ij} \cdot \pi_{ij}$ at every pair — the complementary slackness product:

$$s_{ij} \cdot \pi_{ij} \approx 0 \quad \forall i,j$$

In unregularized OT, complementary slackness is exact: $s_{ij} \cdot \pi_{ij} = 0$ everywhere, meaning mass only flows along paths where $\phi_i + \psi_j = \tilde{C}_{ij}$ exactly. The regularized version softens this — mass can flow along slightly suboptimal paths, but exponentially suppressed by $\exp(-s_{ij}/\varepsilon)$. The output confirms this: `weighted_slack.max() = 4.52e-05` and `mean = 1.46e-07` — both negligible.

---

These three conditions together — primal feasibility ($\pi \in \Pi(\mathbf{m},\mathbf{v})$), dual feasibility ($s_{ij} \geq 0$), and complementary slackness ($s_{ij}\pi_{ij} \approx 0$) — are the **KKT conditions** for the regularized OT problem. Satisfying all three simultaneously is both necessary and sufficient for global optimality. Sinkhorn has found the exact solution.

---

$\pi$ is certified optimal from every angle: primal feasibility, primal-dual objective match, and KKT conditions. The mathematical groundwork is complete. Everything from here is about *using* $\pi$ to build the geodesic.

---

The geometric payoff of everything built so far: the transport plan $\pi$ tells us *how much mass* moves from pixel $i$ to pixel $j$. This asks: what does the mass distribution look like *mid-journey*? The answer is the **Wasserstein geodesic** — the mathematically correct "straight line" between $\mu$ and $\nu$ in the space of probability measures.

---

At time $t \in [0,1]$, define the interpolation map:

$$T_t(\mathbf{x}_i, \mathbf{x}_j) = (1-t)\,\mathbf{x}_i + t\,\mathbf{x}_j$$

This moves each unit of mass linearly from its source position $\mathbf{x}_i$ toward its target position $\mathbf{x}_j$. At $t=0$: mass stays at $\mathbf{x}_i$ — we recover $\mu$. At $t=1$: mass arrives at $\mathbf{x}_j$ — we recover $\nu$. At $t=0.5$: mass sits at the midpoint $\frac{\mathbf{x}_i + \mathbf{x}_j}{2}$ — the intermediate frame.

The resulting measure at time $t$ is the **pushforward** of $\pi$ under $T_t$:

$$\mu_t = (T_t)_\sharp \pi$$

Formally, for any pixel location $\mathbf{x}_k$:

$$\mu_t(k) = \sum_{\substack{i,j \\ T_t(\mathbf{x}_i,\mathbf{x}_j) = \mathbf{x}_k}} \pi_{ij}$$

In words: accumulate all mass $\pi_{ij}$ whose interpolated position lands at pixel $k$ at time $t$.

---

**`frame = np.zeros(784)`**

Initializes a blank canvas — the measure $\mu_t$ before any mass has been placed:

$$\mu_t^{(0)} = \mathbf{0} \in \mathbb{R}^{784}$$

---

**`if pi[i].sum() < 1e-12: continue`**

Skips source pixel $i$ if it carries no outgoing mass. Background pixels satisfy $m_i \approx 0$, so $\sum_j \pi_{ij} \approx 0$ — computing their interpolation would waste time and contribute nothing.

---

**`interp_coords = (1 - t) * coords[i] + t * coords`**

For fixed source pixel $i$ and all target pixels $j$ simultaneously, computes the interpolated 2D position:

$$\mathbf{z}_{ij}(t) = (1-t)\,\mathbf{x}_i + t\,\mathbf{x}_j \in \mathbb{R}^2$$

`coords[i]` has shape $(2,)$ and `coords` has shape $(784, 2)$, so broadcasting gives `interp_coords` shape $(784, 2)$ — one interpolated coordinate per target pixel $j$.

On the anchor at $t = 0.5$, for source pixel $i=0$ at $(0,0)$:

$$\mathbf{z}_{00}(0.5) = 0.5(0,0) + 0.5(0,0) = (0,0)$$
$$\mathbf{z}_{01}(0.5) = 0.5(0,0) + 0.5(0,1) = (0,\ 0.5)$$
$$\mathbf{z}_{02}(0.5) = 0.5(0,0) + 0.5(1,0) = (0.5,\ 0)$$
$$\mathbf{z}_{03}(0.5) = 0.5(0,0) + 0.5(1,1) = (0.5,\ 0.5)$$

---

**`rows = np.clip(np.round(interp_coords[:, 0]).astype(int), 0, 27)`**

Rounds each interpolated coordinate to the nearest integer pixel, then clamps to $[0, 27]$:

$$r_{ij} = \text{clip}\!\left(\text{round}(z_{ij}^{(0)}),\ 0,\ 27\right), \qquad c_{ij} = \text{clip}\!\left(\text{round}(z_{ij}^{(1)}),\ 0,\ 27\right)$$

On the anchor at $t=0.5$: $(0, 0.5, 0.5, 0.5) \xrightarrow{\text{round}} (0, 1, 1, 1)$ for rows... wait — let's be precise. $z_{01}^{(0)} = 0$ rounds to 0; $z_{02}^{(0)} = 0.5$ rounds to 0 or 1 depending on rounding rule (NumPy uses banker's rounding — 0.5 rounds to 0). This is a subtle discretization artifact: two continuously distinct positions can collapse onto the same pixel after rounding, which is why the next line uses `np.add.at` rather than direct assignment.

---

**`target_indices = rows * 28 + cols`**

Converts 2D rounded coordinates back to flat indices:

$$k_{ij} = 28\, r_{ij} + c_{ij} \in \{0,\ldots,783\}$$

---

**`np.add.at(frame, target_indices, pi[i])`**

Deposits mass $\pi_{ij}$ at pixel $k_{ij}$ for all $j$ simultaneously:

$$\mu_t(k_{ij}) \mathrel{+}= \pi_{ij} \quad \forall j$$

The critical word is `add.at` rather than `frame[target_indices] += pi[i]`. Standard NumPy indexing with repeated indices is **not atomic** — if $k_{i,j_1} = k_{i,j_2}$ (two paths land on the same pixel), only one increment would register. `np.add.at` guarantees all contributions are accumulated correctly, regardless of collisions.

---

The output confirms `frames sum ≈ 1.0` for every $t$. This is not accidental — it follows directly from the marginal constraints on $\pi$:

$$\sum_k \mu_t(k) = \sum_{i,j} \pi_{ij} = 1 \quad \forall t$$

Mass is neither created nor destroyed during transport — it only moves.

---

In the space of probability measures equipped with the $W_2$ metric, the curve $t \mapsto \mu_t$ is the **unique constant-speed geodesic** connecting $\mu$ and $\nu$. This is McCann's interpolation theorem (1997). The key property: each particle of mass travels in a straight line in $\mathbb{R}^2$ at constant speed. No other interpolation scheme — including pixel blending — produces straight-line trajectories for individual mass particles.

---

We now hold 10 frames $\{\mu_t\}_{t=0}^{1}$ — the Wasserstein geodesic sampled at uniform time steps. Pixel blending is linear interpolation on the simplex; Wasserstein is linear interpolation in $\mathbb{R}^2$ for each mass particle. The geometry lives in the image plane, not in intensity space.

---

We simply confirms that the geodesic has the correct endpoints:

$$\mu_0 = \mu \qquad \text{and} \qquad \mu_1 = \nu$$

Both output hold to tolerance $10^{-5}$. The small residual is purely from the pixel rounding — $T_t$ at $t=0$ maps every mass particle to $\mathbf{x}_i$ exactly, but `np.round` can introduce sub-pixel error at intermediate $t$ values. At the endpoints $t \in \{0,1\}$ no rounding ambiguity occurs since $(1-t)\mathbf{x}_i + t\mathbf{x}_j$ lands exactly on a grid point.

---

Naive Pixel Blending Baseline: constructs the **straw man** that the Wasserstein geodesic is contrasted against. The formula is as simple as it looks — and that simplicity is precisely its flaw.

---

**`(1-t) * m.reshape(28, 28) + t * v.reshape(28, 28)`**

A convex combination of the two measures, computed pointwise at each pixel:

$$\mu_t^{\text{blend}}(k) = (1-t)\,m_k + t\,v_k \qquad \forall k \in \{0,\ldots,783\}$$

This is a straight line on the probability simplex $\Delta^{783}$ — linear interpolation between two points in intensity space, with no reference to pixel coordinates whatsoever.

On the anchor at $t = 0.5$:

$$\mu_{0.5}^{\text{blend}} = 0.5\,(0.0,\ 0.2835,\ 0.3150,\ 0.4016) + 0.5\,(0.2373,\ 0.2881,\ 0.2034,\ 0.2712)$$
$$= (0.1186,\ 0.2858,\ 0.2592,\ 0.3364)$$

No pixel moves. Pixel 0 goes from mass $0.0$ to mass $0.1186$ purely by arithmetic averaging — no mass traveled there from anywhere.

---

Mass conservation holds trivially by linearity:

$$\sum_k \mu_t^{\text{blend}}(k) = (1-t)\sum_k m_k + t\sum_k v_k = (1-t)\cdot 1 + t \cdot 1 = 1$$

---

The contrast: At $t=0.5$ on the real images, consider a pixel in the middle of a stroke of the "3" that has no corresponding ink in the "8":

- **Blending:** $\mu_{0.5}^{\text{blend}}(k) = 0.5\,m_k + 0\cdot v_k = 0.5\,m_k$ — the stroke simply fades to half intensity. No mass moves; the stroke just dims.
- **Wasserstein:** $\mu_{0.5}(k)$ accumulates mass from all $(i,j)$ pairs whose midpoint $\frac{\mathbf{x}_i + \mathbf{x}_j}{2}$ lands at $k$ — mass physically migrates from the "3" stroke toward wherever the "8" needs it.

---

Both methods produce valid probability measures at every $t$, and both have correct boundary conditions. The Wasserstein geodesic maintains spatial correlation, ensuring mass physically travels through intermediate coordinates rather than simply fading out in one spot and fading in at another.

---
---