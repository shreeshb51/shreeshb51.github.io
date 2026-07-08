# Anchor Example — Complete Step-by-Step Solution
---

## Introduction of the Anchor Example

To trace our generative modeling framework, we first initialize the clean data tensor $x_0$ and the target noise tensor $x_1$ according to our exact geometric specifications.

Our tensors have a batch size of $N=2$, channel count $C=3$ (RGB), and spatial dimensions $H=4, W=4$, yielding a shape of `(2, 3, 4, 4)`.

* **Sample 1 (The Checkerboard):** A structural grid where values alternate perfectly across space between $+1.0$ (white) and $-1.0$ (black). This structure is duplicated identically across all $C=3$ channels.
* **Sample 2 (The Spatial Gradient):** A smooth left-to-right intensity transition matrix scaled symmetrically within $[-1.0, +1.0]$. This structure is duplicated identically across all $C=3$ channels.
* **Target Noise ($x_1$):** A static, reproducible normal distribution tensor where every single element is locked to $+0.5$.

<br>

### Numerical State of the Anchor Tensors

#### Clean Data Tensor ($x_0$)

For Channel $c \in \{0, 1, 2\}$ of Sample 0 (Checkerboard):

$$x_0[0, c] = \begin{bmatrix} +1.0 & -1.0 & +1.0 & -1.0 \\ -1.0 & +1.0 & -1.0 & +1.0 \\ +1.0 & -1.0 & +1.0 & -1.0 \\ -1.0 & +1.0 & -1.0 & +1.0 \end{bmatrix}$$

For Channel $c \in \{0, 1, 2\}$ of Sample 1 (Spatial Gradient):

$$x_0[1, c] = \begin{bmatrix} -1.0 & -0.33 & +0.33 & +1.0 \\ -1.0 & -0.33 & +0.33 & +1.0 \\ -1.0 & -0.33 & +0.33 & +1.0 \\ -1.0 & -0.33 & +0.33 & +1.0 \end{bmatrix}$$

<br>

#### Target Noise Tensor ($x_1$)

For Channel $c \in \{0, 1, 2\}$ of Sample $n \in \{0, 1\}$:

$$x_1[n, c] = \begin{bmatrix} +0.5 & +0.5 & +0.5 & +0.5 \\ +0.5 & +0.5 & +0.5 & +0.5 \\ +0.5 & +0.5 & +0.5 & +0.5 \\ +0.5 & +0.5 & +0.5 & +0.5 \end{bmatrix}$$

---

## Cell 4: Global Variable Initialization and Seeding

This cell calculates the empirical mean and standard deviation for each color channel across the entire training dataset of 50,000 raw images. This operation establishes the baseline dataset statistics before any scaling transformations are applied.

<br>

### Mathematical Formulation

Given a dataset tensor $\mathbf{X}$ of shape $(M, H, W, C)$, where $M = 50000$ is the total number of images, $H = 32$ is the height, $W = 32$ is the width, and $C = 3$ is the number of color channels, let $X_{m, i, j, c}$ denote the pixel intensity value at sample $m$, spatial coordinate $(i, j)$, and channel $c$.

The empirical mean $\mu_c$ for a specific channel $c$ is computed by taking the arithmetic average over all samples and spatial locations:

$$\mu_c = \frac{1}{M \cdot H \cdot W} \sum_{m=1}^{M} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{m, i, j, c}$$

The empirical standard deviation $\sigma_c$ for a specific channel $c$ is computed as the square root of the channel's variance, which measures the average squared deviation of the pixels from their respective channel mean:

$$\sigma_c = \sqrt{\frac{1}{M \cdot H \cdot W} \sum_{m=1}^{M} \sum_{i=1}^{H} \sum_{j=1}^{W} (X_{m, i, j, c} - \mu_c)^2}$$

<br>

### Code-to-Equation Breakdown

#### Line 1: `imgs_f = imgs.astype(np.float32)`

* **Explanation:** This line converts the 8-bit unsigned integer pixel values ($[0, 255]$) into 32-bit floating-point numbers to prevent arithmetic overflow or rounding truncation during subsequent aggregation operations.
* **Mathematical Operation:** $X_{m,i,j,c} \in \mathbb{N}_0 \cap [0, 255] \implies X_{m,i,j,c} \in \mathbb{R}$
* **Anchor Demonstration:** Since our anchor tensor $x_0$ is already specified in floating-point format, this operation preserves its elements exactly as defined.

<br>

#### Line 2: `ch_mean = imgs_f.mean(axis=(0, 1, 2))`

* **Explanation:** This line collapses axes 0 (samples), 1 (rows), and 2 (columns) by summing all corresponding pixel values and dividing by the total count $M \cdot H \cdot W$, yielding a unique scalar mean for each channel $c$.
* **Mathematical Operation:** Evaluation of $\mu_c$ for $c \in \{0, 1, 2\}$.
* **Anchor Demonstration:** Let's explicitly calculate $\mu_c$ using our anchor data tensor $x_0$ over its $N=2$ samples, $H=4$ rows, and $W=4$ columns (totaling $2 \times 4 \times 4 = 32$ elements per channel).

For any channel $c$, the sum of all elements in Sample 0 is:

$$\sum_{i=1}^{4}\sum_{j=1}^{4} x_0[0, c, i, j] = (+1.0 - 1.0 + 1.0 - 1.0) + (-1.0 + 1.0 - 1.0 + 1.0) + (+1.0 - 1.0 + 1.0 - 1.0) + (-1.0 + 1.0 - 1.0 + 1.0) = 0.0$$

The sum of all elements in Sample 1 is:

$$\sum_{i=1}^{4}\sum_{j=1}^{4} x_0[1, c, i, j] = 4 \times (-1.0 - 0.33 + 0.33 + 1.0) = 4 \times (0.0) = 0.0$$

Combining these yields the total sum for channel $c$:

$$\text{Total Sum}_c = 0.0 + 0.0 = 0.0$$

$$\mu_c = \frac{0.0}{32} = 0.0$$

Thus, on our anchor example:

$$\text{ch\_mean} = [0.0, 0.0, 0.0]$$

<br>

#### Line 3: `ch_std = imgs_f.std(axis=(0, 1, 2))`

* **Explanation:** This line computes the standard deviation across axes 0, 1, and 2 for each channel by taking the square root of the mean squared difference between each pixel and its channel mean $\mu_c$.
* **Mathematical Operation:** Evaluation of $\sigma_c$ for $c \in \{0, 1, 2\}$.
* **Anchor Demonstration:** Because $\mu_c = 0.0$ for all channels on our anchor dataset, the deviation formula simplifies to $\sqrt{\frac{1}{32} \sum (X_{m,i,j,c})^2}$. We compute the sum of squares for each sample.

For Sample 0 (all 16 entries have a magnitude of $|1.0|$):

$$\sum_{i=1}^{4}\sum_{j=1}^{4} (x_0[0, c, i, j] - 0.0)^2 = 16 \times (1.0)^2 = 16.0$$

For Sample 1 (each row contains elements $-1.0, -0.33, +0.33, +1.0$):

$$\text{Row Sum of Squares} = (-1.0)^2 + (-0.33)^2 + (+0.33)^2 + (+1.0)^2 = 1.0 + 0.1089 + 0.1089 + 1.0 = 2.2178$$

Since there are 4 identical rows:

$$\sum_{i=1}^{4}\sum_{j=1}^{4} (x_0[1, c, i, j] - 0.0)^2 = 4 \times 2.2178 = 8.8712$$

Summing the squared deviations across both samples:

$$\text{Total Variance Sum}_c = 16.0 + 8.8712 = 24.8712$$

Dividing by the total number of items ($N \cdot H \cdot W = 32$):

$$\text{Variance}_c = \frac{24.8712}{32} = 0.777225$$

$$\sigma_c = \sqrt{0.777225} \approx 0.8816$$

Thus, on our anchor example:

$$\text{ch\_std} = [0.882, 0.882, 0.882]$$

---

## Cell 6: CIFAR-10 Dataset Normalization and Data Loader Setups

This cell implements an affine transformation to normalize the raw dataset pixel intensities from their original integer range $[0, 255]$ into a bounded symmetric floating-point range $[-1, 1]$. This normalization step ensures stable numerical dynamics, prevents early saturation in deep neural networks, and forms the mathematical boundary conditions for both the DDPM forward process and the Rectified Flow velocity fields.

<br>

### Mathematical Formulation

Let $X_{m, i, j, c} \in [0, 255]$ be the raw pixel intensity value of the image tensor. The normalization map $f: [0, 255] \to [-1, 1]$ is defined by the linear transformation:

$$f(X) = \frac{X - \alpha}{\beta}$$

Where $\alpha = 127.5$ shifts the midpoint of the domain to zero, and $\beta = 127.5$ scales the absolute boundary limits symmetrically to unity. Expanding this relation reveals its explicit linear form:

$$f(X) = \frac{X}{127.5} - \frac{127.5}{127.5} = \frac{X}{127.5} - 1.0$$

The empirical statistics of the transformed domain—minimum ($v_{\min}$), maximum ($v_{\max}$), global mean ($\mu_{\text{global}}$), and global standard deviation ($\sigma_{\text{global}}$)—are evaluated over the entire dataset across all samples ($M$), rows ($H$), columns ($W$), and channels ($C$):

$$v_{\min} = \min_{m, i, j, c} f(X_{m, i, j, c})$$

$$v_{\max} = \max_{m, i, j, c} f(X_{m, i, j, c})$$

$$\mu_{\text{global}} = \frac{1}{M \cdot H \cdot W \cdot C} \sum_{m=1}^{M} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} f(X_{m, i, j, c})$$

$$\sigma_{\text{global}} = \sqrt{\frac{1}{M \cdot H \cdot W \cdot C} \sum_{m=1}^{M} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} (f(X_{m, i, j, c}) - \mu_{\text{global}})^2}$$

<br>

### Code-to-Equation Breakdown

#### Line 1: `imgs_norm = (imgs_f / 127.5) - 1.0`

* **Explanation:** This line scales every pixel element in the floating-point array dividing by $127.5$, and then shifts the range down by subtracting $1.0$.
* **Mathematical Operation:** Evaluation of $f(X)$ on every element in the tensor.
* **Anchor Demonstration:** Per your explicit setup guidelines, our anchor data tensor $x_0$ is *already* defined within the target normalized space $[-1, 1]$. To demonstrate the execution of this line, we look at how raw values map to our anchor values.
	* A raw white pixel $X = 255.0$ yields: $\frac{255.0}{127.5} - 1.0 = 2.0 - 1.0 = +1.0$ (matching the positive elements of the Checkerboard pattern).
	* A raw black pixel $X = 0.0$ yields: $\frac{0.0}{127.5} - 1.0 = 0.0 - 1.0 = -1.0$ (matching the negative elements of the Checkerboard and Spatial Gradient boundaries).
	* A raw pixel of $X = 85.425$ yields: $\frac{85.425}{127.5} - 1.0 = 0.67 - 1.0 = -0.33$ (matching the lower intermediate value of the Spatial Gradient).

<br>

#### Line 3 & 4: `vmin, vmax = imgs_norm.min(), imgs_norm.max()`

* **Explanation:** These lines search the entire dataset tensor for the absolute minimum and maximum element values.
* **Mathematical Operation:** $v_{\min}$ and $v_{\max}$.
* **Anchor Demonstration:** Looking at all elements across Sample 0 and Sample 1 of our anchor tensor $x_0$:
	* The lowest value present is $-1.0$ (found in both samples). Thus, $v_{\min} = -1.0$.
	* The highest value present is $+1.0$ (found in both samples). Thus, $v_{\max} = +1.0$.

<br>

#### Line 5 & 6: `vmean = imgs_norm.mean()` and `vstd = imgs_norm.std()`

* **Explanation:** These lines compute the scalar global mean and global standard deviation across every element in the normalized tensor.
* **Mathematical Operation:** Evaluation of $\mu_{\text{global}}$ and $\sigma_{\text{global}}$.
* **Anchor Demonstration:** * From Cell 4, the mean of each individual channel $c$ was calculated as $\mu_c = 0.0$. The global mean is the average of the channel means:

$$\mu_{\text{global}} = \frac{0.0 + 0.0 + 0.0}{3} = 0.0$$

Because the mean is $0.0$, the global variance matches our individual channel variances calculated in Cell 4 ($\text{Variance}_c = 0.777225$). Taking the square root gives the global standard deviation:

$$\sigma_{\text{global}} = \sqrt{0.777225} \approx 0.8816$$

---

## Cell 7: Linear and Cosine Variance Schedules for DDPM Forward Pass

This cell defines the discrete-time forward variance schedule for the Denoising Diffusion Probabilistic Model (DDPM) baseline. The forward process systematically degrades the data structure by adding Gaussian noise over $T=1000$ discrete steps. Precomputing these parameters allows us to sample the noisy state at any arbitrary time step $t$ analytically in a single step, avoiding a sequential simulation loop.

<br>

### Mathematical Formulation

The forward process is modeled as a Markov chain where the transition distribution from step $t-1$ to step $t$ is governed by a variance parameter $\beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})$$

We define $\alpha_t = 1 - \beta_t$. By unrolling the recursive definition of the process, we can express the distribution of $x_t$ directly given the clean initial data $x_0$:

$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1}$$

Substituting $x_{t-1}$ recursively down to $x_0$ yields:

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

Where $\bar{\alpha}_t$ represents the cumulative product of the variance scale factors up to time step $t$:

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)$$

A linear schedule distributes $\beta_t$ evenly over the interval $[t_1, t_T]$:

$$\beta_t = \beta_{\min} + \frac{t - 1}{T - 1}(\beta_{\max} - \beta_{\min})$$

<br>

### Code-to-Equation Breakdown

#### Line 5: `betas = torch.linspace(beta_min, beta_max, T)`

* **Explanation:** Generates a 1D tensor of $T=1000$ equally spaced values starting from $\beta_{\min}=10^{-4}$ to $\beta_{\max}=0.02$.
* **Mathematical Operation:** Computes $\beta_t$ for $t \in \{1, 2, \dots, 1000\}$.
* **Anchor Demonstration:** Let's calculate the values for the first two steps ($t=1, t=2$) and the final step ($t=1000$):

$$\beta_1 = 10^{-4} + \frac{1 - 1}{999}(0.02 - 10^{-4}) = 0.000100$$

$$\beta_2 = 10^{-4} + \frac{2 - 1}{999}(0.02 - 10^{-4}) = 0.000100 + \frac{0.0199}{999} \approx 0.000120$$

$$\beta_{1000} = 10^{-4} + \frac{1000 - 1}{999}(0.02 - 10^{-4}) = 0.020000$$

<br>

#### Line 6: `alphas = 1.0 - betas`

* **Explanation:** Subtracts each variance element from $1.0$ to compute the remaining signal scale factors.
* **Mathematical Operation:** $\alpha_t = 1 - \beta_t$.
* **Anchor Demonstration:** Evaluating our targeted steps:

$$\alpha_1 = 1.0 - 0.000100 = 0.999900$$

$$\alpha_2 = 1.0 - 0.000120 = 0.999880$$

$$\alpha_{1000} = 1.0 - 0.020000 = 0.980000$$

<br>

#### Line 7: `alpha_bars = torch.cumprod(alphas, dim=0)`

* **Explanation:** Computes the cumulative product of the `alphas` tensor along its single axis, yielding $\bar{\alpha}_t$ for every step.
* **Mathematical Operation:** $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.
* **Anchor Demonstration:** Let's compute the cumulative entries for $t=1$ and $t=2$:

$$\bar{\alpha}_1 = \alpha_1 = 0.999900$$

$$\bar{\alpha}_2 = \alpha_1 \times \alpha_2 = 0.999900 \times 0.999880 = 0.999780$$

For $t=1000$, multiplying all $1000$ terms results in a heavily decayed value:

$$\bar{\alpha}_{1000} = \prod_{s=1}^{1000} \alpha_s \approx 0.040358$$

---

## Cell 8: Precomputing Alpha and Beta Noise Schedules

This cell visualizes the precise signal-to-noise ratio (SNR) transition across the discrete timeline of the DDPM framework. It tracks the magnitude of the coefficients scaling the ground-truth image and the added Gaussian noise vector, identifying the exact phase transition point where the corrupting noise dominates the residual structural signal.

<br>

### Mathematical Formulation

The closed-form marginal distribution of a noisy latent state $x_t$ given the initial clean state $x_0$ is expressed as:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

This formulation explicitly decomposes $x_t$ into two orthogonal geometric components:

* **The Signal Component:** Scaled by the monotonic decreasing function $s(t) = \sqrt{\bar{\alpha}_t}$. As $t \to T$, $s(t) \to 0$, compressing the data manifold toward the origin.
* **The Noise Component:** Scaled by the monotonic increasing function $\sigma(t) = \sqrt{1 - \bar{\alpha}_t}$. As $t \to T$, $\sigma(t) \to 1$, expanding the variance to fill the isotropic ambient space.

The **Signal-to-Noise Ratio (SNR)** at any given step is defined as the ratio of the squared coefficients:

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

The **crossover point** occurs at the exact time index $t^*$ where the signal power drops below the noise power, establishing the threshold $\text{SNR}(t^*) < 1$, which simplifies directly to:

$$\bar{\alpha}_{t^t} < 0.5 \implies \sqrt{\bar{\alpha}_{t^*}} < \sqrt{0.5} \approx 0.7071$$

<br>

### Code-to-Equation Breakdown

#### Line 3 & 4: `alpha_bars.sqrt().numpy()` and `(1 - alpha_bars).sqrt().numpy()`

* **Explanation:** These lines evaluate the element-wise square root of the cumulative product array and its complement to compute the exact coordinate trajectories of $s(t)$ and $\sigma(t)$.
* **Mathematical Operation:** Computes $s(t) = \sqrt{\bar{\alpha}_t}$ and $\sigma(t) = \sqrt{1 - \bar{\alpha}_t}$ for $t \in \{1, 2, \dots, 1000\}$.
* **Anchor Demonstration:** Let's calculate the explicit values using the precomputed parameters established in Cell 7 for steps $t=1, t=2$, and $t=1000$:

**At step $t=1$:**

$$s(1) = \sqrt{\bar{\alpha}_1} = \sqrt{0.999900} \approx 0.999950$$

$$\sigma(1) = \sqrt{1 - \bar{\alpha}_1} = \sqrt{1 - 0.999900} = \sqrt{0.000100} = 0.010000$$

**At step $t=2$:**

$$s(2) = \sqrt{\bar{\alpha}_2} = \sqrt{0.999780} \approx 0.999890$$

$$\sigma(2) = \sqrt{1 - \bar{\alpha}_2} = \sqrt{1 - 0.999780} = \sqrt{0.000220} \approx 0.014832$$

**At step $t=1000$:**

$$s(1000) = \sqrt{\bar{\alpha}_{1000}} = \sqrt{0.040358} \approx 0.200893$$

$$\sigma(1000) = \sqrt{1 - \bar{\alpha}_{1000}} = \sqrt{1 - 0.040358} = \sqrt{0.959642} \approx 0.979613$$

<br>

#### Line 12: `crossover = (alpha_bars < 0.5).nonzero(as_tuple=True)[0][0].item() + 1`

* **Explanation:** Finds the first index in the precomputed array where the cumulative signal scale drops below $0.5$, adjusting for zero-indexing by adding $1$.
* **Mathematical Operation:** Identifies $t^* = \min \{ t \in [1, T] \mid \bar{\alpha}_t < 0.5 \}$.
* **Anchor Demonstration:** For a linear schedule from $\beta_{\min}=10^{-4}$ to $\beta_{\max}=0.02$, this condition is typically met near the middle of the schedule (empirically around $t = 378$). At this point:

$$\bar{\alpha}_{378} \approx 0.499 \implies \sqrt{\bar{\alpha}_{378}} \approx 0.7064 \quad \text{and} \quad \sqrt{1 - \bar{\alpha}_{378}} \approx 0.7078$$

Because $0.7064 < 0.7078$, the noise coefficient officially overtakes the signal coefficient.

---

## Cell 9: Uniform Random Step Selection for Forward Discretization

This cell implements the closed-form forward sampling formula for a Denoising Diffusion Probabilistic Model (DDPM). Instead of iteratively applying noise step-by-step, it exploits the mathematical properties of compounding Gaussian distributions to jump directly from the clean initial image $x_0$ to any arbitrary noisy intermediate state $x_t$ at step $t$.

<br>

### Mathematical Formulation

The forward marginal distribution $q(x_t \mid x_0)$ is derived by unrolling the Markov chain $q(x_s \mid x_{s-1}) = \mathcal{N}(x_s; \sqrt{1 - \beta_s}x_{s-1}, \beta_s \mathbf{I})$ from $s=1$ to $t$. Let $\alpha_s = 1 - \beta_s$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

Step 1: Express $x_t$ as a function of $x_{t-1}$:

$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1}, \quad \text{where } \epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})$$

Step 2: Express $x_{t-1}$ as a function of $x_{t-2}$:

$$x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon_{t-2}, \quad \text{where } \epsilon_{t-2} \sim \mathcal{N}(0, \mathbf{I})$$

Step 3: Substitute $x_{t-1}$ into the equation for $x_t$:

$$x_t = \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon_{t-2}\right) + \sqrt{1 - \alpha_t}\epsilon_{t-1}$$

$$x_t = \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t \alpha_{t-1}}\epsilon_{t-2} + \sqrt{1 - \alpha_t}\epsilon_{t-1}$$

Step 4: Combine the independent Gaussian noise terms. The sum of two independent Gaussians $\mathcal{N}(0, \sigma_1^2 \mathbf{I})$ and $\mathcal{N}(0, \sigma_2^2 \mathbf{I})$ is a single Gaussian distribution $\mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. Summing their variances yields:

$$\sigma_{\text{combined}}^2 = \left(\alpha_t(1 - \alpha_{t-1})\right) + (1 - \alpha_t) = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \alpha_{t-1}$$

Thus, the substitution simplifies to:

$$x_t = \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\epsilon', \quad \text{where } \epsilon' \sim \mathcal{N}(0, \mathbf{I})$$

Step 5: Inductively repeat this expansion back to $x_0$:

$$x_t = \sqrt{\prod_{s=1}^t \alpha_s}x_0 + \sqrt{1 - \prod_{s=1}^t \alpha_s}\epsilon = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

<br>

### Code-to-Equation Breakdown

#### Line 16: `ab = alpha_bars[t].view(-1, 1, 1, 1)`

* **Explanation:** Retrieves the cumulative variance product $\bar{\alpha}_t$ for each sample in the batch based on its assigned time index $t$, reshaping it from a 1D tensor of size `(B,)` into a 4D tensor of size `(B, 1, 1, 1)` to allow broadcasting across channels and spatial dimensions.
* **Mathematical Operation:** Maps the batch vector $\mathbf{t} = [t_0, t_1, \dots, t_{B-1}]^T$ to the scalar coefficient tensor $\bar{\alpha}_{\mathbf{t}}$.
* **Anchor Demonstration:** To evaluate this line on our anchor example, we choose an arbitrary heterogeneous time batch where Sample 0 is at time step $t=0$ and Sample 1 is at time step $t=1$. Using our precomputed schedule values from Cell 7 ($\bar{\alpha}_1 = 0.999900$ and $\bar{\alpha}_2 = 0.999780$), and noting that python 0-indexing maps $t=0 \implies \bar{\alpha}_1$ and $t=1 \implies \bar{\alpha}_2$:

$$\text{ab}[0, 0, 0, 0] = \bar{\alpha}_1 = 0.999900$$

$$\text{ab}[1, 0, 0, 0] = \bar{\alpha}_2 = 0.999780$$

<br>

#### Line 17: `eps = torch.randn_like(x0)`

* **Explanation:** Generates a random noise tensor matching the shape of $x_0$, filled with elements sampled independently from a standard normal distribution.
* **Mathematical Operation:** $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.
* **Anchor Demonstration:** Per your explicit setup guidelines for the anchor example, this random operation is locked to a static, reproducible tensor where every single element is $+0.5$.

$$\text{For all } n \in \{0, 1\}, c \in \{0, 1, 2\}, i \in \{1..4\}, j \in \{1..4\}: \quad \text{eps}[n, c, i, j] = +0.5$$

<br>

#### Line 18: `x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps`

* **Explanation:** Computes the element-wise interpolation between the clean image tensor and the static noise tensor using the square roots of the signal weight and variance weight.
* **Mathematical Operation:** $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$.
* **Anchor Demonstration:** We calculate the exact numerical values for every element of the resulting tensor $x_t$ for each sample across all channels $c$.

<br>

##### Sample 0 (The Checkerboard at $t=0 \implies \bar{\alpha}_1 = 0.999900$)

The scalar multipliers are:

$$\sqrt{\bar{\alpha}_1} = \sqrt{0.999900} \approx 0.999950$$

$$\sqrt{1 - \bar{\alpha}_1} = \sqrt{1 - 0.999900} = \sqrt{0.000100} = 0.010000$$

Every element where $x_0 = +1.0$:

$$x_t[0, c, i, j] = (0.999950 \times 1.0) + (0.010000 \times 0.5) = 0.999950 + 0.005000 = 1.004950$$

Every element where $x_0 = -1.0$:

$$x_t[0, c, i, j] = (0.999950 \times -1.0) + (0.010000 \times 0.5) = -0.999950 + 0.005000 = -0.994950$$

The spatial layout of $x_t[0, c]$ for all three channels is:

$$x_t[0, c] = \begin{bmatrix} 1.00495 & -0.99495 & 1.00495 & -0.99495 \\ -0.99495 & 1.00495 & -0.99495 & 1.00495 \\ 1.00495 & -0.99495 & 1.00495 & -0.99495 \\ -0.99495 & 1.00495 & -0.99495 & 1.00495 \end{bmatrix}$$

<br>

##### Sample 1 (The Spatial Gradient at $t=1 \implies \bar{\alpha}_2 = 0.999780$)

The scalar multipliers are:

$$\sqrt{\bar{\alpha}_2} = \sqrt{0.999780} \approx 0.999890$$

$$\sqrt{1 - \bar{\alpha}_2} = \sqrt{1 - 0.999780} = \sqrt{0.000220} \approx 0.014832$$

Every row in this sample consists of four unique entries for $x_0$: $\begin{bmatrix}-1.0 & -0.33 & +0.33 & +1.0\end{bmatrix}$. We transform each entry individually:

**1. For $x_0 = -1.0$ (Column 0):**

$$x_t[1, c, i, 0] = (0.999890 \times -1.0) + (0.014832 \times 0.5) = -0.999890 + 0.007416 = -0.992474$$

**2. For $x_0 = -0.33$ (Column 1):**

$$x_t[1, c, i, 1] = (0.999890 \times -0.33) + (0.014832 \times 0.5) = -0.329964 + 0.007416 = -0.322548$$

**3. For $x_0 = +0.33$ (Column 2):**

$$x_t[1, c, i, 2] = (0.999890 \times 0.33) + (0.014832 \times 0.5) = +0.329964 + 0.007416 = +0.337380$$

**4. For $x_0 = +1.0$ (Column 3):**

$$x_t[1, c, i, 3] = (0.999890 \times 1.0) + (0.014832 \times 0.5) = 0.999890 + 0.007416 = 1.007306$$

The spatial layout of $x_t[1, c]$ for all three channels is:

$$x_t[1, c] = \begin{bmatrix} -0.992474 & -0.322548 & +0.337380 & 1.007306 \\ -0.992474 & -0.322548 & +0.337380 & 1.007306 \\ -0.992474 & -0.322548 & +0.337380 & 1.007306 \\ -0.992474 & -0.322548 & +0.337380 & 1.007306 \end{bmatrix}$$

---

## Cell 10: Closed-Form Latent State Sampling (Forward Diffusion Pass)

This cell provides a qualitative verification of the forward diffusion process by tracking the spatial degradation of a structural signal across specific checkpoints in the discrete timeline. As time steps advance, the mathematical operations systematically suppress the variance of the data manifold while inflating an independent isotropic Gaussian distribution, demonstrating the continuous transformation of structured pixel information into high-entropy noise.

<br>

### Mathematical Formulation

Let $x_0 \in \mathbb{R}^{C \times H \times W}$ represent the clean data tensor. For any selected discrete time step $t_k \in \{0, 100, 300, 500, 700, 999\}$, the latent state $x_{t_k}$ is simulated using the marginal distribution relation verified in Cell 9:

$$x_{t_k} = \sqrt{\bar{\alpha}_{t_k}}x_0 + \sqrt{1 - \bar{\alpha}_{t_k}}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

To visualize this continuous floating-point state using standard graphic libraries, the values must be mapped from the model's operational domain $[-1, 1]$ back to the standard visualization domain $[0, 1]$. This is achieved using an inverse affine transformation $g: [-1, 1] \to [0, 1]$:

$$g(x_{t_k}) = \frac{x_{t_k} + 1.0}{2.0}$$

Because adding unconstrained normal noise can push pixel coordinates outside the formal boundaries of the display format, a non-linear clipping operator $\text{clip}(v, v_{\min}, v_{\max})$ is applied element-wise:

$$\bar{x}_{t_k} = \max\left(0.0, \min\left(1.0, g(x_{t_k})\right)\right)$$

<br>

### Code-to-Equation Breakdown

#### Line 3 & 4: Data Preparation

```python
sample_np = raw.data[42].astype(np.float32) / 127.5 - 1.0
x0 = torch.tensor(sample_np).permute(2, 0, 1).unsqueeze(0)
```

* **Explanation:** These lines extract an image from the dataset, normalize it to $[-1, 1]$, change its memory layout from channels-last `(H, W, C)` to channels-first `(C, H, W)`, and add a batch dimension to create a valid 4D input tensor of shape `(1, 3, 32, 32)`.
* **Mathematical Operation:** $x_0 \in \mathbb{R}^{1 \times 3 \times 32 \times 32}$.
* **Anchor Demonstration:** To remain perfectly continuous and aligned with your instructions, we bypass the arbitrary index 42 from the dataset and continue tracing our established anchor tensor $x_0$ consisting of Sample 0 (Checkerboard) and Sample 1 (Spatial Gradient).

<br>

#### Line 11: `x_t, _ = q_sample(x0, t_tensor, alpha_bars)`

* **Explanation:** Computes the noisy state tensor $x_t$ at the given time index.
* **Mathematical Operation:** Simulates $x_{t_k} \sim q(x_{t_k} \mid x_0)$.
* **Anchor Demonstration:** Let's observe the mathematical transformation at a mid-range checkpoint, $t = 500$. We pull the precomputed value from our schedule: $\bar{\alpha}_{500} \approx 0.3621$.
The scalar multipliers are:

$$\sqrt{\bar{\alpha}_{500}} = \sqrt{0.3621} \approx 0.6017$$

$$\sqrt{1 - \bar{\alpha}_{500}} = \sqrt{1 - 0.3621} = \sqrt{0.6379} \approx 0.7987$$

Let's compute the forward step at this time index for specific pixels using our static anchor noise $\epsilon = +0.5$:

**For Sample 0 where $x_0 = +1.0$:**

$$x_{500} = (0.6017 \times 1.0) + (0.7987 \times 0.5) = 0.6017 + 0.3994 = 1.0011$$

**For Sample 0 where $x_0 = -1.0$:**

$$x_{500} = (0.6017 \times -1.0) + (0.7987 \times 0.5) = -0.6017 + 0.3994 = -0.2023$$

**For Sample 1 where $x_0 = -0.33$ (Column 1):**

$$x_{500} = (0.6017 \times -0.33) + (0.7987 \times 0.5) = -0.1986 + 0.3994 = 0.2008$$

<br>

#### Line 13 & 14: `img = (img + 1) / 2` and `img = np.clip(img, 0, 1)`

* **Explanation:** These lines shift and scale the noisy tensor back into a $[0, 1]$ range and clip any outlying values to guarantee compatibility with the image plotting function.
* **Mathematical Operation:** Evaluation of $\bar{x}_{t_k} = \min(1.0, \max(0.0, \frac{x_{t_k} + 1.0}{2.0}))$.
* **Anchor Demonstration:** Let's pass our computed $t=500$ intermediate anchor values through this visualization pipeline:

**Sample 0 positive pixel ($1.0011$):**

$$\bar{x} = \frac{1.0011 + 1.0}{2.0} = \frac{2.0011}{2.0} = 1.00055 \xrightarrow{\text{clip}} 1.0000$$

**Sample 0 negative pixel ($-0.2023$):**

$$\bar{x} = \frac{-0.2023 + 1.0}{2.0} = \frac{0.7977}{2.0} = 0.39885 \xrightarrow{\text{clip}} 0.3989$$

**Sample 1 intermediate pixel ($0.2008$):**

$$\bar{x} = \frac{0.2008 + 1.0}{2.0} = \frac{1.2008}{2.0} = 0.60040 \xrightarrow{\text{clip}} 0.6004$$

---

## Cell 11: Plotting Forward Diffusion Image Degradation

This cell provides quantitative validation that at the final discrete time step $t = T - 1$, the forward process successfully destroys all structural information. It demonstrates that the target distribution converges to an isotropic Gaussian noise distribution, ensuring that the model learns to invert a clean map starting from a known, tractable noise distribution.

<br>

### Mathematical Formulation

The exact distribution of the final latent state $x_{T-1}$ given the clean image $x_0$ is defined by:

$$x_{T-1} = \sqrt{\bar{\alpha}_{T-1}}x_0 + \sqrt{1 - \bar{\alpha}_{T-1}}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

For the framework to be mathematically sound, two conditions must be satisfied as $T \to \infty$:

**1. Signal Erasure:** The information coefficient must vanish:

$$\lim_{T \to \infty} \sqrt{\bar{\alpha}_{T-1}} \approx 0 \implies \sqrt{\bar{\alpha}_{T-1}}x_0 \to \mathbf{0}$$

**2. Noise Domination:** The variance coefficient must approach unity:

$$\lim_{T \to \infty} \sqrt{1 - \bar{\alpha}_{T-1}} \approx 1 \implies 1 - \bar{\alpha}_{T-1} \to 1$$

When these conditions are met, the conditional marginal distribution matches the standard normal distribution:

$$q(x_{T-1} \mid x_0) \approx \mathcal{N}(0, \mathbf{I})$$

<br>

### Code-to-Equation Breakdown

#### Line 2 & 3: `t_T = torch.tensor([T - 1])` and `x_T_samples = ...`

* **Explanation:** These lines sample 1,000 independent realization tensors at the final time index ($t=999$) to construct an empirical distribution for statistical analysis.
* **Mathematical Operation:** Generates an empirical ensemble of states governed by $q(x_{999} \mid x_0)$.
* **Anchor Demonstration:** Let's calculate the exact analytic parameters for this step using our precomputed schedule value from Cell 7: $\bar{\alpha}_{999} = 0.040358$.

The exact signal scaling factor is:

$$\sqrt{\bar{\alpha}_{999}} = \sqrt{0.040358} = 0.200893$$

The exact noise scaling factor is:

$$\sqrt{1 - \bar{\alpha}_{999}} = \sqrt{1 - 0.040358} = \sqrt{0.959642} = 0.979613$$

Using our static anchor noise ($\epsilon = +0.5$), we compute the final forward state values ($x_{999}$) element-by-element:

**For Sample 0 where $x_0 = +1.0$:**

$$x_{999} = (0.200893 \times 1.0) + (0.979613 \times 0.5) = 0.200893 + 0.489807 = 0.690700$$

**For Sample 0 where $x_0 = -1.0$:**

$$x_{999} = (0.200893 \times -1.0) + (0.979613 \times 0.5) = -0.200893 + 0.489807 = 0.288914$$

**For Sample 1 where $x_0 = -1.0$ (Column 0):**

$$x_{999} = (0.200893 \times -1.0) + (0.979613 \times 0.5) = 0.288914$$

**For Sample 1 where $x_0 = +1.0$ (Column 3):**

$$x_{999} = (0.200893 \times 1.0) + (0.979613 \times 0.5) = 0.690700$$

<br>

#### Line 8: `max |signal contrib| = (alpha_bars[-1].sqrt() * x0.abs().max()).item()`

* **Explanation:** Computes the maximum upper bound of residual signal remaining from the clean image after running the forward diffusion process to completion.
* **Mathematical Operation:** Evaluates $||\sqrt{\bar{\alpha}_{999}} \cdot \max(|x_0|)||_{\infty}$.
* **Anchor Demonstration:** For both Sample 0 and Sample 1, the absolute maximum value of any pixel entry in the clean tensor is exactly $1.0$.

$$\text{max } |x_0| = 1.0$$

Multiplying this value by our signal scale factor yields:

$$\text{Signal Contribution} = 0.200893 \times 1.0 = 0.200893$$

> **Why this result matters:** In a true production setup with infinite time steps, this value approaches zero. In this specific discrete schedule setup ($T=1000$), a residual signal contribution of $0.200893$ indicates that around 20% of the original coordinate scale persists. This structural leakage is an inherent limitation of traditional discrete DDPM schedules, which motivates the transition to the Rectified Flow (Flow Matching) framework later in the pipeline.

---

## Cell 12: Empirical Mean Squared Error Denoising Loss Function

This cell implements the training objective for the Denoising Diffusion Probabilistic Model (DDPM): the Denoising Score Matching loss. This function optimizes the neural network to approximate the score function of the data distribution by training it to predict the exact noise vector added to a clean image at any given time step.

<br>

### Mathematical Formulation

The standard training objective minimizes the variational bound on the negative log-likelihood. Under a reparameterization where the network explicitly targets the added noise vector, this simplifies to a weighted Mean Squared Error (MSE) objective:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon_\theta(x_t, t) - \epsilon \|^2 \right]$$

Where:

* $t \sim \mathcal{U}(\{0, 1, \dots, T-1\})$ represents a time index sampled uniformly across the discrete schedule.
* $x_0 \sim q(x_0)$ is a mini-batch of clean data images.
* $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is the target standard Gaussian noise tensor.
* $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ is the composite noisy latent state.
* $\epsilon_\theta(x_t, t)$ is the neural network prediction of the noise vector.

The Mean Squared Error operator computes the unweighted average of the squared differences across all tensor elements:

$$\text{MSE}(\hat{Y}, Y) = \frac{1}{B \cdot C \cdot H \cdot W} \sum_{b=1}^{B} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} (\hat{Y}_{b,c,i,j} - Y_{b,c,i,j})^2$$

<br>

### Code-to-Equation Breakdown

#### Line 15: `B = x0.shape[0]`

* **Explanation:** Extracts the mini-batch size dimension from the input data tensor.
* **Mathematical Operation:** Evaluates the cardinality variable $B$.
* **Anchor Demonstration:** For our anchor setup, $B = 2$.

<br>

#### Line 16: `t = torch.randint(0, len(alpha_bars), (B,), device=x0.device)`

* **Explanation:** Samples a random integer uniformly between 0 and $T-1$ for each sample in the mini-batch.
* **Mathematical Operation:** Simulates $t_b \sim \mathcal{U}(\{0, 1, \dots, 999\})$ for $b \in \{0, 1\}$.
* **Anchor Demonstration:** To remain consistent with our previous cumulative calculations from Cell 9, we select the time indices:

$$\mathbf{t} = [0, 1]^T$$

<br>

#### Line 17: `x_t, eps = q_sample(x0, t, alpha_bars.to(x0.device))`

* **Explanation:** Passes the clean image batch and the sampled time steps to the forward process function to compute the noisy state and the target noise tensor.
* **Mathematical Operation:** Evaluation of $q(x_t \mid x_0)$ and extraction of $\epsilon$.
* **Anchor Demonstration:** These exact outputs were derived element-by-element in Cell 9.
	* The target noise tensor `eps` contains $+0.5$ at every position.
	* The intermediate noisy state tensor `x_t` has the exact spatial configurations calculated in Cell 9:

$$x_t[0, c] = \begin{bmatrix} 1.00495 & -0.99495 & 1.00495 & -0.99495 \\ -0.99495 & 1.00495 & -0.99495 & 1.00495 \\ 1.00495 & -0.99495 & 1.00495 & -0.99495 \\ -0.99495 & 1.00495 & -0.99495 & 1.00495 \end{bmatrix}$$

$$x_t[1, c] = \begin{bmatrix} -0.992474 & -0.322548 & +0.337380 & 1.007306 \\ -0.992474 & -0.322548 & +0.337380 & 1.007306 \\ -0.992474 & -0.322548 & +0.337380 & 1.007306 \\ -0.992474 & -0.322548 & +0.337380 & 1.007306 \end{bmatrix}$$

<br>

#### Line 18: `eps_pred = model(x_t, t)`

* **Explanation:** Evaluates the score network on the noisy data matrix and time steps to get the predicted noise tensor.
* **Mathematical Operation:** Evaluates $\epsilon_\theta(x_t, t)$.
* **Anchor Demonstration:** Since the network is not yet defined, we assume a mock initial state where the model outputs a uniform zero prediction tensor for all elements:

$$\text{eps\_pred}[n, c, i, j] = 0.0$$

<br>

#### Line 19: `return nn.functional.mse_loss(eps_pred, eps)`

* **Explanation:** Computes the average squared deviation between the predicted noise tensor and the ground-truth target noise tensor.
* **Mathematical Operation:** Evaluates the loss scalar $\mathcal{L}_{\text{simple}}(\theta)$.
* **Anchor Demonstration:** Every single element of our ground truth noise tensor `eps` equals $+0.5$, and every element of our mock prediction `eps_pred` equals $0.0$.
The squared difference for any individual element is:

$$(\hat{Y} - Y)^2 = (0.0 - 0.5)^2 = (-0.5)^2 = 0.25$$

Because every element across both samples, all 3 channels, and all 16 spatial positions has an identical squared deviation of $0.25$, the global average is:

$$\mathcal{L}_{\text{simple}} = \frac{1}{2 \times 3 \times 4 \times 4} \sum_{b=0}^{1}\sum_{c=0}^{2}\sum_{i=1}^{4}\sum_{j=1}^{4} 0.25 = \frac{92 \times 0.25}{92} = 0.25$$

---

## Cell 13: Sinusoidal Positional Embeddings for Continuous Timesteps

This cell provides a rigorous mathematical bridge between Tweedie’s formula and the reparameterized diffusion objective. It numerically validates that predicting the added noise vector $\epsilon$ is mathematically equivalent to predicting the true score function $\nabla_{x_t} \log q(x_t \mid x_0)$ of the diffused data distribution, scaled by a time-dependent variance factor.

<br>

### Mathematical Formulation

The score function is defined as the gradient of the log probability density function with respect to the latent state variables:

$$\text{Score} \equiv \nabla_{x_t} \log q(x_t \mid x_0)$$

Because the forward perturbation kernel $q(x_t \mid x_0)$ is a standard isotropic Gaussian distribution $\mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$, its probability density function is written explicitly as:

$$\left. q(x_t \mid x_0) = \left(2\pi(1 - \bar{\alpha}_t)\right)^{-\frac{d}{2}} \exp\left( -\frac{\|x_t - \sqrt{\bar{\alpha}_t}x_0\|^2}{2(1 - \bar{\alpha}_t)} \right) \right.$$

Taking the natural logarithm of this density function eliminates the exponential operator:

$$\log q(x_t \mid x_0) = -\frac{d}{2}\log\left(2\pi(1 - \bar{\alpha}_t)\right) - \frac{\|x_t - \sqrt{\bar{\alpha}_t}x_0\|^2}{2(1 - \bar{\alpha}_t)}$$

Differentiating this scalar scalar field with respect to the vector $x_t$ using the chain rule ($\nabla_v \|v\|^2 = 2v$) gives the analytic score:

$$\nabla_{x_t} \log q(x_t \mid x_0) = \mathbf{0} - \frac{2(x_t - \sqrt{\bar{\alpha}_t}x_0)}{2(1 - \bar{\alpha}_t)} = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1 - \bar{\alpha}_t}$$

By substituting the forward reparameterization formula $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ into this expression, we rewrite the numerator in terms of the noise vector $\epsilon$:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\left(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon\right) - \sqrt{\bar{\alpha}_t}x_0}{1 - \bar{\alpha}_t} = -\frac{\sqrt{1 - \bar{\alpha}_t}\epsilon}{1 - \bar{\alpha}_t}$$

Simplifying the algebraic fraction yields the exact relation mapping the noise space to the score space:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

<br>

### Code-to-Equation Breakdown

#### Line 5: `x_t, eps = q_sample(x0, t_val, alpha_bars)`

* **Explanation:** Samples an intermediate noisy state using the time index $t=499$.
* **Mathematical Operation:** Generates a realization from $q(x_{500} \mid x_0)$.
* **Anchor Demonstration:** To preserve cumulative state tracking, we pull the exact $t=500$ values calculated in Cell 10 for our anchor tensor ($\bar{\alpha}_{500} \approx 0.3621$, $\sqrt{\bar{\alpha}_{500}} \approx 0.6017$, $\sqrt{1 - \bar{\alpha}_{500}} \approx 0.7987$, and $\epsilon = +0.5$).
	* For Sample 0 when $x_0 = +1.0 \implies x_t = 1.0011$
	* For Sample 0 when $x_0 = -1.0 \implies x_t = -0.2023$
	* For Sample 1 when $x_0 = -1.0 \implies x_t = -0.2023$
	* For Sample 1 when $x_0 = +1.0 \implies x_t = 1.0011$

<br>

#### Line 8: `score_analytic = -(x_t - ab.sqrt() * x0) / (1 - ab)`

* **Explanation:** Computes the score values directly from the noisy latent coordinates and clean images.
* **Mathematical Operation:** Evaluation of $-\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1 - \bar{\alpha}_t}$.
* **Anchor Demonstration:** Let's calculate the value for Sample 0 where $x_0 = +1.0$ and $x_t = 1.0011$:

$$\text{Numerator} = -\left(1.0011 - (0.6017 \times 1.0)\right) = -(1.0011 - 0.6017) = -0.3994$$

$$\text{Denominator} = 1 - 0.3621 = 0.6379$$

$$\text{score\_analytic} = \frac{-0.3994}{0.6379} \approx -0.626117$$

Let's calculate the value for Sample 0 where $x_0 = -1.0$ and $x_t = -0.2023$:

$$\text{Numerator} = -\left(-0.2023 - (0.6017 \times -1.0)\right) = -(-0.2023 + 0.6017) = -0.3994$$

$$\text{Denominator} = 0.6379$$

$$\text{score\_analytic} = \frac{-0.3994}{0.6379} \approx -0.626117$$

<br>

#### Line 9: `score_from_noise = -eps / (1 - ab).sqrt()`

* **Explanation:** Evaluates the score vector using the reparameterized noise mapping formula.
* **Mathematical Operation:** Evaluation of $-\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$.
* **Anchor Demonstration:** Every element in our static noise tensor $\epsilon$ equals $+0.5$. The denominator is $\sqrt{1 - 0.3621} = \sqrt{0.6379} \approx 0.7987$.

$$\text{score\_from\_noise} = \frac{-0.5}{0.7987} \approx -0.626117$$

Comparing both evaluations for every element across the entire tensor yields:

$$\text{score\_analytic} - \text{score\_from\_noise} = -0.626117 - (-0.626117) = 0.0$$

Thus, `max_diff` equals exactly $0.0$, proving numerical and algebraic equivalence.

---

## Cell 14: Downsampling Residual Blocks with Convolutional Skip Connections

This cell evaluates and visualizes the magnitude of the score scaling factor across the discrete timeline of the DDPM framework. It tracks the magnitude of the coefficient that maps the predicted noise vector to the analytic score function. This scaling behavior highlights why standard DDPM targets noise rather than directly modeling the raw score function, which becomes singular near $t=0$.

<br>

### Mathematical Formulation

As proven in Cell 13, the relationship between the true score function and the added noise vector is governed by a time-dependent variance coefficient:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

The magnitude of this vector field is governed by the scalar function $S(t)$:

$$S(t) = \frac{1}{\sqrt{1 - \bar{\alpha}_t}}$$

We analyze the limits of this scaling factor at the boundaries of our timeline:

* **As $t \to 0$ (Initial Step):** $\bar{\alpha}_t \to 1$. Consequently, the denominator approaches zero:

$$\lim_{t \to 0} \sqrt{1 - \bar{\alpha}_t} = 0 \implies \lim_{t \to 0} S(t) = +\infty$$

* **As $t \to T$ (Final Step):** $\bar{\alpha}_t \to 0$. Consequently, the denominator approaches unity:

$$\lim_{t \to T} \sqrt{1 - \bar{\alpha}_t} = 1 \implies \lim_{t \to T} S(t) = 1$$


> **Why this result matters:** The singularity at $t=0$ creates severe numerical instability if a neural network is optimized to predict the raw score directly. By reparameterizing the network to target the unscaled noise vector $\epsilon$ instead, we isolate this variance instability, enabling smooth and bounded optimization throughout training.

<br>

### Code-to-Equation Breakdown

#### Line 2: `scale = 1.0 / (1 - alpha_bars).sqrt()`

* **Explanation:** Performs an element-wise evaluation of the reciprocal of the noise standard deviation across all $T=1000$ steps.
* **Mathematical Operation:** Evaluation of $S(t) = (1 - \bar{\alpha}_t)^{-\frac{1}{2}}$.
* **Anchor Demonstration:** Let's calculate the explicit scalar values using the precomputed parameters established in Cell 7 for steps $t=0$, $t=499$, and $t=999$ (corresponding to $t=1$, $t=500$, and $t=1000$ in 1-indexed notation):

**At step $t=0$ ($t=1$ in schedule):**

$$\bar{\alpha}_1 = 0.999900 \implies 1 - \bar{\alpha}_1 = 0.000100$$

$$S(1) = \frac{1}{\sqrt{0.000100}} = \frac{1}{0.010000} = 100.0000$$

**At step $t=499$ ($t=500$ in schedule):**

$$\bar{\alpha}_{500} \approx 0.3621 \implies 1 - \bar{\alpha}_{500} = 0.6379$$

$$S(500) = \frac{1}{\sqrt{0.6379}} \approx \frac{1}{0.7987} \approx 1.2520$$

**At step $t=999$ ($t=1000$ in schedule):**

$$\bar{\alpha}_{1000} = 0.040358 \implies 1 - \bar{\alpha}_{1000} = 0.959642$$

$$S(1000) = \frac{1}{\sqrt{0.959642}} \approx \frac{1}{0.979613} \approx 1.0208$$

The numerical values show that the scale factor drops by two orders of magnitude across the schedule, decreasing from $100.0$ down to approximately $1.02$.

---

## Cell 15: Upsampling Residual Blocks with Nearest Neighbor Interpolation

This cell implements a Sinusoidal Timestep Embedding module. Because neural network weights are static, a standard neural architecture cannot easily process inputs that change characteristics over time unless it is explicitly informed of the current time step. This module maps a discrete scalar time coordinate $t$ into a high-dimensional geometric coordinate space. This mapping uses transcendental functions of varying spatial frequencies, providing the network with a unique, continuous, and smooth representation of each noise level.

<br>

### Mathematical Formulation

Let $t \in \mathbb{R}$ represent a scalar time step, and let $D$ be the target embedding dimension, where $D$ is an even integer ($D \pmod 2 = 0$). We define $d = \frac{D}{2}$ as the dimensionality of each half-component.

The positional encoding function $\Phi: \mathbb{R} \to \mathbb{R}^D$ maps the scalar $t$ to a vector of interleaved or concatenated harmonic components:

$$\Phi(t) = \left[ \sin(t \cdot \omega_0), \sin(t \cdot \omega_1), \dots, \sin(t \cdot \omega_{d-1}), \cos(t \cdot \omega_0), \cos(t \cdot \omega_1), \dots, \cos(t \cdot \omega_{d-1}) \right]^T$$

The frequency values $\omega_i$ are distributed geometrically over a logarithmic scale bounded between a minimum and maximum wavelength:

$$\omega_i = 10000^{-\frac{i}{d-1}}$$

To compute these frequencies in a numerically stable manner without repeated division or exponentiation of small bases, we rewrite the geometric progression using the identity $a^b = \exp(b \ln a)$:

$$\omega_i = \exp\left( \ln\left(10000^{-\frac{i}{d-1}}\right) \right) = \exp\left( -\frac{i}{d-1} \ln(10000) \right)$$

By altering the sign or rearranging the index direction, the code implements an equivalent frequency progression:

$$\omega_i = \exp\left( \frac{i}{d-1} \ln(10000) \right)$$

When multiplied by $t$, this generates the scaling arguments $\theta_{b, i}$ for each batch item $b$ and frequency index $i$:

$$\theta_{b, i} = t_b \cdot \omega_i$$

<br>

### Code-to-Equation Breakdown

#### Line 13: `half = self.dim // 2`

* **Explanation:** Divides the target embedding dimension by two to allocate half the capacity to sine operations and half to cosine operations.
* **Mathematical Operation:** Computes $d = \frac{D}{2}$.
* **Anchor Demonstration:** For this verification step, we initialize an embedding layer with `dim=4` ($D=4$), which yields $d = 2$.

<br>

#### Line 14: `freqs = torch.exp(math.log(10000) * torch.arange(half, device=t.device) / (half - 1))`

* **Explanation:** Generates a 1D tensor of frequency weights using a log-linear scale.
* **Mathematical Operation:** Evaluates $\omega_i = \exp\left( \frac{i}{d-1} \ln(10000) \right)$ for $i \in \{0, 1, \dots, d-1\}$.
* **Anchor Demonstration:** With $d=2$, our index array `torch.arange(2)` contains elements $[0, 1]^T$. The denominator term is $half - 1 = 2 - 1 = 1$. Let's compute each element of the `freqs` tensor by hand:

$$\omega_0 = \exp\left( \ln(10000) \times \frac{0}{1} \right) = \exp(0) = 1.0000$$

$$\omega_1 = \exp\left( \ln(10000) \times \frac{1}{1} \right) = \exp(\ln(10000)) = 10000.0000$$

<br>

Thus, the precomputed frequency vector is:

$$\mathbf{\omega} = [1.0, 10000.0]^T$$

<br>

#### Line 15: `args = t[:, None].float() * freqs[None, :]`

* **Explanation:** Reshapes the 1D batch tensor `t` to shape `(B, 1)` and the 1D frequency tensor `freqs` to shape `(1, half)`. It then performs an outer-product multiplication using broadcasting to generate a 2D matrix of shape `(B, half)`.
* **Mathematical Operation:** Evaluation of $\theta_{b, i} = t_b \cdot \omega_i$.
* **Anchor Demonstration:** We trace our established anchor time batch containing $N=2$ samples at time steps $\mathbf{t} = [0, 1]^T$. We compute the product matrix element-by-element:

**For Batch Row 0 ($t_0 = 0$):**

$$\theta_{0, 0} = t_0 \times \omega_0 = 0 \times 1.0 = 0.0$$

$$\theta_{0, 1} = t_0 \times \omega_1 = 0 \times 10000.0 = 0.0$$

**For Batch Row 1 ($t_1 = 1$):**

$$\theta_{1, 0} = t_1 \times \omega_0 = 1 \times 1.0 = 1.0$$

$$\theta_{1, 1} = t_1 \times \omega_1 = 1 \times 10000.0 = 10000.0$$

<br>

**This yields the argument tensor:**

$$\mathbf{\Theta} = \begin{bmatrix} 0.0 & 0.0 \\ 1.0 & 10000.0 \end{bmatrix}$$

<br>

#### Line 16: `return torch.cat([args.sin(), args.cos()], dim=-1)`

* **Explanation:** Computes the element-wise sine and cosine transformations of the argument matrix, concatenating them along the final channel axis to output a final embedding matrix of shape `(B, dim)`.
* **Mathematical Operation:** Evaluation of $\Phi(t) = [\sin(\mathbf{\Theta}), \cos(\mathbf{\Theta})]$.
* **Anchor Demonstration:** We evaluate the trigonometric functions for each row of our argument matrix $\mathbf{\Theta}$:

**For Batch Row 0 ($t=0$):**

$$\sin(\theta_{0,0}) = \sin(0.0) = 0.0000$$

$$\sin(\theta_{0,1}) = \sin(0.0) = 0.0000$$

$$\cos(\theta_{0,0}) = \cos(0.0) = 1.0000$$

$$\cos(\theta_{0,1}) = \cos(0.0) = 1.0000$$

Concatenating these elements gives the first row vector:

$$\Phi(0) = [0.0, 0.0, 1.0, 1.0]$$

**For Batch Row 1 ($t=1$):**

$$\sin(\theta_{1,0}) = \sin(1.0) \approx 0.8415$$

$$\sin(\theta_{1,1}) = \sin(10000.0) \approx -0.3056$$

$$\cos(\theta_{1,0}) = \cos(1.0) \approx 0.5403$$

$$\cos(\theta_{1,1}) = \cos(10000.0) \approx -0.9522$$

Concatenating these elements gives the second row vector:

$$\Phi(1) = [0.8415, -0.3056, 0.5403, -0.9522]$$

<br>

**Combining both rows yields the final embedding tensor output:**

$$\text{Output Tensor} = \begin{bmatrix} 0.0000 & 0.0000 & 1.0000 & 1.0000 \\ 0.8415 & -0.3056 & 0.5403 & -0.9522 \end{bmatrix}$$

---

## Cell 16: Time-Conditioned U-Net Encoder-Decoder Architecture

This cell implements the main computational unit of the time-conditioned score network: a Residual Block (`ResBlock`). This layer extracts multi-scale spatial features from the latent tensor while simultaneously incorporating the high-dimensional time embedding vector via additive broadcasting across space.

<br>

### Mathematical Formulation

Let $x \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ represent the input activation tensor and $e_t \in \mathbb{R}^{D}$ denote the continuous time embedding vector. The forward pass maps the inputs through a sequence of convolutions, normalization layers, and non-linear activations:

**1. First Spatial Projection & Normalization:**

$$h_1 = \text{SiLU}\left( \text{GroupNorm}\left( \mathbf{W}_1 * x + b_1 \right) \right)$$

Where $*$ represents a standard 2D spatial convolution using a $3 \times 3$ kernel, and $\text{GroupNorm}$ partitions the channels into 8 distinct blocks, normalizing the mean and variance across each group's spatial and channel axes.

**2. Time Embedding Injection:**
The embedding vector is modulated by a linear projection layers and added directly to the spatial feature map:

$$y_t = \mathbf{W}_m \cdot \text{SiLU}(e_t) + b_m, \quad y_t \in \mathbb{R}^{C_{\text{out}}}$$

$$h_2 = h_1 \oplus y_t$$

Where $\oplus$ denotes broadcasting addition along the height and width spatial dimensions. For any coordinate index $(c, i, j)$, the injection operation is explicitly evaluated as:

$$h_2[c, i, j] = h_1[c, i, j] + y_t[c]$$

**3. Second Spatial Projection & Residual Blending:**
The intermediate state undergoes another non-linear mapping before being combined with the shortcut path:

$$h_3 = \text{SiLU}\left( \text{GroupNorm}\left( \mathbf{W}_2 * h_2 + b_2 \right) \right)$$


$$\text{Output} = h_3 + \mathcal{S}(x)$$

Where $\mathcal{S}(x)$ represents the identity map if $C_{\text{in}} = C_{\text{out}}$, or a $1 \times 1$ spatial convolution projection if channel dimensions do not match.

<br>

### Code-to-Equation Breakdown

#### Line 19: `h = self.act(self.norm1(self.conv1(x)))`

* **Explanation:** Evaluates the initial feature extraction step. It computes the $3 \times 3$ padded spatial convolution, applies group normalization, and scales the outputs using the SiLU ($\text{SiLU}(v) = v \cdot \sigma(v)$) activation function.
* **Mathematical Operation:** Evaluation of $h_1$.

<br>

#### Line 20: `h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]`

* **Explanation:** Passes the time vector through the activation function and projects it to match the channel count of the spatial activations. The unsqueeze operator `[:, :, None, None]` reshapes the vector to shape `(B, C_out, 1, 1)` to allow broadcasting across the spatial grid.
* **Mathematical Operation:** Evaluation of $h_2 = h_1 \oplus y_t$.
* **Anchor Demonstration:** Let's calculate the numerical injection on our anchor example. Assume at channel $c=0$, spatial position $(0, 0)$, the feature layer has value $h_1[0, 0, 0, 0] = 0.4500$. Suppose the projected time embedding scalar for that channel is $y_t[0] = -0.1200$.
The addition maps directly to:

$$h_2[0, 0, 0, 0] = h_1[0, 0, 0, 0] + y_t[0] = 0.4500 + (-0.1200) = 0.3300$$

This exact same time scalar $y_t[0] = -0.1200$ is added to every spatial coordinate index $(i, j) \in \{1..4\} \times \{1..4\}$ across that specific channel block.

<br>

#### Line 21 & 22: `h = self.act(self.norm2(self.conv2(h)))` and `return h + self.skip(x)`

* **Explanation:** Applies the final convolution block and combines the resulting feature tensor with the original input tensor via the residual connection shortcut.
* **Mathematical Operation:** $h_3 + \mathcal{S}(x)$.

---

## Cell 17: U-Net Model Parameter Count Calculation

This cell implements the core structural engine of your framework: a time-conditioned convolutional U-Net. It processes a noisy latent tensor $x_t$ along with its corresponding time index $t$, outputting a tensor of identical spatial dimensions representing the predicted noise map $\epsilon_\theta(x_t, t)$. The network functions as a localized function approximator that uses skip connections to preserve high-frequency spatial topologies across different dimensional resolutions.

<br>

### Mathematical Formulation

Let the network be defined as a parameterized mapping $\Phi_\theta: (\mathbb{R}^{C \times H \times W} \times \mathbb{N}_0) \to \mathbb{R}^{C \times H \times W}$. This architecture can be decomposed into four distinct mathematical stages:

**1. Temporal Encoding Multi-Layer Perceptron (MLP):**
Given a scalar batch tensor of discrete time coordinates $t$, it is mapped through the sinusoidal positional encoder $\Psi(t)$ defined in Cell 15. The output vector undergoes an affine projection followed by a non-linear activation:

$$e_t = \text{SiLU}\left( \mathbf{W}_t \cdot \Psi(t) + b_t \right), \quad \mathbf{W}_t \in \mathbb{R}^{D \times D}, \ e_t \in \mathbb{R}^D$$

**2. Hierarchical Multi-Scale Spatial Encoding (Downsampling Path):**
The network contracts the spatial footprint while expanding the latent feature depth. Let $\mathcal{R}_k$ denote a conditioned residual block and $\mathcal{D}$ denote the spatial downsampling operator. Your network uses a maximum pooling operation with a kernel size of 2 and stride of 2:

$$\mathcal{D}(A)_{b, c, i, j} = \max_{m, n \in \{0, 1\}} A_{b, c, 2i+m, 2j+n}$$

The encoder layer updates are formalized as:

$$e_1 = \mathcal{R}_1(x, e_t) \in \mathbb{R}^{C_1 \times H \times W}$$

$$e_2 = \mathcal{R}_2\left(\mathcal{D}(e_1), e_t\right) \in \mathbb{R}^{C_2 \times \frac{H}{2} \times \frac{W}{2}}$$

$$e_3 = \mathcal{R}_3\left(\mathcal{D}(e_2), e_t\right) \in \mathbb{R}^{C_3 \times \frac{H}{4} \times \frac{W}{4}}$$

**3. Latent Bottleneck Processing:**
At the lowest spatial resolution, the activations pass through structural layers designed to process global context without changing the spatial dimensions:

$$b_0 = \mathcal{D}(e_3) \in \mathbb{R}^{C_3 \times \frac{H}{8} \times \frac{W}{8}}$$

$$b_1 = \mathcal{R}_{\text{bot1}}(b_0, e_t)$$

$$b_2 = \mathcal{R}_{\text{bot2}}(b_1, e_t)$$

**4. Hierarchical Spatial Decoding with Concatenated Short-Circuits (Upsampling Path):**
The network reconstructs spatial coordinates using an upsampling operator $\mathcal{U}$, implemented via nearest-neighbor spatial interpolation. For an activation $A \in \mathbb{R}^{C \times H' \times W'}$, the upsampled tensor has coordinates:

$$\mathcal{U}(A)_{b, c, i, j} = A_{b, c, \lfloor i/2 \rfloor, \lfloor j/2 \rfloor}$$

The upsampled feature maps are concatenated along the channel axis with the corresponding encoder activations before entering the decoder blocks:

$$d_3 = \mathcal{R}_{\text{dec3}}\left( \left[ \mathcal{U}(b_2) \parallel e_3 \right], e_t \right) \in \mathbb{R}^{C_2 \times \frac{H}{4} \times \frac{W}{4}}$$

$$d_2 = \mathcal{R}_{\text{dec2}}\left( \left[ \mathcal{U}(d_3) \parallel e_2 \right], e_t \right) \in \mathbb{R}^{C_1 \times \frac{H}{2} \times \frac{W}{2}}$$

$$d_1 = \mathcal{R}_{\text{dec1}}\left( \left[ \mathcal{U}(d_2) \parallel e_1 \right], e_t \right) \in \mathbb{R}^{C_1 \times H \times W}$$

Where $\parallel$ represents the tensor concatenation operator along axis 1.

**5. Linear Spatial Projection Output:**
The final output maps the feature channel space back to the original image dimensions using a $1 \times 1$ spatial convolution:

$$\epsilon_\text{pred} = \mathbf{W}_\text{out} * d_1 + b_\text{out}, \quad \mathbf{W}_\text{out} \in \mathbb{R}^{C_{\text{in}} \times C_1 \times 1 \times 1}$$

<br>

### Code-to-Equation Breakdown

#### Line 33: `t_emb = self.time_mlp(t)`

* **Explanation:** Encodes the batch of scalar time indices into a continuous embedding vector using a combination of sinusoidal mapping and linear transformations.
* **Mathematical Operation:** Evaluation of $e_t$.

<br>

#### Lines 35-37: Encoder Operations

```python
e1 = self.enc1(x, t_emb)
e2 = self.enc2(self.down(e1), t_emb)
e3 = self.enc3(self.down(e2), t_emb)
```

* **Explanation:** Alternates feature processing with spatial downsampling via max-pooling to contract the tensor's spatial footprint.
* **Mathematical Operation:** Computation of $e_1, e_2, e_3$.
* **Anchor Demonstration:** We trace the dimensions using our anchor input $x \in \mathbb{R}^{2 \times 3 \times 4 \times 4}$ with $ base_ch = 64$:
	* $x \to \text{shape } (2, 3, 4, 4)$
	* `enc1` maps channels $3 \to 64$, keeping spatial dimensions at $(4, 4)$. Thus, $e_1 \in \mathbb{R}^{2 \times 64 \times 4 \times 4}$.
	* `down(e1)` applies a $2 \times 2$ max pooling operation, reducing dimensions to $(2, 2)$. `enc2` maps channels $64 \to 128$. Thus, $e_2 \in \mathbb{R}^{2 \times 128 \times 2 \times 2}$.
	* `down(e2)` reduces spatial dimensions to $(1, 1)$. `enc3` maps channels $128 \to 256$. Thus, $e_3 \in \mathbb{R}^{2 \times 256 \times 1 \times 1}$.

<br>

#### Line 39: `b = self.bot2(self.bot1(self.down(e3), t_emb), t_emb)`

* **Explanation:** Processes the lowest layer of the spatial pyramid. On our specific anchor tensor, calling `down(e3)` on an existing $1 \times 1$ feature grid applies a max pooling operation over a single pixel, leaving the dimensions unchanged at $(1, 1)$.
* **Mathematical Operation:** Evaluation of $b_2$.
* **Anchor Demonstration:** $b \in \mathbb{R}^{2 \times 256 \times 1 \times 1}$.

<br>

#### Lines 41-43: Decoder Operations with Skip Connections

```python
d3 = self.dec3(torch.cat([self.up(b),  e3], dim=1), t_emb)
d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
```

* **Explanation:** Reconstructs the high-resolution spatial grid by upsampling the feature maps and concatenating them with the corresponding activations saved from the encoder path.
* **Mathematical Operation:** Evaluation of $d_3, d_2, d_1$.
* **Anchor Demonstration:** We trace the dimensional concatenation step-by-step:
	* `up(b)` maps the spatial dimensions from $(1, 1) \to (2, 2)$ via copy expansion. It is concatenated with $e_3 \in \mathbb{R}^{2 \times 256 \times 1 \times 1}$.


> **Note on Dimensional Mismatch:** Because our anchor example features a downscaled $4 \times 4$ resolution, the downsampling path hits a spatial size of $1 \times 1$ at $e_3$. Upsampling it via `up(b)` yields a $2 \times 2$ tensor, which cannot be directly concatenated with the $1 \times 1$ tensor $e_3$. In full production on CIFAR-10 ($32 \times 32$), the dimensions scale down smoothly ($32 \to 16 \to 8 \to 4$), so the shapes align at every decoder step ($4 \to 8 \to 16 \to 32$).


To maintain exact mathematical continuity on our anchor example, we focus on the final resolution block `dec1` where the structural feature maps match:
* Assume $\mathcal{U}(d_2) \in \mathbb{R}^{2 \times 64 \times 4 \times 4}$ and $e_1 \in \mathbb{R}^{2 \times 64 \times 4 \times 4}$.
* `torch.cat` combines these tensors along axis 1, expanding the channel dimension: $64 + 64 = 128$. This yields a concatenated tensor of shape $(2, 128, 4, 4)$.
* `dec1` projects these channels back down to 64, outputting $d_1 \in \mathbb{R}^{2 \times 64 \times 4 \times 4}$.

<br>

#### Line 44: `return self.out(d1)`

* **Explanation:** Projects the hidden feature channels back down to match the input image dimensions.
* **Mathematical Operation:** $\mathbf{W}_\text{out} * d_1 + b_\text{out}$.
* **Anchor Demonstration:** The output convolution layer uses a $1 \times 1$ kernel to project the channels from $64 \to 3$, which returns a final tensor of shape `(2, 3, 4, 4)`, matching our anchor dimensions.

---

## Cell 18: Instantiating AdamW Optimizer for DDPM Baseline

This cell provides numerical validation of the backward pass through the time-conditioned U-Net score network. It calculates the mean of the output tensor to form a scalar optimization target, computes the partial derivatives for all learnable parameters using automatic differentiation, and inspects the structural distribution of gradient norms to check for numerical errors like vanishing or exploding gradients.

<br>

### Mathematical Formulation

Let $\mathbf{Y} = \Phi_\theta(x, t) \in \mathbb{R}^{B \times C \times H \times W}$ represent the multi-channel output tensor produced by the U-Net forward pass.

**1. Scalar Objective Mapping:**
The forward pass output is collapsed into a scalar field value $L \in \mathbb{R}$ by taking the global arithmetic mean over all tensor coordinates:

$$L = \frac{1}{B \cdot C \cdot H \cdot W} \sum_{b=1}^{B} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} Y_{b,c,i,j}$$

**2. The Backward Pass (Reverse-Mode Automatic Differentiation):**
Using the chain rule, the gradient of this scalar objective with respect to any individual weight parameter $w_{k} \in \theta$ is computed by propagating error sensitivities backward through the network's computational graph:

$$\frac{\partial L}{\partial w_k} = \sum_{b,c,i,j} \frac{\partial L}{\partial Y_{b,c,i,j}} \cdot \frac{\partial Y_{b,c,i,j}}{\partial w_k}$$

Differentiating our scalar objective formula shows that the initial error sensitivity injected at the output layer is a constant uniform matrix:

$$\frac{\partial L}{\partial Y_{b,c,i,j}} = \frac{1}{B \cdot C \cdot H \cdot W}$$

**3. Euclidean Matrix Norm Evaluation:**
To monitor training stability, the individual parameter gradient matrices $\nabla_{\mathbf{W}_k} L$ are compressed into a scalar magnitude using the Frobenius (Euclidean) norm:

$$\|\nabla_{\mathbf{W}_k} L\|_F = \sqrt{\sum_{m} \sum_{n} \left( \frac{\partial L}{\partial W_{k, m, n}} \right)^2}$$

<br>

### Code-to-Equation Breakdown

#### Line 2: `loss = out.mean()`

* **Explanation:** Sums up every element in the output tensor and divides by the total element count to create a scalar tracking value.
* **Mathematical Operation:** Evaluation of $L$.
* **Anchor Demonstration:** For our anchor dimension setup ($B=2, C=3, H=4, W=4$), the denominator is $2 \times 3 \times 4 \times 4 = 92$. The scalar loss is:

$$L = \frac{1}{92} \sum_{b=0}^{1}\sum_{c=0}^{2}\sum_{i=1}^{4}\sum_{j=1}^{4} Y_{b,c,i,j}$$

<br>

#### Line 3: `loss.backward()`

* **Explanation:** Traverses the computational graph in reverse order, calculating the partial derivatives for all active model weights.
* **Mathematical Operation:** Evaluation of $\frac{\partial L}{\partial w_k}$ for all parameters in $\theta$.

<br>

#### Line 5: `grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}`

* **Explanation:** Iterates through the model parameters, evaluates the Frobenius norm of each weight gradient tensor, and stores the scalar values in a dictionary.
* **Mathematical Operation:** Evaluation of $\|\nabla_{\mathbf{W}_k} L\|_F$.

---

## Cell 20: Denoising Training Loop, Gradient Norm Clipping, and Loss Minimization

This cell implements the training loop that optimizes the weights of the time-conditioned U-Net score network. It samples mini-batches of normalized clean data, applies the forward corruption process over uniformly distributed time steps, executes gradient descent with gradient norm clipping to stabilize training, and updates parameters via the AdamW optimizer.

<br>

### Mathematical Formulation

The optimization routine minimizes the empirical expectation of the Denoising Score Matching loss function across the training dataset:

$$\theta^* = \arg\min_\theta \frac{1}{|D|} \sum_{i \in D} \mathcal{L}_{\text{simple}}(\theta; x_{0}^{(i)})$$

For each mini-batch step, three core mathematical sequences occur:

**1. Gradient Computation via Backpropagation:**
The parameter gradients are evaluated for the mini-batch objective function:

$$\mathbf{g}_k = \nabla_{\mathbf{w}_k} \mathcal{L}_{\text{simple}}(\theta)$$

**2. Gradient Norm Clipping:**
To prevent training instability caused by high error gradients, the global gradient vector is scaled if its total Euclidean norm exceeds a maximum threshold $c_{\max} = 1.0$:

$$\mathbf{G} = \left[ \mathbf{g}_1, \mathbf{g}_2, \dots, \mathbf{g}_K \right]^T$$

$$\text{If } \|\mathbf{G}\|_2 > c_{\max} \implies \mathbf{g}_k \leftarrow \mathbf{g}_k \cdot \frac{c_{\max}}{\|\mathbf{G}\|_2} \quad \forall k$$

Where $\|\mathbf{G}\|_2 = \sqrt{\sum_k \|\mathbf{g}_k\|_F^2}$.

**3. AdamW Parameter Update:**
The optimizer updates parameter matrices using a decoupled weight decay scheme alongside running estimates of the first and second uncentered moments of the gradients:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{G}_t$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{G}_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

$$\mathbf{w}_{t} = \mathbf{w}_{t-1} - \eta \cdot \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \mathbf{w}_{t-1} \right)$$

Where $\eta = 2 \times 10^{-4}$ is the learning rate, $\lambda$ is the weight decay coefficient, and $\beta_1, \beta_2$ are the moment decay constants.

<br>

### Code-to-Equation Breakdown

#### Line 10: `loss = diffusion_loss(model, imgs, ab_dev)`

* **Explanation:** Evaluates the denoising loss for the current mini-batch as defined in Cell 12.
* **Mathematical Operation:** Evaluation of $\mathcal{L}_{\text{simple}}(\theta)$.
* **Anchor Demonstration:** Referred to previous step (Cell 12). The mini-batch baseline yields a scalar value corresponding to the mean squared error of the predicted noise vector.

<br>

#### Line 11: `loss.backward()`

* **Explanation:** Propagates the scalar loss error backward through the layers of the U-Net computational graph to calculate parameter gradients.
* **Mathematical Operation:** Evaluates $\mathbf{G}_t$.

<br>

#### Line 12: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

* **Explanation:** Calculates the total Euclidean norm of all model parameter gradients combined and scales them down if the norm exceeds $1.0$.
* **Mathematical Operation:** Evaluates $\|\mathbf{G}\|_2$ and rescales the gradient vector if $\|\mathbf{G}\|_2 > 1.0$.
* **Anchor Demonstration:** Suppose our U-Net model has two parameters with Frobenius gradient norms $\|\mathbf{g}_1\|_F = 0.5$ and $\|\mathbf{g}_2\|_F = 1.2$.
The total global gradient norm is:

$$\|\mathbf{G}\|_2 = \sqrt{(0.5)^2 + (1.2)^2} = \sqrt{0.25 + 1.44} = \sqrt{1.69} = 1.30$$

Since $1.30 > 1.0$, the clipping condition triggers. The scaling factor is $\frac{1.0}{1.30} \approx 0.7692$.
The individual gradients are updated to:

$$\mathbf{g}_1 \leftarrow 0.7692 \times \mathbf{g}_1 \implies \|\mathbf{g}_1\|_F \approx 0.3846$$

$$\mathbf{g}_2 \leftarrow 0.7692 \times \mathbf{g}_2 \implies \|\mathbf{g}_2\|_F \approx 0.9231$$

The modified global norm matches the target limit: $\sqrt{(0.3846)^2 + (0.9231)^2} = \sqrt{0.1479 + 0.8521} = 1.00$.

<br>

#### Line 13: `optimizer.step()`

* **Explanation:** Updates the U-Net parameter weights based on the clipped gradients using the AdamW step logic.
* **Mathematical Operation:** Evaluation of $\mathbf{w}_t$.

<br>

#### Line 16: `avg = epoch_loss / len(train_loader)`

* **Explanation:** Computes the average loss across all batches processed during the epoch.
* **Mathematical Operation:** Calculates the empirical training loss mean $\mu_{\mathcal{L}} = \frac{1}{B_{\text{total}}} \sum_{b=1}^{B_{\text{total}}} \mathcal{L}_b$.

---

## Cell 22: DDPM Stochastic Ancestral Reverse Sampler

This cell implements the complete discrete-time ancestral sampling algorithm for the Denoising Diffusion Probabilistic Model (DDPM) baseline. Operating in reverse chronological order from $t = T-1$ down to $t = 0$, the sampler parameterizes the intractable true reverse distribution $p(x_{t-1} \mid x_t)$ using a Gaussian approximation $p_\theta(x_{t-1} \mid x_t)$. It leverages the trained U-Net score network to subtract the predicted noise vector sequentially, adding a stochastic correction term at each step to guide the latent trajectory back toward the data manifold.

<br>

### Mathematical Formulation

The true reverse transition $q(x_{t-1} \mid x_t, x_0)$ is tractable when conditioned on the clean initial image $x_0$. By applying Bayes' rule over the forward Markovian transitions, this distribution can be shown to be a Gaussian distribution:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$$

The analytic parameter functions for the mean and variance are defined as follows:

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t$$

Because $x_0$ is unknown during generation, we use the forward reparameterization equation to express it as a function of the current latent state $x_t$ and the true noise vector $\epsilon$:

$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon\right)$$

Substituting this definition of $x_0$ into the formula for the true reverse mean $\tilde{\mu}_t$ reveals the direct relationship between the target mean vector and the noise map:

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\left[ \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon\right) \right] + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t$$

$$\tilde{\mu}_t(x_t, \epsilon) = \left[ \frac{\beta_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \right]x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}\epsilon$$

Simplifying the term inside the brackets using the identity $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$ yields:

$$\frac{\beta_t + \alpha_t(1 - \bar{\alpha}_{t-1})}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{\beta_t + \alpha_t - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1}{\sqrt{\alpha_t}}$$

Thus, the localized reverse mean is given by:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t) \right)$$

The generative process simulates this distribution at each step by substituting the neural network's prediction $\epsilon_\theta(x_t, t)$ for the unknown noise vector:

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

Where Ho et al. set the variance parameter to $\sigma_t^2 = \beta_t$ as a pragmatic choice that matches the upper bound of the reverse process variance.

<br>

### Code-to-Equation Breakdown

#### Line 16: `x = torch.randn(n_samples, 3, 32, 32, device=device)`

* **Explanation:** Generates the initial high-entropy latent state batch from a standard normal distribution.
* **Mathematical Operation:** Initial condition $x_{T-1} \sim \mathcal{N}(0, \mathbf{I})$.
* **Anchor Demonstration:** To align with our previous multi-sample tracking framework, we evaluate this sampler using our two-sample anchor grid ($B=2, C=3, H=4, W=4$). Per our setup guidelines, the random tensor is locked to a uniform constant:

$$\text{For all elements: } x_{999}[n, c, i, j] = +0.5$$

<br>

#### Line 22: `coeff = be[t] / (1 - ab[t]).sqrt()`

* **Explanation:** Computes the scalar scaling coefficient applied to the predicted noise vector.
* **Mathematical Operation:** Evaluates the multiplier $C_t = \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}$.
* **Anchor Demonstration:** Let's calculate the value at the very first step of the reverse loop, $t = 999$. We fetch the parameters from Cell 7 ($\beta_{999} = 0.020000$ and $\bar{\alpha}_{999} = 0.040358$):

$$C_{999} = \frac{0.020000}{\sqrt{1 - 0.040358}} = \frac{0.020000}{\sqrt{0.959642}} = \frac{0.020000}{0.979613} \approx 0.020416$$

<br>

#### Line 23: `mean = (1 / al[t].sqrt()) * (x - coeff * eps_pred)`

* **Explanation:** Combines the current latent coordinates with the scaled prediction tensor, then scales the result by the reciprocal square root of $\alpha_t$ to compute the localized distribution mean.
* **Mathematical Operation:** Evaluates $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - C_t \cdot \epsilon_\theta(x_t, t)\right)$.
* **Anchor Demonstration:** We calculate the element-wise values at step $t=999$ using our anchor parameters ($\alpha_{999} = 0.980000 \implies \sqrt{\alpha_{999}} \approx 0.989949$). We assume our network outputs a uniform zero prediction for this initial test step:

$$\epsilon_\theta(x_{999}, 999)[n, c, i, j] = 0.0$$

<br>

Substituting these values for every position gives:

$$\mu_{999} = \frac{1}{0.989949} \cdot \left( 0.5 - (0.020416 \times 0.0) \right) = \frac{0.5}{0.989949} \approx 0.505076$$

<br>

#### Line 26: `sigma = be[t].sqrt()`

* **Explanation:** Computes the standard deviation coefficient for the stochastic correction term.
* **Mathematical Operation:** Evaluates $\sigma_t = \sqrt{\beta_t}$.
* **Anchor Demonstration:** At step $t = 999$:

$$\sigma_{999} = \sqrt{0.020000} \approx 0.141421$$

<br>

#### Line 27: `x = mean + sigma * torch.randn_like(x)`

* **Explanation:** Adds random Gaussian noise to the calculated mean vector to complete the stochastic transition step down the Markov chain.
* **Mathematical Operation:** Updates the latent state to $x_{t-1} = \mu_\theta + \sigma_t z$.
* **Anchor Demonstration:** Using our static anchor noise convention ($z = +0.5$ at all positions), we calculate the updated latent state values for the next step ($x_{998}$):

$$x_{998}[n, c, i, j] = 0.505076 + (0.141421 \times 0.5) = 0.505076 + 0.070711 = 0.575787$$

<br>

This updated tensor $x_{998}$ serves as the input for the next step ($t=998$) of the reverse loop, and this process repeats sequentially down to $t=0$.

---

## Cell 23: Accelerated Strided Sub-Sequence Sampler (DDIM Formulation)

This cell implements an accelerated strided sampler that reduces the Number of Function Evaluations (NFE) during inference. It uses the deterministic Denoising Diffusion Implicit Model (DDIM) formulation to skip steps along a sub-sequence of the original timeline $\tau \subset \{0, 1, \dots, T-1\}$. At each sub-step, the sampler uses the network's noise prediction to project an estimate of the clean image $x_0$, then uses that estimate to extrapolate the next target latent state.

<br>

### Mathematical Formulation

Let $\tau = [\tau_S, \tau_{S-1}, \dots, \tau_1]$ be a decreasing sub-sequence of $S$ indices sampled from the full timeline. For any current sub-step index $\tau_i = t$, we denote the subsequent step in our sampling sequence as $\tau_{i+1} = s$ (where $s < t$).

**1. Reconstructing the Clean Image Estimate ($\hat{x}_0$):**
Using Tweedie's formula, the model isolates the underlying signal by subtracting the scaled noise prediction $\epsilon_\theta(x_t, t)$ from the current latent state $x_t$:

$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**2. Extrapolating to the Next Latent State ($x_s$):**
The forward marginal distribution is rewritten to define $x_s$ as a combination of our clean data estimate $\hat{x}_0$ and a directional noise vector:

$$x_s = \sqrt{\bar{\alpha}_s}\hat{x}_0 + \sqrt{1 - \bar{\alpha}_s - \sigma_s^2}\epsilon_\theta(x_t, t) + \sigma_s z_t, \quad z_t \sim \mathcal{N}(0, \mathbf{I})$$

Setting the stochastic coefficient $\sigma_s = 0$ makes the reverse trajectory deterministic, mapping a specific initial noise block directly to a unique generated image. Your code instead implements the stochastic DDIM variant by matching the true posterior variance across skipped intervals:

$$\sigma_s = \sqrt{\frac{1 - \bar{\alpha}_s}{1 - \bar{\alpha}_t} \left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_s}\right)}$$

<br>

### Code-to-Equation Breakdown

#### Line 13: `timesteps = list(reversed(range(0, T, T // n_steps)))[:n_steps]`

* **Explanation:** Generates a uniformly spaced subset of time indices, reversing their order to step backward from noise to data.
* **Mathematical Operation:** Defines the sub-sequence array $\tau$ where $|\tau| = S = 100$.
* **Anchor Demonstration:** For $T=1000$ and $n\_steps=100$, the stride size is $1000 // 100 = 10$. The loop constructs the index list $\tau = [990, 980, \dots, 10, 0]$.

<br>

#### Line 25: `x0_hat = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()`

* **Explanation:** Reconstructs the clean pixel coordinates using the current noisy latent state and the network's noise prediction.
* **Mathematical Operation:** Evaluation of $\hat{x}_0$.
* **Anchor Demonstration:** We trace the initial step of this sequence ($i=0 \implies t = 990$) using our two-sample anchor grid ($B=2, C=3, H=4, W=4$). We pull the parameters from Cell 7 ($\bar{\alpha}_{990} \approx 0.0465 \implies \sqrt{\bar{\alpha}_{990}} \approx 0.2156$ and $\sqrt{1 - \bar{\alpha}_{990}} \approx 0.9765$). Assuming our initial state tensor contains $+0.5$ at all positions and the network outputs zeros:

$$\hat{x}_0 = \frac{0.5 - (0.9765 \times 0.0)}{0.2156} = \frac{0.5}{0.2156} \approx 2.3191$$

<br>

#### Line 26: `x0_hat = x0_hat.clamp(-1, 1)`

* **Explanation:** Restricts the values of the clean image estimate to the model's operational data range $[-1, 1]$.
* **Mathematical Operation:** $\hat{x}_0 \leftarrow \max(-1.0, \min(1.0, \hat{x}_0))$.
* **Anchor Demonstration:** Our unconstrained estimate exceeds the upper bound ($2.3191 > 1.0$), so the clamping operation clips it to the maximum threshold:

$$\hat{x}_0[n, c, i, j] = +1.0$$

<br>

#### Line 29: `mean = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps_pred`

* **Explanation:** Blends the clamped clean data estimate with the predicted noise direction vector to compute the target latent state mean for the next step.
* **Mathematical Operation:** Evaluation of the deterministic directional update $\mu = \sqrt{\bar{\alpha}_s}\hat{x}_0 + \sqrt{1 - \bar{\alpha}_s}\epsilon_\theta$.
* **Anchor Demonstration:** The next step in our sub-sequence is $s = 980$. We fetch its parameter: $\bar{\alpha}_{980} \approx 0.0532 \implies \sqrt{\bar{\alpha}_{980}} \approx 0.2307$. Since our network prediction is zero, the directional update evaluates to:

$$\text{mean} = (0.2307 \times 1.0) + (\sqrt{1 - 0.0532} \times 0.0) = 0.2307$$

---

## Cell 24: Inverse Affine Mapping for Output Image Grid Visualization

This cell implements the final visual evaluation stage for the DDPM baseline, arranging a batch of 16 generated samples into a $4 \times 4$ image grid. The visualization function applies an inverse affine transformation to map the outputs from the model's operational domain back to standard pixel values, providing a direct visual check of the model's generation quality.

<br>

### Mathematical Formulation

Let $X_{\text{out}} \in \mathbb{R}^{B \times C \times H \times W}$ represent the final output tensor returned by the reverse sampling loop at $t=0$. To display these raw continuous vectors using standard graphics libraries, the pixel channels must be rescaled and constrained to the standard display space $[0, 1]$.

**1. Inverse Affine Mapping:**
The model's internal data manifold is centralized around zero within the boundaries $[-1, 1]$. We apply an inverse scaling function $g: [-1, 1] \to [0, 1]$ to shift and compress the coordinates:

$$g(X_{\text{out}}) = \frac{X_{\text{out}} + 1.0}{2.0}$$

**2. Boundary Saturation (Clamping):**
Because numerical compounding errors during the 1,000-step reverse loop can push final values outside the expected data range, a hard bounding operator is applied element-wise before mapping to pixel intensities:

$$\bar{X} = \max\left(0.0, \min\left(1.0, g(X_{\text{out}})\right)\right)$$

<br>

### Code-to-Equation Breakdown

#### Line 5: `imgs = (imgs.clamp(-1, 1) + 1) / 2`

* **Explanation:** Clips out-of-bounds latent coordinates to the range $[-1, 1]$ and applies an inverse affine transformation to rescale them to a clean $[0, 1]$ range for image display.
* **Mathematical Operation:** Evaluation of $\bar{X} = \frac{\text{clamp}(X_{\text{out}}, -1, 1) + 1}{2}$.
* **Anchor Demonstration:** We trace this operation using our previous reverse ancestral sampling baseline outputs from Cell 22. Recall that our anchor tensor evaluation at step $t=998$ yielded a uniform value of $+0.575787$ across all elements.
Assuming the reverse loop continues down to $t=0$ and maintains this exact constant value, the output tensor processing steps unfold as follows:
	* **Clamping:** $+0.575787$ is already well within $[-1.0, 1.0]$, so the tensor remains unchanged.
	* **Rescaling:**
	
$$\bar{X} = \frac{0.575787 + 1.0}{2.0} = \frac{1.575787}{2.0} \approx 0.7879$$

Every item in the grid maps to a uniform intensity of $0.7879$. Across all three channels (Red, Green, Blue), this equal balance creates a solid, light-gray square grid.

---

## Cell 26: Rectified Flow Linear Interpolation Objective and Target Velocity Field

This cell implements the training objective for the Rectified Flow (Flow Matching) framework. Instead of corrupting data using a non-linear geometric schedule along a curved Markovian path like traditional DDPM, Rectified Flow constructs a deterministic, straight-line trajectory connecting the clean data distribution directly to a standard normal noise distribution. The neural network is reparameterized as a velocity vector field $v_\theta(x_t, t)$ that learns to predict the constant directional velocity required to transport a point from the data manifold to pure noise across a continuous time domain $t \in [0, 1]$.

<br>

### Mathematical Formulation

Let $x_0 \sim q(x_0)$ represent a sample from the empirical data distribution, and let $x_1 \sim \mathcal{N}(0, \mathbf{I})$ represent an independent sample drawn from the target isotropic Gaussian noise distribution.

**1. The Linear Flow Interpolation (The Flow Map):**
The continuous conditional trajectory $X_t: [0, 1] \to \mathbb{R}^{C \times H \times W}$ is defined as a simple linear combination (or algebraic mixture) of the boundary endpoints:

$$X_t(x_0, x_1) = (1 - t)x_0 + t x_1$$

**2. The Field Velocity Derivative:**
The exact ground-truth velocity vector field $u(x_t, t)$ that drives this moving point along its path is found by differentiating the flow map equation with respect to the continuous time variable $t$:

$$u(x_t, t) = \frac{\text{d}X_t}{\text{d}t} = \frac{\text{d}}{\text{d}t}\left[ (1 - t)x_0 + t x_1 \right] = -x_0 + x_1 = x_1 - x_0$$

Notice that this velocity vector is completely independent of the time variable $t$. It is a constant vector field for any given pair of endpoints, describing a perfectly straight coordinate bridge.

**3. The Flow Matching Objective Function:**
The objective minimizes the Mean Squared Error between the velocity field predicted by the neural network $v_\theta(x_t, t)$ and the true constant target velocity vector across all points on the interpolated path:

$$\mathcal{L}_{\text{RF}}(\theta) = \mathbb{E}_{x_0 \sim q(x_0), x_1 \sim \mathcal{N}(0, \mathbf{I}), t \sim \mathcal{U}([0,1])} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

<br>

### Code-to-Equation Breakdown

#### Line 11: `x1 = torch.randn_like(x0)`

* **Explanation:** Generates a target high-entropy Gaussian noise tensor matching the structural dimensions of the incoming data batch.
* **Mathematical Operation:** Samples $x_1 \sim \mathcal{N}(0, \mathbf{I})$.
* **Anchor Demonstration:** For our anchor grid ($B=2, C=3, H=4, W=4$), we follow our established rules and fix this random tensor to a uniform constant:

$$\text{For all elements: } x_1[n, c, i, j] = +0.5$$

<br>

#### Line 12 & 13: `t = torch.rand(B, device=x0.device)` and `t4 = t.view(-1, 1, 1, 1)`

* **Explanation:** Samples a continuous time scalar uniformly between 0 and 1 for each batch item and unsqueezes it to shape `(B, 1, 1, 1)` for spatial broadcasting.
* **Mathematical Operation:** Samples $t_b \sim \mathcal{U}([0, 1])$ for $b \in \{0, 1\}$.
* **Anchor Demonstration:** To map this directly onto our anchor framework, we choose the continuous time values:

$$\mathbf{t} = [0.25, 0.75]^T$$

<br>

#### Line 14: `x_t = (1 - t4) * x0 + t4 * x1`

* **Explanation:** Blends the clean image tensor with the noise tensor using the time-dependent linear scaling factors to find the intermediate coordinates along the straight-line trajectory.
* **Mathematical Operation:** Evaluation of $x_t = (1 - t)x_0 + t x_1$.
* **Anchor Demonstration:** We calculate the intermediate states element-by-element for our anchor samples:

**For Sample 0 (Checkerboard) at $t_0 = 0.25$:**
* Where $x_0 = +1.0$:

$$x_{0.25} = (1 - 0.25)(1.0) + 0.25(0.5) = 0.75 + 0.125 = 0.8750$$

* Where $x_0 = -1.0$:

$$x_{0.25} = (1 - 0.25)(-1.0) + 0.25(0.5) = -0.75 + 0.125 = -0.6250$$

**For Sample 1 (Spatial Gradient) at $t_1 = 0.75$:**

* At Column 0 where $x_0 = -1.0$:

$$x_{0.75} = (1 - 0.75)(-1.0) + 0.75(0.5) = -0.25 + 0.375 = 0.1250$$

* At Column 3 where $x_0 = +1.0$:

$$x_{0.75} = (1 - 0.75)(1.0) + 0.75(0.5) = 0.25 + 0.375 = 0.6250$$

<br>

#### Line 15: `target = x1 - x0`

* **Explanation:** Computes the true constant velocity vector by subtracting the clean data coordinates from the noise coordinates.
* **Mathematical Operation:** Evaluates the target vector $\mathbf{u} = x_1 - x_0$.
* **Anchor Demonstration:** We calculate the target values element-by-element across our samples:

For Sample 0 where $x_0 = +1.0$:

$$\mathbf{u} = 0.5 - 1.0 = -0.5000$$

For Sample 0 where $x_0 = -1.0$:

$$\mathbf{u} = 0.5 - (-1.0) = 1.5000$$

For Sample 1 where $x_0 = -1.0$ (Column 0):

$$\mathbf{u} = 0.5 - (-1.0) = 1.5000$$

For Sample 1 where $x_0 = +1.0$ (Column 3):

$$\mathbf{u} = 0.5 - 1.0 = -0.5000$$

<br>

#### Line 17 & 18: `t_int = (t * (T - 1)).long()` and `v_pred = model(x_t, t_int)`

* **Explanation:** Maps the continuous continuous time interval $[0, 1]$ to the discrete integer timeline $[0, T-1]$ so the time coordinates can be processed by the U-Net's sinusoidal embedding layer.
* **Mathematical Operation:** Evaluates $t_{\text{discrete}} = \lfloor t \cdot 999 \rfloor$.
* **Anchor Demonstration:** For our continuous anchor times:
	* $t_0 = 0.25 \implies t_{\text{discrete}, 0} = \lfloor 0.25 \times 999 \rfloor = \lfloor 249.75 \rfloor = 249$
	* $t_1 = 0.75 \implies t_{\text{discrete}, 1} = \lfloor 0.75 \times 999 \rfloor = \lfloor 749.25 \rfloor = 749$

These discrete indices are converted into embedding vectors to condition the model's velocity predictions: $v_\theta(x_t, \mathbf{t}_{\text{discrete}})$.

---

## Cell 29: Continuous ODE Reverse Sampler via First-Order Euler Integration

This cell implements the deterministic generative sampler for the Rectified Flow framework. Generation in Rectified Flow is framed as solving an ordinary differential equation (ODE) rather than tracking a stochastic Markov chain. By integrating the learned velocity field $v_\theta(x_t, t)$ backward in time from $t=1$ (pure noise) to $t=0$ (clean data) using a first-order Euler integration scheme, the sampler traces deterministic trajectories across the data manifold.

<br>

### Mathematical Formulation

The generation process is governed by a first-order, time-dependent vector ODE that maps a high-entropy latent probability distribution directly to the empirical data distribution:

$$\frac{\mathrm{d}X_t}{\mathrm{d}t} = v_\theta(X_t, t)$$

To solve this continuous-time system numerically over a finite discrete computational budget, the continuous time interval is discretized into $S$ uniform sub-intervals. Let $S = \text{n\_steps}$ represent the total number of function evaluations, and let the infinitesimal time differential be approximated by the finite step size $\Delta t$:

$$\Delta t = \frac{1.0}{S}$$

The discrete integration grid is defined chronologically backward from noise to data:

$$\tau_i = 1.0 - i \cdot \Delta t, \quad \text{for } i \in \{0, 1, \dots, S-1\}$$

Applying a first-order forward Euler discretization to the continuous derivative yields the discrete difference equation used for the state updates:

$$\frac{X_{t - \Delta t} - X_t}{-\Delta t} \approx v_\theta(X_t, t) \implies X_{t - \Delta t} = X_t - v_\theta(X_t, t) \cdot \Delta t$$

<br>

### Code-to-Equation Breakdown

#### Line 10: `x = torch.randn(n_samples, 3, 32, 32, device=device)`

* **Explanation:** Samples the initial high-entropy spatial coordinates from an isotropic standard Gaussian distribution at the starting boundary $t=1$.
* **Mathematical Operation:** Sets the initial condition for the ODE solver: $X_1 \sim \mathcal{N}(0, \mathbf{I})$.
* **Anchor Demonstration:** For our testing baseline, we initialize our standard two-sample anchor grid ($B=2, C=3, H=4, W=4$). Per our established rules, the random tensor elements are locked to a uniform constant value:

$$\text{For all coordinates: } x_1[n, c, i, j] = +0.5$$

<br>

#### Line 13 & 14: `t_val = 1.0 - i * dt` and `t_int = torch.full(...)`

* **Explanation:** Evaluates the current position along the continuous timeline $[1, 0]$ and scales it to the integer coordinate domain $[0, T-1]$ to match the input expectations of the U-Net time embedding layers.
* **Mathematical Operation:** Evaluates $\tau_i$ and maps it to $t_{\text{discrete}} = \lfloor \tau_i \cdot 999 \rfloor$.
* **Anchor Demonstration:** Let's trace the very first integration step where $i=0$ with a configuration of $S=100 \implies \Delta t = 0.01$:

$$\tau_0 = 1.0 - 0 \times 0.01 = 1.0$$

$$t_{\text{discrete}} = \lfloor 1.0 \times 999 \rfloor = 999$$

The scalar integer tensor `t_int` is filled with the value $999$ across all entries in the batch.

<br>

#### Line 15: `v = model(x, t_int)`

* **Explanation:** Passes the current latent state and the discrete time index to the U-Net model to predict the instantaneous velocity vectors.
* **Mathematical Operation:** Evaluates the velocity vector field $v_\theta(X_t, t_{\text{discrete}})$.
* **Anchor Demonstration:** For this initialization test step, we assume our network outputs a uniform zero velocity prediction vector:

$$v_\theta(X_1, 999)[n, c, i, j] = 0.0$$

<br>

#### Line 16: `x = x - v * dt`

* **Explanation:** Displaces the current coordinates along the predicted velocity vector field over a finite step size $\Delta t$ to estimate the next state in reverse time.
* **Mathematical Operation:** Updates the latent coordinate tensor using the Euler step: $X_{t - \Delta t} = X_t - v \cdot \Delta t$.
* **Anchor Demonstration:** Substituting our anchor elements into the update equation for step $i=0$ gives:

$$x_{0.99} = x_1 - v \cdot \Delta t = 0.5 - (0.0 \times 0.01) = 0.5$$

<br>

Because the mock velocity output is zero, the state coordinates remain at $+0.5$. This updated tensor $x_{0.99}$ becomes the input for the next integration step ($i=1$), where $\tau_1 = 0.99 \implies t_{\text{discrete}} = \lfloor 0.99 \times 999 \rfloor = 989$, and the process repeats sequentially until reaching the final data manifold at $t=0$.

---

## Cell 31: Feature Space Mapping and Multivariate Gaussian Parameter Estimation

This cell implements the feature extraction and statistical profiling function required to evaluate generation quality via the Fréchet Inception Distance (FID). It maps generated or real images from their spatial pixel domain into a high-dimensional, semantically dense feature space using a pre-trained Inception-v3 network. It then summarizes this feature distribution by calculating its empirical mean vector and covariance matrix.

<br>

### Mathematical Formulation

Let $\mathcal{X} = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$ represent a collection of $N$ images where $x^{(i)} \in \mathbb{R}^{3 \times 32 \times 32}$. The function maps this dataset to a continuous distribution of feature representations through three operations:

**1. Bilinear Spatial Interpolation:**
The pre-trained Inception-v3 network requires an input resolution of at least $75 \times 75$. To upsample the $32 \times 32$ CIFAR-10 images to a target size $H' \times W' = 75 \times 75$, a continuous coordinate mapping is defined. For a target coordinate $(i', j')$, the corresponding continuous source coordinates $(i, j)$ are found via:

$$i = i' \cdot \frac{32}{75}, \quad j = j' \cdot \frac{32}{75}$$

Let $i_0 = \lfloor i \rfloor, i_1 = i_0 + 1$, and $j_0 = \lfloor j \rfloor, j_1 = j_0 + 1$ denote the indices of the four surrounding source pixels. The interpolated intensity value is computed as a weighted average based on linear distances:
$$\begin{aligned}
f(i', j') = &(1 - \Delta i)(1 - \Delta j) \cdot f(i_0, j_0) + \Delta i (1 - \Delta j) \cdot f(i_1, j_0) \
* &(1 - \Delta i)\Delta j \cdot f(i_0, j_1) + \Delta i \Delta j \cdot f(i_1, j_1)
\end{aligned}$$
Where $\Delta i = i - i_0$ and $\Delta j = j - j_0$.

**2. Feature Space Mapping:**
The upsampled images are passed through the pre-trained Inception-v3 network $\Phi: \mathbb{R}^{3 \times 75 \times 75} \to \mathbb{R}^{2048}$, extracting the 2048-dimensional continuous vector from the final global pooling layer (`pool3`):

$$\mathbf{f}^{(i)} = \Phi(x^{(i)}), \quad \mathbf{f}^{(i)} \in \mathbb{R}^{2048}$$

**3. Multivariate Gaussian Parameter Estimation:**
The feature extraction processes the image ensemble into a matrix $\mathbf{F} \in \mathbb{R}^{N \times 2048}$. The empirical distribution is modeled as a multivariate Gaussian distribution $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$:
* **Empirical Mean Vector ($\boldsymbol{\mu} \in \mathbb{R}^{2048}$):**

$$\mu_j = \frac{1}{N} \sum_{i=1}^{N} F_{i, j}$$

* **Empirical Covariance Matrix ($\mathbf{\Sigma} \in \mathbb{R}^{2048 \times 2048}$):**

$$\Sigma_{j, k} = \frac{1}{N - 1} \sum_{i=1}^{N} (F_{i, j} - \mu_j)(F_{i, k} - \mu_k)$$

<br>

### Code-to-Equation Breakdown

#### Line 12: `imgs_up = F.interpolate(imgs_01, size=(75, 75), mode="bilinear", align_corners=False)`

* **Explanation:** Rescales the spatial grid from $32 \times 32 \to 75 \times 75$ using a 2D bilinear interpolation filter to match the network's input requirements.
* **Mathematical Operation:** Evaluation of $f(i', j')$.

<br>

#### Line 20: `f = inception(x.to(device))[0].squeeze(-1).squeeze(-1)`

* **Explanation:** Extracts the `pool3` activations from the model. The `.squeeze()` calls remove the redundant trailing spatial dimensions, collapsing the final feature tensor from shape `(B, 2048, 1, 1)` to a 2D matrix of shape `(B, 2048)`.
* **Mathematical Operation:** Evaluation of $\mathbf{f}^{(i)} = \Phi(x^{(i)})$.

<br>

#### Line 24: `mu = feats.mean(axis=0)`

* **Explanation:** Calculates the arithmetic mean along the batch axis for each of the 2048 feature channels.
* **Mathematical Operation:** Evaluation of $\boldsymbol{\mu}$.
* **Anchor Demonstration:** Consider an anchor setup with $N=3$ samples projected into a simplified $D=2$ dimensional feature space:

$$\mathbf{F} = \begin{bmatrix} 1.0 & 4.0 \\ 2.0 & 5.0 \\ 3.0 & 9.0 \end{bmatrix}$$

The mean vector components are:

$$\mu_1 = \frac{1.0 + 2.0 + 3.0}{3} = 2.0, \quad \mu_2 = \frac{4.0 + 5.0 + 9.0}{3} = 6.0 \implies \boldsymbol{\mu} = [2.0, 6.0]^T$$

<br>

#### Line 25: `sigma = np.cov(feats, rowvar=False)`

* **Explanation:** Computes the sample covariance matrix across the features. The parameter `rowvar=False` ensures that each column is treated as a separate variable, yielding a symmetric square matrix of shape `(2048, 2048)`.
* **Mathematical Operation:** Evaluation of $\mathbf{\Sigma}$.
* **Anchor Demonstration:** Using our 2D anchor matrix $\mathbf{F}$ and mean vector $\boldsymbol{\mu}$:

**Variance of Feature 1 ($\Sigma_{1,1}$):**

$$\Sigma_{1,1} = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3 - 1} = \frac{1 + 0 + 1}{2} = 1.0$$

**Variance of Feature 2 ($\Sigma_{2,2}$):**

$$\Sigma_{2,2} = \frac{(4-6)^2 + (5-6)^2 + (9-6)^2}{3 - 1} = \frac{4 + 1 + 9}{2} = 7.0$$

**Covariance between Features 1 and 2 ($\Sigma_{1,2} = \Sigma_{2,1}$):**

$$\Sigma_{1,2} = \frac{(1-2)(4-6) + (2-2)(5-6) + (3-2)(9-6)}{3 - 1} = \frac{(-1)(-2) + (0)(-1) + (1)(3)}{2} = \frac{2 + 0 + 3}{2} = 2.5$$

<br>

This yields the sample covariance matrix:

$$\mathbf{\Sigma} = \begin{bmatrix} 1.0 & 2.5 \\ 2.5 & 7.0 \end{bmatrix}$$

---

## Cell 33: Fréchet Inception Distance Matrix Trace Computation

This cell implements the quantitative evaluation metric: the Fréchet Inception Distance (FID). FID measures the similarity between two multivariate Gaussian distributions fitted to the features of real and generated images. A lower FID score indicates that the generated images have feature distributions closer to the real data, which correlates well with human judgment of visual quality and diversity.

<br>

### Mathematical Formulation

Let $\mathcal{N}(\boldsymbol{\mu}_1, \mathbf{\Sigma}_1)$ and $\mathcal{N}(\boldsymbol{\mu}_2, \mathbf{\Sigma}_2)$ be the two estimated multivariate Gaussian distributions over the 2048-dimensional Inception-v3 feature space. The Fréchet Distance (also known as the Wasserstein-2 distance) between these two distributions is defined as:

$$d^2\left((\boldsymbol{\mu}_1, \mathbf{\Sigma}_1), (\boldsymbol{\mu}_2, \mathbf{\Sigma}_2)\right) = \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|_2^2 + \text{Tr}\left(\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2 - 2\left(\mathbf{\Sigma}_1\mathbf{\Sigma}_2\right)^{1/2}\right)$$

Where:

* $\|\cdot\|_2^2$ is the squared Euclidean norm of the difference between the mean vectors.
* $\text{Tr}(\cdot)$ is the trace operator, which sums the diagonal elements of the resulting matrix.
* $\left(\mathbf{\Sigma}_1\mathbf{\Sigma}_2\right)^{1/2}$ is the unique square root matrix such that $\mathbf{M} = \left(\mathbf{\Sigma}_1\mathbf{\Sigma}_2\right)^{1/2} \implies \mathbf{M}\mathbf{M} = \mathbf{\Sigma}_1\mathbf{\Sigma}_2$.

<br>

### Code-to-Equation Breakdown

#### Line 7: `diff = mu1 - mu2`

* **Explanation:** Computes the displacement vector between the two distribution centroids.
* **Mathematical Operation:** Evaluates $\Delta\boldsymbol{\mu} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$.

<br>

#### Line 8: `covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)`

* **Explanation:** Performs a matrix-level square root operation on the product of the two covariance matrices.
* **Mathematical Operation:** Evaluates $\mathbf{M} = \left(\mathbf{\Sigma}_1\mathbf{\Sigma}_2\right)^{1/2}$.

<br>

#### Line 9 & 10: `if np.iscomplexobj(covmean): covmean = covmean.real`

* **Explanation:** Handles numerical rounding errors. The matrix square root algorithm can produce small imaginary artifacts (e.g., $10^{-16}i$) when processing nearly singular covariance matrices. This step discards the imaginary component to keep the downstream calculations real.
* **Mathematical Operation:** $\mathbf{M} \leftarrow \text{Re}(\mathbf{M})$.

<br>

#### Line 11: `fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)`

* **Explanation:** Evaluates the vector dot product to calculate the squared distance between the means, combines the covariance matrices, and calculates the trace of the final matrix to output the FID score.
* **Mathematical Operation:** Evaluates $d^2 = (\Delta\boldsymbol{\mu})^T(\Delta\boldsymbol{\mu}) + \text{Tr}\left(\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2 - 2\mathbf{M}\right)$.
* **Anchor Demonstration:** Consider two simple 2D distributions ($D=2$).

Let the parameters for Distribution 1 be:

$$\boldsymbol{\mu}_1 = [2.0, 6.0]^T, \quad \mathbf{\Sigma}_1 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$

Let the parameters for Distribution 2 be:

$$\boldsymbol{\mu}_2 = [3.0, 4.0]^T, \quad \mathbf{\Sigma}_2 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$

<br>

**Mean Distance Component:**

$$\Delta\boldsymbol{\mu} = [2.0 - 3.0, 6.0 - 4.0]^T = [-1.0, 2.0]^T$$

$$\|\Delta\boldsymbol{\mu}\|_2^2 = (-1.0)^2 + (2.0)^2 = 1.0 + 4.0 = 5.0$$

<br>

**Covariance Product and Matrix Square Root:**

$$\mathbf{\Sigma}_1\mathbf{\Sigma}_2 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 16.0 \end{bmatrix}$$

$$\mathbf{M} = (\mathbf{\Sigma}_1\mathbf{\Sigma}_2)^{1/2} = \begin{bmatrix} \sqrt{1.0} & 0.0 \\ 0.0 & \sqrt{16.0} \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$

<br>

**Trace Component Calculation:**

$$\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2 - 2\mathbf{M} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix} + \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix} - \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 8.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix}$$

$$\text{Tr}\left(\begin{bmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{bmatrix}\right) = 0.0 + 0.0 = 0.0$$

<br>

**Total Combined FID Score:**

$$\text{FID} = 5.0 + 0.0 = 5.0$$

---

## Cell 34: Multi-Scale NFE Quality-to-Speed Performance Evaluation Benchmark

This cell implements the multi-scale benchmarking logic that compares the performance of the DDPM and Rectified Flow generative paradigms across different inference budgets (Number of Function Evaluations, or NFE). It evaluates how sample generation quality (measured by FID) scales against computational cost, showing how the straight-line paths of Rectified Flow allow for stable integration with fewer steps compared to the curved paths of DDPM.

<br>

### Mathematical Formulation

Let $\mathcal{N} = \{10, 20, 50, 100, 200\}$ represent the discrete set of allowed function evaluation budgets. The benchmark maps each budget value $N \in \mathcal{N}$ to corresponding FID metrics by tracking two main properties:

**1. Iterative Latent Generation Accumulation:**
The generation function splits a total evaluation target of $M = 10000$ samples into independent, non-overlapping batch chunks of size $B_e = 500$. The generation loops are defined as:

$$\mathbf{X}_{\text{DDPM}}^{(N)} = \bigcup_{b=1}^{M / B_e} \Phi_{\text{strided}}\left(B_e, N\right)$$

$$\mathbf{X}_{\text{RF}}^{(N)} = \bigcup_{b=1}^{M / B_e} \Phi_{\text{Euler}}\left(B_e, N\right)$$

Where $\Phi_{\text{strided}}$ uses the strided discrete-time sampler derived in Cell 23, and $\Phi_{\text{Euler}}$ evaluates the continuous ODE velocity fields derived in Cell 29.

**2. Empirical Distribution Evaluation Mapping:**
For each NFE budget setting, the extracted image arrays are passed into the statistics pipeline to yield parametric profiles:

$$\left(\boldsymbol{\mu}_{\text{DDPM}}^{(N)}, \mathbf{\Sigma}_{\text{DDPM}}^{(N)}\right) = \text{ExtractStats}\left(\mathbf{X}_{\text{DDPM}}^{(N)}\right)$$

$$\left(\boldsymbol{\mu}_{\text{RF}}^{(N)}, \mathbf{\Sigma}_{\text{RF}}^{(N)}\right) = \text{ExtractStats}\left(\mathbf{X}_{\text{RF}}^{(N)}\right)$$

The target distances are then evaluated using the Fréchet distance metric derived in Cell 33:

$$\text{FID}_{\text{DDPM}}(N) = d^2\left((\boldsymbol{\mu}_{\text{real}}, \mathbf{\Sigma}_{\text{real}}), (\boldsymbol{\mu}_{\text{DDPM}}^{(N)}, \mathbf{\Sigma}_{\text{DDPM}}^{(N)})\right)$$

$$\text{FID}_{\text{RF}}(N) = d^2\left((\boldsymbol{\mu}_{\text{real}}, \mathbf{\Sigma}_{\text{real}}), (\boldsymbol{\mu}_{\text{RF}}^{(N)}, \mathbf{\Sigma}_{\text{RF}}^{(N)})\right)$$

<br>

### Code-to-Equation Breakdown

#### Line 6: `for _ in tqdm(range(n_total // batch), desc="generating"):`

* **Explanation:** Divides the total sample budget into discrete batch steps to keep memory usage stable during feature extraction.
* **Mathematical Operation:** Sets up the loop for the union collection over $\frac{M}{B_e} = \frac{10000}{500} = 20$ iterations.

<br>

#### Line 20 & 21: Sampler Lambda Function Definitions

```python
ddpm_fn = lambda n, k=nfe: ddpm_sample_strided(model, alpha_bars, betas, n_samples=n, n_steps=k, device=DEVICE)
rf_fn = lambda n, k=nfe: rf_sample(rf_model, n_samples=n, n_steps=k, device=DEVICE)
```

* **Explanation:** Creates anonymous wrapper functions that bind the current NFE loop budget (`nfe`) to the respective model samplers.
* **Mathematical Operation:** Binds $N$ to the step count parameters of $\Phi_{\text{strided}}$ and $\Phi_{\text{Euler}}$.

<br>

#### Line 23 & 24: `ddpm_imgs = generate_samples(ddpm_fn, N_EVAL, EVAL_BATCH)`

* **Explanation:** Loops through the data generator to accumulate the 10,000 generated images for the given NFE budget.
* **Mathematical Operation:** Evaluates the complete dataset tensor $\mathbf{X}^{(N)}$.

<br>

#### Line 29 & 30: `ddpm_fids[nfe] = compute_fid(mu_real, sigma_real, mu_ddpm, s_ddpm)`

* **Explanation:** Calculates the final FID scores by comparing the feature statistics of the generated images against the precomputed real data baseline.
* **Mathematical Operation:** Evaluates $\text{FID}_{\text{DDPM}}(N)$ and $\text{FID}_{\text{RF}}(N)$.
* **Anchor Demonstration:** Consider an NFE budget setting of $N=10$.
	* For DDPM, skipping large time steps introduces discretization errors because the non-linear reverse trajectory deviates from the true path. This can cause the generated features to drift, leading to mismatched statistics and a higher FID score (e.g., $\text{FID}_{\text{DDPM}}(10) \approx 150.0$).
	* For Rectified Flow, the network learns a straight-line velocity field. Because the paths are straight lines, a first-order Euler solver can trace them accurately even with a large step size ($\Delta t = 0.1$). This preserves sample quality with fewer steps, resulting in a lower FID score (e.g., $\text{FID}_{\text{RF}}(10) \approx 45.0$).

---

## Cell 36: Absolute Performance Distance Delta Comparison Matrix

This cell aggregates the empirical results from the multi-scale benchmarking pipeline, organizing the metrics into a clear comparative summary table. It calculates the absolute performance delta between the two frameworks across each NFE budget, quantifying the efficiency advantage of using a straight-line flow matching formulation over a standard curved-trajectory diffusion model.

<br>

### Mathematical Formulation

Let $\mathcal{N} = \{10, 20, 50, 100, 200\}$ represent the discrete set of function evaluation budgets. The performance comparison maps the evaluation metrics via two operations:

**1. Absolute Distance Delta Matrix:**
The performance advantage $\Delta_{\text{FID}}(N)$ for a given NFE budget $N$ is defined as the scalar difference between the two computed Fréchet Inception Distances:

$$\Delta_{\text{FID}}(N) = \text{FID}_{\text{DDPM}}(N) - \text{FID}_{\text{RF}}(N)$$

**2. Metric Sign Interpretation:**
Because the Fréchet Inception Distance maps to a positive semi-definite error metric where a lower value indicates closer alignment with the true data distribution:

$$\text{If } \Delta_{\text{FID}}(N) > 0 \implies \text{FID}_{\text{RF}}(N) < \text{FID}_{\text{DDPM}}(N) \quad (\text{Rectified Flow out-performs DDPM})$$

$$\text{If } \Delta_{\text{FID}}(N) < 0 \implies \text{FID}_{\text{RF}}(N) > \text{FID}_{\text{DDPM}}(N) \quad (\text{DDPM out-performs Rectified Flow})$$

<br>

### Code-to-Equation Breakdown

#### Line 5: `adv = ddpm_fids[nfe] - rf_fids[nfe]`

* **Explanation:** Evaluates the absolute performance difference between the two generation methods for the current NFE budget loop index.
* **Mathematical Operation:** Evaluation of $\Delta_{\text{FID}}(N)$.
* **Anchor Demonstration:** Suppose we trace the loop step where the budget is restricted to $N=10$, using our hypothetical benchmarking parameters from Cell 34 ($\text{FID}_{\text{DDPM}}(10) = 150.00$ and $\text{FID}_{\text{RF}}(10) = 45.00$):

$$\Delta_{\text{FID}}(10) = 150.00 - 45.00 = +105.00$$

The positive sign ($+105.00$) indicates that Rectified Flow achieves a lower error distribution. This difference shrinks at higher budgets (e.g., $N=200$) as the discretization errors of the DDPM sampler decay, causing the two models' scores to converge (e.g., $\Delta_{\text{FID}}(200) \approx +5.00$).

---

## FLOW

```
                  +-----------------------------------+
                  |      Raw CIFAR-10 Dataset         |
                  |     (32x32 Pixel Images)          |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        Data Preprocessing         |
                  |     - Min-Max Scaling to [-1, 1]  |
                  |     - Channel-First Formatting    |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-------------------------+                       +-------------------------+
|  PATH A: DDPM Baseline  |                       | PATH B: Rectified Flow  |
+-------------------------+                       +-------------------------+
| 1. Forward Process:     |                       | 1. Flow Interpolation:  |
|    Corrupt data along   |                       |    Construct straight   |
|    a non-linear,        |                       |    lines between clean  |
|    curved variance      |                       |    data and random      |
|    cosine schedule.     |                       |    Gaussian noise.      |
|                         |                       |                         |
| 2. Optimization:        |                       | 2. Optimization:        |
|    Train a U-Net to     |                       |    Train a U-Net to     |
|    predict the noise    |                       |    predict velocity     |
|    added at a given     |                       |    vectors driving      |
|    discrete step t.     |                       |    the straight path.   |
|                         |                       |                         | 
| 3. Sampling (Reverse):  |                       | 3. Sampling (Reverse):  |
|    Start from noise;    |                       |    Start from noise;    |
|    step backward        |                       |    integrate backwards  |
|    stochastically via   |                       |    deterministically    |
|    Markovian transitions|                       |    using an ODE solver  |
|    (Full or Strided).   |                       |    (Euler method).      |
+-----------+-------------+                       +-----------+-------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-------------------------------------+
                  |        Feature Extraction           |
                  |  - Generate 10,000 images per pipe  |
                  |  - Bilinear upsample (32x32->75x75) |
                  |  - Extract pool3 layer features     |
                  |    using pre-trained Inception-v3   |
                  +-----------------+-------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Statistical Profiling (FID)    |
                  |  - Calculate mean vector (mu)     |
                  |  - Calculate covariance (sigma)   |
                  |  - Compute Fréchet Distance vs    |
                  |    real CIFAR-10 distribution     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       Benchmark Evaluation        |
                  |  Compare Sample Quality (FID) vs  |
                  |  Inference Speed Budget (NFE)     |
                  +-----------------------------------+
```

---

**1. The Starting Point:** The pipeline loads real world images (CIFAR-10) and standardizes their pixels so the math models can process them optimally.

**2. The Training Splits:** The project splits into two completely different philosophical approaches to generating artificial data:

* **DDPM (Path A)**: Destroys images by twisting them slowly into noise along a curved path over 1,000 steps, and trains the network to guess how to untwist them step-by-step.
	
* **Rectified Flow (Path B):** Connects the real image and pure random noise using a perfectly straight directional path, training the network to map out the velocity needed to travel along that line.

**3. The Generation & Checkpoint:** Both models start with completely random static noise and trace their learned trajectories backward to formulate crisp, artificial images.

**4. The Ultimate Test:** To see which method is truly better, we run both sets of artificial images through a standard vision network (Inception-v3) to pull out core features. We mathematically compare these features against real image statistics to output an **FID Score**. A lower score means higher quality, proving which model constructs better images under tighter computation constraints (NFE).

---
---
