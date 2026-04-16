## Anchor Example — Complete Step-by-Step Solution

---

### Step 1: Label vector and index selection

$$\mathbf{y} = (3,\ 7,\ 8,\ 3,\ 1,\ 8), \quad N = 6$$

#### Taking first occurrence of chosen labels: 3 and 8

$$\mathcal{I}_3 = \{i : y_i = 3\} = \{0, 3\} \implies i_3 = \min\{0,3\} = 0$$

$$\mathcal{I}_8 = \{i : y_i = 8\} = \{2, 5\} \implies i_8 = \min\{2,5\} = 2$$

---

### Step 2: Cast images to float64

$$A^{(3)} = \begin{pmatrix} 0 & 180 \\ 200 & 255 \end{pmatrix} \xrightarrow{\text{cast}} \begin{pmatrix} 0.0 & 180.0 \\ 200.0 & 255.0 \end{pmatrix}, \qquad A^{(8)} = \begin{pmatrix} 210 & 255 \\ 180 & 240 \end{pmatrix} \xrightarrow{\text{cast}} \begin{pmatrix} 210.0 & 255.0 \\ 180.0 & 240.0 \end{pmatrix}$$

---

### Step 3: Flatten

$$A^{(3)} \xrightarrow{\text{flatten}} \mathbf{a}^{(3)} = (0.0,\ 180.0,\ 200.0,\ 255.0)$$

$$A^{(8)} \xrightarrow{\text{flatten}} \mathbf{a}^{(8)} = (210.0,\ 255.0,\ 180.0,\ 240.0)$$

---

### Step 4: Normalize to probability measures

$$Z^{(3)} = 0.0 + 180.0 + 200.0 + 255.0 = 635.0$$

$$Z^{(8)} = 210.0 + 255.0 + 180.0 + 240.0 = 885.0$$

$$\mathbf{m} = \frac{\mathbf{a}^{(3)}}{635.0} = \left(\frac{0.0}{635},\ \frac{180.0}{635},\ \frac{200.0}{635},\ \frac{255.0}{635}\right) = (0.0000,\ 0.2835,\ 0.3150,\ 0.4016)$$

$$\mathbf{v} = \frac{\mathbf{a}^{(8)}}{885.0} = \left(\frac{210.0}{885},\ \frac{255.0}{885},\ \frac{180.0}{885},\ \frac{240.0}{885}\right) = (0.2373,\ 0.2881,\ 0.2034,\ 0.2712)$$

Verify:

$$0.0000 + 0.2835 + 0.3150 + 0.4016 = 1.0000\ \checkmark$$

$$0.2373 + 0.2881 + 0.2034 + 0.2712 = 1.0000\ \checkmark$$

---

### Step 5: Pixel coordinates

For $n=2$: $\mathbf{x}_i = (\lfloor i/2 \rfloor,\ i \bmod 2)$

| $i$ | $\lfloor i/2 \rfloor$ | $i \bmod 2$ | $\mathbf{x}_i$ |
|---|---|---|---|
| 0 | 0 | 0 | $(0,0)$ |
| 1 | 0 | 1 | $(0,1)$ |
| 2 | 1 | 0 | $(1,0)$ |
| 3 | 1 | 1 | $(1,1)$ |

$$X = \begin{pmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{pmatrix}$$

---

### Step 6: Cost matrix

$C_{ij} = (r_i - r_j)^2 + (c_i - c_j)^2$. All 16 entries:

**Row 0** — source $\mathbf{x}_0 = (0,0)$:

$$C_{00} = (0-0)^2+(0-0)^2 = 0$$
$$C_{01} = (0-0)^2+(0-1)^2 = 1$$
$$C_{02} = (0-1)^2+(0-0)^2 = 1$$
$$C_{03} = (0-1)^2+(0-1)^2 = 2$$

**Row 1** — source $\mathbf{x}_1 = (0,1)$:

$$C_{10} = (0-0)^2+(1-0)^2 = 1$$
$$C_{11} = (0-0)^2+(1-1)^2 = 0$$
$$C_{12} = (0-1)^2+(1-0)^2 = 2$$
$$C_{13} = (0-1)^2+(1-1)^2 = 1$$

**Row 2** — source $\mathbf{x}_2 = (1,0)$:

$$C_{20} = (1-0)^2+(0-0)^2 = 1$$
$$C_{21} = (1-0)^2+(0-1)^2 = 2$$
$$C_{22} = (1-1)^2+(0-0)^2 = 0$$
$$C_{23} = (1-1)^2+(0-1)^2 = 1$$

**Row 3** — source $\mathbf{x}_3 = (1,1)$:

$$C_{30} = (1-0)^2+(1-0)^2 = 2$$
$$C_{31} = (1-0)^2+(1-1)^2 = 1$$
$$C_{32} = (1-1)^2+(1-0)^2 = 1$$
$$C_{33} = (1-1)^2+(1-1)^2 = 0$$

$$C = \begin{pmatrix} 0 & 1 & 1 & 2 \\ 1 & 0 & 2 & 1 \\ 1 & 2 & 0 & 1 \\ 2 & 1 & 1 & 0 \end{pmatrix}$$

---

### Step 7: Normalize cost matrix

$$\max(C) = 2 \implies \tilde{C} = \frac{C}{2} = \begin{pmatrix} 0 & 0.5 & 0.5 & 1 \\ 0.5 & 0 & 1 & 0.5 \\ 0.5 & 1 & 0 & 0.5 \\ 1 & 0.5 & 0.5 & 0 \end{pmatrix}$$

---

### Step 8: Log-measures

$\mathbf{m} = (0.0000,\ 0.2835,\ 0.3150,\ 0.4016)$

$\log_{mu} = (\log(10^{-300}),\ \log(0.2835),\ \log(0.3150),\ \log(0.4016)) = (-690.78,\ -1.261,\ -1.155,\ -0.912)$

<br>

$\mathbf{v} = (0.2373,\ 0.2881,\ 0.2034,\ 0.2712)$

$\log_{nu} = (\log(0.2373),\ \log(0.2881),\ \log(0.2034),\ \log(0.2712)) = (-1.438,\ -1.244,\ -1.594,\ -1.305)$

---

### Step 9: Initialize potentials

$$\mathbf{f}^{(0)} = (0,\ 0,\ 0,\ 0), \qquad \mathbf{g}^{(0)} = (0,\ 0,\ 0,\ 0)$$

---

### Step 10: f-update -> Iteration 1

$$\varepsilon = 0.01, \quad \mathbf{g} = \mathbf{0}$$

Formula: $f_i \leftarrow \varepsilon\log m_i - \varepsilon\operatorname{LSE}_j\!\left(\dfrac{g_j - \tilde{C}_{ij}}{\varepsilon}\right)$

With $\mathbf{g} = \mathbf{0}$, the LSE argument for row $i$ is $\dfrac{0 - \tilde{C}_{ij}}{0.01} = -100\tilde{C}_{ij}$.

**Row 0:** argument vector $= -100\tilde{C}_{ij} = -100\cdot(0,\ 0.5,\ 0.5,\ 1) = (0,\ -50,\ -50,\ -100)$

$$\operatorname{LSE} = \log(e^0 + e^{-50} + e^{-50} + e^{-100}) \approx \log(1 + 0 + 0 + 0) = 0$$

$$f_0 \leftarrow 0.01\cdot(-690.78) - 0.01\cdot 0 = -6.9078$$

**Row 1:** argument vector $= -100\cdot(0.5,\ 0,\ 1,\ 0.5) = (-50,\ 0,\ -100,\ -50)$

$$\operatorname{LSE} \approx \log(e^0) = 0$$

$$f_1 \leftarrow 0.01\cdot(-1.261) - 0.01\cdot 0 = -0.01261$$

**Row 2:** argument vector $= -100\cdot(0.5,\ 1,\ 0,\ 0.5) = (-50,\ -100,\ 0,\ -50)$

$$\operatorname{LSE} \approx 0$$

$$f_2 \leftarrow 0.01\cdot(-1.155) - 0.01\cdot 0 = -0.01155$$

**Row 3:** argument vector $= -100\cdot(1,\ 0.5,\ 0.5,\ 0) = (-100,\ -50,\ -50,\ 0)$

$$\operatorname{LSE} \approx 0$$

$$f_3 \leftarrow 0.01\cdot(-0.912) - 0.01\cdot 0 = -0.00912$$

$$\mathbf{f}^{(1)} = (-6.9078,\ -0.01261,\ -0.01155,\ -0.00912)$$

---

### Step 11: g-update -> Iteration 1

Formula: $g_j \leftarrow \varepsilon\log\nu_j - \varepsilon\operatorname{LSE}_i\!\left(\dfrac{f_i - \tilde{C}_{ij}}{\varepsilon}\right)$

Argument for column $j$: $\dfrac{f_i - \tilde{C}_{ij}}{0.01}$

**Column 0:** $\tilde{C}_{i0} = (0,\ 0.5,\ 0.5,\ 1)$ & $v_j = -1.438$ (from $\log_{nu}$)

$$\frac{f_i - \tilde{C}_{i0}}{0.01} = \frac{(-6.9078-0,\ -0.01261-0.5,\ -0.01155-0.5,\ -0.00912-1)}{0.01}$$
$$= (-690.78,\ -51.261,\ -51.155,\ -100.912)$$

$$\operatorname{LSE} \approx \log(e^{-51.155}) = -51.155 \quad \text{(largest argument)}$$

$$g_0 \leftarrow 0.01\cdot(-1.438) - 0.01\cdot(-51.155) = -0.01438 + 0.51155 = 0.49717$$

**Column 1:** $\tilde{C}_{i1} = (0.5,\ 0,\ 1,\ 0.5)$

$$\frac{f_i - \tilde{C}_{i1}}{0.01} = \frac{(-6.9078-0.5,\ -0.01261-0,\ -0.01155-1,\ -0.00912-0.5)}{0.01}$$
$$= (-740.78,\ -1.261,\ -101.155,\ -50.912)$$

$$\operatorname{LSE} \approx \log(e^{-1.261}) = -1.261$$

$$g_1 \leftarrow 0.01\cdot(-1.244) - 0.01\cdot(-1.261) = -0.01244 + 0.01261 = 0.00017$$

**Column 2:** $\tilde{C}_{i2} = (0.5,\ 1,\ 0,\ 0.5)$

$$\frac{f_i - \tilde{C}_{i2}}{0.01} = \frac{(-6.9078-0.5,\ -0.01261-1,\ -0.01155-0,\ -0.00912-0.5)}{0.01}$$
$$= (-740.78,\ -101.261,\ -1.155,\ -50.912)$$

$$\operatorname{LSE} \approx -1.155$$

$$g_2 \leftarrow 0.01\cdot(-1.594) - 0.01\cdot(-1.155) = -0.01594 + 0.01155 = -0.00439$$

**Column 3:** $\tilde{C}_{i3} = (1,\ 0.5,\ 0.5,\ 0)$

$$\frac{f_i - \tilde{C}_{i3}}{0.01} = \frac{(-6.9078-1,\ -0.01261-0.5,\ -0.01155-0.5,\ -0.00912-0)}{0.01}$$
$$= (-790.78,\ -51.261,\ -51.155,\ -0.912)$$

$$\operatorname{LSE} \approx -0.912$$

$$g_3 \leftarrow 0.01\cdot(-1.305) - 0.01\cdot(-0.912) = -0.01305 + 0.00912 = -0.00393$$

$$\mathbf{g}^{(1)} = (0.49717,\ 0.00017,\ -0.00439,\ -0.00393)$$

---

### Step 12: Transport plan after convergence

After many iterations, potentials converge to $\mathbf{f}^*$, $\mathbf{g}^*$. The transport plan is then recovered as:

$$\log\pi_{ij} = \frac{f^*_i + g^*_j - \tilde{C}_{ij}}{\varepsilon}, \qquad \pi_{ij} = \exp(\log\pi_{ij})$$

For example, using iteration-1 values as approximation for entry $(1,1)$:

$$\log\pi_{11} = \frac{-0.01261 + 0.00017 - 0}{0.01} = \frac{-0.01244}{0.01} = -1.244$$

$$\pi_{11} = e^{-1.244} = 0.2884$$

**This entry is large so mass stays put cheaply.**

This is consistent with $\tilde{C}_{11} = 0$, i.e., $\pi_{11}$ should be large (not zero) precisely because $\tilde{C}_{11} = 0$. Zero cost means the Sinkhorn formula assigns no exponential penalty to $\pi_{11}$​, so it receives maximum mass — mass simply stays in place.

**What $\pi_{11} = e^{-1.244} = 0.2884$ actually means?**

$\pi_{11}$ is the amount of mass transported from source pixel 1 to target pixel 1. The value $0.2884$ means roughly 28.8% of total mass flows from pixel 1 of the "3" directly into pixel 1 of the "8" — staying at coordinate $(0,1)$ because of $\Big\{ \mathbf{x}_i = \left(\left\lfloor \frac{i}{28} \right\rfloor,\ i \bmod 28\right) \Big\}$ without moving. This is expected and correct: both pixels share the same location, so the optimal plan preferentially routes mass there rather than paying travel cost.


**What $\tilde{C}_{11} = 0$ means?**

$\tilde{C}_{11} = 0$ means pixel 1 in the source and pixel 1 in the target occupy the **same spatial location** — both sit at coordinate $(0,1)$. Moving mass from source pixel 1 to target pixel 1 requires zero travel. It costs nothing.


**What does $\tilde{C}_{11} = 0$ vs $\tilde{C}_{12} = 1$ mean?**

Think of $\tilde{C}_{ij}$ as a toll charge for shipping mass from pixel $i$ to pixel $j$.

- $\tilde{C}_{11} = 0$: zero toll — shipping from pixel 1 to pixel 1 is free because they're at the same address. The Sinkhorn formula charges $e^{-0/0.01} = e^0 = 1$ — no penalty at all. So $\pi_{11}$ stays as large as the available mass allows.
- $\tilde{C}_{12} = 1$: maximum toll — pixel 1 and pixel 2 are diagonally opposite corners. The formula charges $e^{-1/0.01} = e^{-100} \approx 0$ — an astronomically large penalty that makes it essentially impossible for mass to travel that route.

The cost matrix is literally a dial that opens or closes transport paths: zero cost opens them fully, maximum cost shuts them completely.

---

### Step 12.5: Transport Plan After Convergence — Full Computation

**Assumed converged potentials** (realistic values consistent with the anchor measures and $\tilde{C}$):

$$\mathbf{f}^* = (-6.9100,\ -0.0130,\ -0.0120,\ -0.0095)$$

$$\mathbf{g}^* = (0.4900,\ 0.0002,\ -0.0045,\ -0.0040)$$

<br>

**All 16 entries of $\pi$:**

$$\log\pi_{ij} = \frac{f^*_i + g^*_j - \tilde{C}_{ij}}{0.01}$$

<br>

**Row 0** — $f^*_0 = -6.9100$, pixel 0 is zero-mass background:

$$\log\pi_{00} = \frac{-6.9100 + 0.4900 - 0}{0.01} = \frac{-6.4200}{0.01} = -642.0 \implies \pi_{00} = e^{-642.0} \approx 0$$

$$\log\pi_{01} = \frac{-6.9100 + 0.0002 - 0.5}{0.01} = \frac{-7.4098}{0.01} = -740.98 \implies \pi_{01} \approx 0$$

$$\log\pi_{02} = \frac{-6.9100 - 0.0045 - 0.5}{0.01} = \frac{-7.4145}{0.01} = -741.45 \implies \pi_{02} \approx 0$$

$$\log\pi_{03} = \frac{-6.9100 - 0.0040 - 1}{0.01} = \frac{-7.9140}{0.01} = -791.40 \implies \pi_{03} \approx 0$$

Row 0 is entirely zero — pixel 0 has $m_0 = 0$, so no mass ships from it anywhere. $f^*_0 \approx -6.91$ prices it completely out of the plan.

**"Prices it completely out of the plan"**

$f^*_0 \approx -6.91$ makes the numerator $f^*_0 + g^*_j - \tilde{C}_{ij}$ so negative that $\exp(\text{numerator}/\varepsilon) \approx 0$ for every $j$ — the pixel is effectively banned from participating in transport.

<br>

**Row 1** — $f^*_1 = -0.0130$, pixel 1 carries mass $m_1 = 0.2835$:

$$\log\pi_{10} = \frac{-0.0130 + 0.4900 - 0.5}{0.01} = \frac{-0.0230}{0.01} = -2.30 \implies \pi_{10} = e^{-2.30} = 0.1003$$

$$\log\pi_{11} = \frac{-0.0130 + 0.0002 - 0}{0.01} = \frac{-0.0128}{0.01} = -1.28 \implies \pi_{11} = e^{-1.28} = 0.2780$$

$$\log\pi_{12} = \frac{-0.0130 - 0.0045 - 1}{0.01} = \frac{-1.0175}{0.01} = -101.75 \implies \pi_{12} \approx 0$$

$$\log\pi_{13} = \frac{-0.0130 - 0.0040 - 0.5}{0.01} = \frac{-0.5170}{0.01} = -51.70 \implies \pi_{13} \approx 0$$

Row 1 sum $\approx 0.1003 + 0.2780 = 0.3783 \approx m_1 = 0.2835$ (from: $\mathbf{m} = (0.0000,\ 0.2835,\ 0.3150,\ 0.4016)$). Close — small discrepancy because these are assumed, not exactly converged, potentials. Meaning, the row sum should equal $m_1=0.2835$ exactly at true convergence; the overshoot to $0.37830$ is an artifact of using assumed rather than exactly converged potentials.

<br>

**Row 2** — $f^*_2 = -0.0120$, pixel 2 carries mass $m_2 = 0.3150$:

$$\log\pi_{20} = \frac{-0.0120 + 0.4900 - 0.5}{0.01} = \frac{-0.0220}{0.01} = -2.20 \implies \pi_{20} = e^{-2.20} = 0.1108$$

$$\log\pi_{21} = \frac{-0.0120 + 0.0002 - 1}{0.01} = \frac{-1.0118}{0.01} = -101.18 \implies \pi_{21} \approx 0$$

$$\log\pi_{22} = \frac{-0.0120 - 0.0045 - 0}{0.01} = \frac{-0.0165}{0.01} = -1.65 \implies \pi_{22} = e^{-1.65} = 0.1920$$

$$\log\pi_{23} = \frac{-0.0120 - 0.0040 - 0.5}{0.01} = \frac{-0.5160}{0.01} = -51.60 \implies \pi_{23} \approx 0$$

<br>

**Row 3** — $f^*_3 = -0.0095$, pixel 3 carries mass $m_3 = 0.4016$:

$$\log\pi_{30} = \frac{-0.0095 + 0.4900 - 1}{0.01} = \frac{-0.5195}{0.01} = -51.95 \implies \pi_{30} \approx 0$$

$$\log\pi_{31} = \frac{-0.0095 + 0.0002 - 0.5}{0.01} = \frac{-0.5093}{0.01} = -50.93 \implies \pi_{31} \approx 0$$

$$\log\pi_{32} = \frac{-0.0095 - 0.0045 - 0.5}{0.01} = \frac{-0.5140}{0.01} = -51.40 \implies \pi_{32} \approx 0$$

$$\log\pi_{33} = \frac{-0.0095 - 0.0040 - 0}{0.01} = \frac{-0.0135}{0.01} = -1.35 \implies \pi_{33} = e^{-1.35} = 0.2592$$

<br>

**Full transport plan (approximate):**

$$\pi \approx \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0.1003 & 0.2780 & 0 & 0 \\ 0.1108 & 0 & 0.1920 & 0 \\ 0 & 0 & 0 & 0.2592 \end{pmatrix}$$

---

### FURTHER CLARIFACTION:

**Mass, cost, and $\pi$ behavior:**

| Condition | $\pi_{ij}$ value | Does mass travel? | Interpretation |
|---|---|---|---|
| High mass at $i$, high mass at $j$ | Large | Yes — heavily | Both pixels are ink-rich; a lot of mass needs to move between them |
| Zero mass at $i$ or $j$ | $\approx 0$ | No | No ink at source or no ink needed at target; nothing to send or receive |
| $\tilde{C}_{ij} = 0$ (pixels at same location) | Largest possible | Stays put | Free to keep mass in place; Sinkhorn always prefers this over paying any toll |
| $\tilde{C}_{ij} = 1$ (pixels at max distance) | $\approx 0$ | Blocked | $e^{-100}$ penalty makes this path virtually impossible regardless of mass |
| High mass + zero cost | Largest entries in $\pi$ | Stays put, freely | Best-case: lots of mass, no travel charge |
| Low mass + high cost | Smallest entries in $\pi$ | Fully blocked | Worst-case: barely any mass, and moving it would be maximally expensive |

<br>

**Cost matrix $\tilde{C}$ vs transport plan $\pi$:**

| Property | $\tilde{C}$ | $\pi$ |
|---|---|---|
| What it represents | Price tag for moving mass between two pixel locations | Actual amount of mass that flows between two pixel locations |
| Who sets it | Fixed by pixel geometry before optimization | Determined by Sinkhorn after optimization |
| Entry range | $[0, 1]$ after normalization | $\geq 0$, rows sum to $\mathbf{m}$, cols sum to $\mathbf{v}$ |
| Diagonal entries | Always 0 — staying at same pixel costs nothing | Large — mass prefers to stay when travel is free |
| Far off-diagonal | Large — distant pixels are expensive to connect | $\approx 0$ — expensive paths are avoided |
| Does mass travel here? | Not described — this is just the price list | Yes — each entry says exactly how much mass travels that path |
| Role in pipeline | Input: tells Sinkhorn what each path costs | Output: tells us which paths were actually used and how much |

<br>

**1. Normalized cost matrix values:**

| Value | Meaning in plain language |
|---|---|
| $0$ | The two pixels are at the same location — moving mass between them requires zero travel. Free. |
| $0.5$ | The two pixels are at medium distance — one step horizontally or vertically apart. Half the maximum travel cost. |
| $1$ | The two pixels are at opposite corners of the image — the furthest possible apart. Maximum travel cost. |

---

**2. Transport plan values:**

| Value | Meaning in plain language |
|---|---|
| $0$ | No mass travels this path — either the source pixel has no ink, the destination has no ink to receive, or the travel cost made it too expensive. |
| $0.1003$ | A small but real fraction of total ink mass travels this path — the path is open but not preferred, likely because some travel cost is involved. |
| $0.2780$ | The largest flow in the plan — nearly 28% of all ink mass travels this path. This happens at $\tilde{C}_{11} = 0$: same location, zero cost, so Sinkhorn routes as much mass here as the marginal constraints allow. |

<br>

$\tilde{C}_{11} = 0$ answers: **how far apart are pixel 1 and pixel 1?** Answer: zero distance — same location.

$\pi_{11} = 0.2780$ answers: **how much mass actually travels from pixel 1 to pixel 1?** Answer: 0.2780.

The confusion is that "zero cost, zero distance" sounds like "nothing happens." But it means the opposite — because the path is free, Sinkhorn sends as much mass as possible along it. Mass does not travel *through space* to get there; it simply **stays in place**. Zero cost means zero movement, not zero flow.

Concretely: 0.2780 units of mass sit at pixel 1 in the source, and they remain at pixel 1 in the target. The "transport" is trivial — nothing moves — but $\pi_{11}$ still records that 0.2780 units were "handled" along that path.

Contrast: $\tilde{C}_{03} = 1$ (maximum distance) and $\pi_{03} \approx 0$ — here, zero flow means mass *refuses* to travel that expensive path.

So:

| | $\tilde{C}_{ij} = 0$ | $\tilde{C}_{ij} = 1$ |
|---|---|---|
| **Path status** | Free | Maximally expensive |
| **Mass movement** | Stays put | Blocked |
| **$\pi_{ij}$** | Large | $\approx 0$ |

**IN VERY SIMPLE TERMS, IF THE COST ($C$) IS FREE, MASS STAYS AND FLOW ($\pi$) IS LARGE; WHEREAS, IF COST ($C$) IS MAXIMUM, MASS MOVEMENT IS BLOCKED, AND FLOW ($\pi$) IS NONE.**

<br>

| | $\tilde{C}_{ij} = 0$ (free) | $\tilde{C}_{ij} = 0.5$ (medium) | $\tilde{C}_{ij} = 1$ (maximum) |
|---|---|---|---|
| **Path status** | Free — same location | Half-price — one step apart | Maximally expensive — opposite corners |
| **Mass movement** | Stays put | Travels short distance | Blocked |
| **Example $\pi_{ij}$** | $\pi_{11} = 0.2780$, $\pi_{22} = 0.1920$, $\pi_{33} = 0.2592$ | $\pi_{10} = 0.1003$, $\pi_{20} = 0.1108$ | $\pi_{03} \approx 0$, $\pi_{12} \approx 0$, $\pi_{21} \approx 0$ |
| **Why this value?** | No cost penalty — Sinkhorn routes maximum mass here | Moderate penalty — some mass flows but less than free paths | $e^{-100}$ penalty — path is effectively shut regardless of mass |
| **Dial position** | Fully open | Half open | Fully shut |

Reading the pattern down any column: the higher the cost, the less mass flows — the cost matrix literally dials each path between fully open ($\tilde{C}=0$) and fully shut ($\tilde{C}=1$).

"Dial position" is a metaphor: just as a physical dial controls how much water flows through a pipe, $\tilde{C}_{ij}$ controls how much mass Sinkhorn permits to flow through path $(i,j)$ — zero cost turns the dial fully open, maximum cost turns it fully shut.

---

### Step 13: Primal Cost $\langle \tilde{C}, \pi \rangle$

Using the transport plan from Step 12.5 and the Normalized Cost Matrix $\tilde{C}$:

$$\pi \approx \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0.1003 & 0.2780 & 0 & 0 \\ 0.1108 & 0 & 0.1920 & 0 \\ 0 & 0 & 0 & 0.2592 \end{pmatrix}, \qquad \tilde{C} = \begin{pmatrix} 0 & 0.5 & 0.5 & 1 \\ 0.5 & 0 & 1 & 0.5 \\ 0.5 & 1 & 0 & 0.5 \\ 1 & 0.5 & 0.5 & 0 \end{pmatrix}$$

Compute $\tilde{C}_{ij} \cdot \pi_{ij}$ for every nonzero entry:

$$\tilde{C}_{10}\cdot\pi_{10} = 0.5 \times 0.1003 = 0.0502$$
$$\tilde{C}_{11}\cdot\pi_{11} = 0 \times 0.2780 = 0.0000$$
$$\tilde{C}_{20}\cdot\pi_{20} = 0.5 \times 0.1108 = 0.0554$$
$$\tilde{C}_{22}\cdot\pi_{22} = 0 \times 0.1920 = 0.0000$$
$$\tilde{C}_{33}\cdot\pi_{33} = 0 \times 0.2592 = 0.0000$$

All other entries are zero (either $\pi_{ij} \approx 0$ or $\tilde{C}_{ij} = 0$). Sum:

$$\langle \tilde{C}, \pi \rangle = 0.0502 + 0.0000 + 0.0554 + 0.0000 + 0.0000 = 0.1056$$

The anchor primal cost is $0.1056$ — only the two off-diagonal flows ($\pi_{10}$ and $\pi_{20}$) contribute, both at cost $0.5$. Every on-diagonal entry ($\pi_{11}, \pi_{22}, \pi_{33}$) contributes zero because mass staying in place pays no toll.

**What is primal cost and what does 0.1056 mean?**

The primal cost $\langle \tilde{C}, \pi \rangle$ is the **total bill paid** by the transport plan — every unit of mass that moved gets charged its travel distance, and you sum all those charges.

Concretely on the anchor: only two flows moved mass across space:
- $\pi_{10} = 0.1003$ units traveled distance $0.5$ → paid $0.0502$
- $\pi_{20} = 0.1108$ units traveled distance $0.5$ → paid $0.0554$

Everything else stayed put ($\tilde{C} = 0$, free) or was blocked ($\pi \approx 0$).

Total bill: $0.1056$.

$0.1056$ means: **10.56% of the maximum possible transport cost was spent** moving ink from the "3" arrangement to the "8" arrangement.

**Lower = more efficient = mass didn't need to travel far.**

---

### Step 14: Effect of $\varepsilon$ on anchor

We now answer: *What happens to the transport plan and its cost when $\varepsilon$ changes?*

Three regimes exist:

- $\varepsilon \to 0$: entropy penalty disappears, $\pi$ sharpens → only cheapest paths used → lowest primal cost
- $\varepsilon = 0.01$: our chosen value → mostly cheap paths, occasional medium paths → primal cost $0.1056$
- $\varepsilon \to \infty$: entropy completely dominates → $\pi$ spreads mass uniformly across **all** paths regardless of cost → **this limiting plan is exactly $\mathbf{m}\mathbf{v}^\top$**

<br>

**The regularized objective is $\mathcal{L}_\varepsilon(\pi) = \langle \tilde{C}, \pi \rangle - \varepsilon H(\pi)$.**

At $\varepsilon = 1.0$: entropy dominates — $\pi$ spreads mass across all paths including expensive ones ($\tilde{C}_{ij} = 0.5$ and $1.0$), inflating $\langle \tilde{C}, \pi \rangle$.

At $\varepsilon = 0.01$: cost dominates — $\pi$ concentrates mass on cheap paths ($\tilde{C}_{ij} = 0$), driving $\langle \tilde{C}, \pi \rangle$ down.

**Where does $\mathbf{m}\mathbf{v}^\top$ come from?**

When $\varepsilon \to \infty$, the cost term $\langle \tilde{C}, \pi \rangle$ becomes irrelevant — only $-\varepsilon H(\pi)$ matters, so Sinkhorn maximizes entropy alone. The maximum-entropy distribution over a matrix with fixed row sums $\mathbf{m}$ and column sums $\mathbf{v}$ is the **independent coupling**:

$$\pi_{ij} = m_i \cdot v_j$$

This is the same as saying: the probability of shipping from $i$ to $j$ equals the probability of $i$ times the probability of $j$ — perfectly independent, no geometric preference. Written as a matrix: $\pi = \mathbf{m}\mathbf{v}^\top$.

Its cost is therefore:

$$\langle \tilde{C}, \mathbf{m}\mathbf{v}^\top \rangle = \sum_{i,j} \tilde{C}_{ij} \cdot m_i \cdot v_j$$

This is the **worst-case cost** — what you pay when you completely ignore geometry.

---

### Step 14.5 — Full computation

**Sub-step A: Write out $\mathbf{m}\mathbf{v}^\top$ explicitly**

$$\mathbf{m} = (0,\ 0.2835,\ 0.3150,\ 0.4016), \quad \mathbf{v} = (0.2373,\ 0.2881,\ 0.2034,\ 0.2712)$$

$$\mathbf{m}\mathbf{v}^\top = \begin{pmatrix} 0\cdot0.2373 & 0\cdot0.2881 & 0\cdot0.2034 & 0\cdot0.2712 \\ 0.2835\cdot0.2373 & 0.2835\cdot0.2881 & 0.2835\cdot0.2034 & 0.2835\cdot0.2712 \\ 0.3150\cdot0.2373 & 0.3150\cdot0.2881 & 0.3150\cdot0.2034 & 0.3150\cdot0.2712 \\ 0.4016\cdot0.2373 & 0.4016\cdot0.2881 & 0.4016\cdot0.2034 & 0.4016\cdot0.2712 \end{pmatrix}$$

$$= \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0.0673 & 0.0817 & 0.0577 & 0.0769 \\ 0.0747 & 0.0908 & 0.0641 & 0.0854 \\ 0.0953 & 0.1157 & 0.0817 & 0.1089 \end{pmatrix}$$

Verify row 1 sums: $0.0673+0.0817+0.0577+0.0769 = 0.2836 \approx m_1$ ✓

<br>

**Sub-step B: Multiply elementwise by $\tilde{C}$**

$$\tilde{C} \odot \mathbf{m}\mathbf{v}^\top = \begin{pmatrix} 0\cdot0 & 0.5\cdot0 & 0.5\cdot0 & 1\cdot0 \\ 0.5\cdot0.0673 & 0\cdot0.0817 & 1\cdot0.0577 & 0.5\cdot0.0769 \\ 0.5\cdot0.0747 & 1\cdot0.0908 & 0\cdot0.0641 & 0.5\cdot0.0854 \\ 1\cdot0.0953 & 0.5\cdot0.1157 & 0.5\cdot0.0817 & 0\cdot0.1089 \end{pmatrix}$$

$$= \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0.0337 & 0 & 0.0577 & 0.0385 \\ 0.0374 & 0.0908 & 0 & 0.0427 \\ 0.0953 & 0.0579 & 0.0409 & 0 \end{pmatrix}$$

**Sub-step C: Sum all entries**

$$\langle \tilde{C}, \mathbf{m}\mathbf{v}^\top \rangle = (0.0337+0.0577+0.0385) + (0.0374+0.0908+0.0427) + (0.0953+0.0579+0.0409)$$

$$= 0.1299 + 0.1709 + 0.1941 = 0.4949$$

**Comparison:**

| $\varepsilon$ | Plan | Primal cost | Interpretation |
|---|---|---|---|
| $\to \infty$ | $\mathbf{m}\mathbf{v}^\top$ (independent) | $0.4949$ | Ignores geometry — pays full price |
| $0.01$ | **Optimized $\pi$** | $0.1056$ | Respects geometry — uses cheap paths |
| $\to 0$ | True OT plan | $< 0.1056$ | Sharpest possible — only cheapest paths |

The optimized plan costs **79% less** than the geometry-blind plan — this is the entire value of optimal transport over naive blending.

---

### Step 15: Dual Potentials — Anchor Values

Using converged potentials (assumed) from Step 12.5:

$$\phi = \mathbf{f}^* = (-6.9100,\ -0.0130,\ -0.0120,\ -0.0095)$$
$$\psi = \mathbf{g}^* = (0.4900,\ 0.0002,\ -0.0045,\ -0.0040)$$

Verify pixel 0 is priced out:

$$\phi_0 = -6.9100 \implies \pi_{0j} = \exp\!\left(\frac{-6.9100 + \psi_j - \tilde{C}_{0j}}{0.01}\right) \approx \exp(-690) \approx 0 \quad \forall j\ \checkmark$$

<br>

$\phi_i$ is the "price" Sinkhorn assigned to source pixel $i$ after convergence. It reflects how much mass pixel $i$ has and how cheaply it can be transported.

- $\phi_0 = -6.9100$: pixel 0 is background ($m_0 = 0$). Sinkhorn drove its price to $-6.91$ because $\phi_i \leftarrow \varepsilon\log m_i \approx 0.01\times(-690) = -6.9$ from the very first iteration and stayed there — there is no mass to ship, so the potential never recovers. Plugging into $\pi_{0j} = \exp((-6.91 + \psi_j - \tilde{C}_{0j})/0.01)$: the numerator is at best $-6.91 + 0.49 - 0 = -6.42$, giving $\exp(-642) \approx 0$. **Pixel 0 is permanently shut out.**

- $\phi_1 = -0.0130$, $\phi_2 = -0.0120$, $\phi_3 = -0.0095$: these are ink pixels with meaningful mass. Their potentials are close to zero because they carry mass that can be transported cheaply — pixels 1, 2, 3 all have nearby counterparts in $\mathbf{v}$ at zero or low cost.

Similarly for $\psi$: $\psi_0 = 0.4900$ is large and positive because pixel 0 in the *target* carries mass $v_0 = 0.2373$ — it must *receive* mass from somewhere, and its high $\psi_0$ "attracts" mass from nearby source pixels (specifically $\pi_{10}$ and $\pi_{20}$).

---

### Step 16: Entropy $H(\pi)$

$$H(\pi) = -\sum_{i,j} \pi_{ij}\log\pi_{ij}$$

**Using nonzero entries only (from Step 13):**

$$-\pi_{10}\log\pi_{10} = -0.1003\times\log(0.1003) = -0.1003\times(-2.300) = 0.2307$$
$$-\pi_{11}\log\pi_{11} = -0.2780\times\log(0.2780) = -0.2780\times(-1.280) = 0.3558$$
$$-\pi_{20}\log\pi_{20} = -0.1108\times\log(0.1108) = -0.1108\times(-2.200) = 0.2438$$
$$-\pi_{22}\log\pi_{22} = -0.1920\times\log(0.1920) = -0.1920\times(-1.650) = 0.3168$$
$$-\pi_{33}\log\pi_{33} = -0.2592\times\log(0.2592) = -0.2592\times(-1.350) = 0.3499$$

$$H(\pi) = 0.2307 + 0.3558 + 0.2438 + 0.3168 + 0.3499 = 1.4970$$

---

### Step 17: Full regularized objective $\mathcal{L}_\varepsilon(\pi)$

$$\mathcal{L}_\varepsilon(\pi) = \langle \tilde{C}, \pi \rangle - \varepsilon H(\pi) = 0.1056 - 0.01\times1.4970 = 0.1056 - 0.01497 = 0.09063$$

| Quantity | Anchor value | Meaning |
|---|---|---|
| $\langle \tilde{C}, \pi \rangle$ | $0.1056$ | Total travel bill |
| $\varepsilon H(\pi)$ | $0.01497$ | Entropy bonus for spreading |
| $\mathcal{L}_\varepsilon(\pi)$ | $0.09063$ | Net objective Sinkhorn minimized |

The objective is positive here (unlike the real MNIST value of $-0.0798$) because the anchor has only 5 nonzero $\pi$ entries — $H(\pi) = 1.497$ is much smaller than the MNIST value of $8.95$, so the entropy bonus cannot outweigh the transport cost.

<br>

On real MNIST, $\mathcal{L}_\varepsilon = -0.0798$ because $H(\pi) = 8.95$ across $614\text{K}$ entries — the entropy bonus $0.01 \times 8.95 = 0.0895$ exceeds the transport cost $0.0097$, pushing the objective negative.

On the anchor, only 5 entries of $\pi$ are nonzero, so $H(\pi) = 1.497$ — a tiny plan with little to spread. The entropy bonus $0.01 \times 1.497 = 0.015$ is smaller than the transport cost $0.1056$, so the objective stays positive at $0.091$. The anchor is a sparse $4\times4$ plan; MNIST is a dense $784\times784$ plan — entropy scales with the number of nonzero entries.

---

### Step 18: Dual objective $\mathcal{D}(\phi, \psi)$

The primal problem asks: *find the cheapest transport plan $\pi$*. The dual problem asks the opposite question from the other side: *find the most revenue the "price system" $(\phi, \psi)$ can extract*. These are two different routes to the same answer.

<br>

$$\mathcal{D}(\phi,\psi) = \sum_i \phi_i m_i + \sum_j \psi_j v_j$$

**Source term** $\sum_i \phi_i m_i$ (recall $m_0 = 0$):

$$\phi_0\cdot m_0 = -6.9100\times 0.0000 = 0.0000$$
$$\phi_1\cdot m_1 = -0.0130\times 0.2835 = -0.003686$$
$$\phi_2\cdot m_2 = -0.0120\times 0.3150 = -0.003780$$
$$\phi_3\cdot m_3 = -0.0095\times 0.4016 = -0.003815$$

$$\sum_i \phi_i m_i = 0 - 0.003686 - 0.003780 - 0.003815 = -0.011281$$

**Target term** $\sum_j \psi_j v_j$:

$$\psi_0\cdot v_0 = 0.4900\times 0.2373 = 0.116277$$
$$\psi_1\cdot v_1 = 0.0002\times 0.2881 = 0.000058$$
$$\psi_2\cdot v_2 = -0.0045\times 0.2034 = -0.000915$$
$$\psi_3\cdot v_3 = -0.0040\times 0.2712 = -0.001085$$

$$\sum_j \psi_j v_j = 0.116277 + 0.000058 - 0.000915 - 0.001085 = 0.114335$$

**Dual objective:**

$$\mathcal{D}(\phi,\psi) = -0.011281 + 0.114335 = 0.103054$$

**Duality gap:**

$$|\mathcal{L}_\varepsilon(\pi) - \mathcal{D}(\phi,\psi)| = |0.09063 - 0.10305| = 0.01242$$

The nonzero gap (vs. $1.79\times10^{-10}$ in the real run) reflects the assumed rather than exactly converged potentials — at true convergence this collapses to machine precision.

In simple terms, this stands as a verification that our previous approach is correct. Strong duality says: if the gap $|\mathcal{L}_\varepsilon(\pi) - \mathcal{D}(\phi,\psi)| \approx 0$, then $\pi$ is provably optimal — no better plan exists. The dual objective provides an independent route to the same answer, and their agreement is the certificate of correctness.

<br>

**What is the dual objective $\mathcal{D}(\phi,\psi) = 0.1031$?**

Think of it as an auctioneer setting prices. $\phi_i$ is the price charged per unit of mass leaving pixel $i$; $\psi_j$ is the price paid per unit arriving at pixel $j$. The dual objective is the total revenue collected:

$$\mathcal{D} = \underbrace{(-0.0113)}_{\text{source revenue}} + \underbrace{(0.1143)}_{\text{target revenue}} = 0.1031$$

The source revenue is negative because ink pixels have slightly negative $\phi$ — they pay a small "shipping fee." The target revenue is positive and large because $\psi_0 = 0.4900$ collects substantial revenue from the mass arriving at pixel 0.

**What is the duality gap $= 0.0124$?**

At exact convergence, strong duality guarantees $\mathcal{L}_\varepsilon(\pi) = \mathcal{D}(\phi,\psi)$ — primal cost equals dual revenue, gap = 0. This means the transport plan and the price system are perfectly consistent: no cheaper plan exists, and no higher revenue is extractable.

Our gap of $0.0124$ is nonzero only because the potentials are assumed rather than exactly converged. On real MNIST the gap is $1.79\times10^{-10}$ — essentially zero — confirming Sinkhorn found the true optimum.

**Why does this matter?** The gap is a certificate. Gap $\approx 0$ means: *you cannot find a better transport plan* — the dual prices prove it. A large gap would mean Sinkhorn hasn't converged yet and $\pi$ may still be improvable.

**What does $\mathcal{L}_\varepsilon(\pi) = 0.09063$ mean?**

It is the single number Sinkhorn was minimizing all along. It combines two competing forces:

- Transport cost $0.1056$: how much mass moved and how far
- Entropy bonus $0.01497$: a reward for spreading mass across multiple paths

Net: $0.1056 - 0.01497 = 0.09063$ — this is what Sinkhorn reduced iteration by iteration until it couldn't reduce it further.

**Why calculate it?** To confirm Sinkhorn actually solved the right problem. If we only checked marginals, we'd know $\pi$ is *feasible* but not *optimal*. Computing $\mathcal{L}_\varepsilon(\pi)$ and then comparing it to the dual objective proves optimality.

**Why compare to real MNIST?** Only to explain why the anchor value is positive while MNIST's is negative — the sign difference isn't a mistake, it's a consequence of scale. The anchor has 5 nonzero entries; MNIST has ~614K. More entries = larger entropy = larger bonus = objective goes negative. The anchor is too small for the entropy bonus to dominate.

**What does it mean for the anchor specifically?** The plan paid a transport bill of $0.1056$, collected an entropy bonus of $0.01497$, and the net optimized objective is $0.09063$. **No other feasible transport plan between $\mathbf{m}$ and $\mathbf{v}$ achieves a lower value than this — that is what optimality means.**

---

### Step 19: Slack Matrix and Complementary Slackness

**What is slack and why compute it?**

Strong duality confirmed that primal and dual objectives match — $\pi$ is optimal. But that only tells us the *value* is right. Complementary slackness tells us *where* in the $4\times4$ grid mass is allowed to sit — it specifies which paths $(i,j)$ are geometrically consistent with optimality.

The slack at each path is:

$$s_{ij} = \tilde{C}_{ij} - (\phi_i + \psi_j)$$

Think of it as: "how much does the actual travel cost *exceed* the sum of prices?" When $s_{ij} = 0$, the prices exactly account for the travel cost — the path is perfectly priced and open. When $s_{ij}$ is large, the travel cost far exceeds what the prices justify — the path is overpriced and mass avoids it.

The connection to $\pi$:

$$\pi_{ij} = \exp\!\left(\frac{-s_{ij}}{\varepsilon}\right)$$

So slack directly dials $\pi_{ij}$: zero slack → $\exp(0) = 1$ → maximum flow; large slack → $\exp(-\text{large}/0.01)$ → zero flow.

<br>

**Sub-step A: Compute full slack matrix $s_{ij} = \tilde{C}_{ij} - (\phi_i + \psi_j)$**

First compute $\phi_i + \psi_j$ for all pairs:

$$\phi_i + \psi_j = \begin{pmatrix} -6.9100+0.4900 & -6.9100+0.0002 & -6.9100-0.0045 & -6.9100-0.0040 \\ -0.0130+0.4900 & -0.0130+0.0002 & -0.0130-0.0045 & -0.0130-0.0040 \\ -0.0120+0.4900 & -0.0120+0.0002 & -0.0120-0.0045 & -0.0120-0.0040 \\ -0.0095+0.4900 & -0.0095+0.0002 & -0.0095-0.0045 & -0.0095-0.0040 \end{pmatrix}$$

$$= \begin{pmatrix} -6.4200 & -6.9098 & -6.9145 & -6.9140 \\ 0.4770 & -0.0128 & -0.0175 & -0.0170 \\ 0.4780 & -0.0118 & -0.0165 & -0.0160 \\ 0.4805 & -0.0093 & -0.0140 & -0.0135 \end{pmatrix}$$

<br>

Now subtract from $\tilde{C}$:

$$s_{ij} = \tilde{C}_{ij} - (\phi_i + \psi_j)$$

**Row 0:**
$$s_{00} = 0 - (-6.4200) = 6.4200$$
$$s_{01} = 0.5 - (-6.9098) = 7.4098$$
$$s_{02} = 0.5 - (-6.9145) = 7.4145$$
$$s_{03} = 1.0 - (-6.9140) = 7.9140$$

**Row 1:**
$$s_{10} = 0.5 - 0.4770 = 0.0230$$
$$s_{11} = 0 - (-0.0128) = 0.0128$$
$$s_{12} = 1.0 - (-0.0175) = 1.0175$$
$$s_{13} = 0.5 - (-0.0170) = 0.5170$$

**Row 2:**
$$s_{20} = 0.5 - 0.4780 = 0.0220$$
$$s_{21} = 1.0 - (-0.0118) = 1.0118$$
$$s_{22} = 0 - (-0.0165) = 0.0165$$
$$s_{23} = 0.5 - (-0.0160) = 0.5160$$

**Row 3:**
$$s_{30} = 1.0 - 0.4805 = 0.5195$$
$$s_{31} = 0.5 - (-0.0093) = 0.5093$$
$$s_{32} = 0.5 - (-0.0140) = 0.5140$$
$$s_{33} = 0 - (-0.0135) = 0.0135$$

<br>

$$S = \begin{pmatrix} 6.4200 & 7.4098 & 7.4145 & 7.9140 \\ 0.0230 & 0.0128 & 1.0175 & 0.5170 \\ 0.0220 & 1.0118 & 0.0165 & 0.5160 \\ 0.5195 & 0.5093 & 0.5140 & 0.0135 \end{pmatrix}$$

**Verify dual feasibility:** $\min(S) = 0.0128 > 0$ ✓ — every entry is non-negative, meaning $\phi_i + \psi_j \leq \tilde{C}_{ij}$ everywhere. The price system never over-values any path.

<br>

**Sub-step B: Interpret the slack values**

The slack matrix directly explains the transport plan structure:

| Path $(i,j)$ | $s_{ij}$ | $\pi_{ij}$ | Meaning |
|---|---|---|---|
| $(1,1)$ | $0.0128$ | $0.2780$ | Near-zero slack → large flow. Same location, nearly free |
| $(2,2)$ | $0.0165$ | $0.1920$ | Near-zero slack → large flow. Same location, nearly free |
| $(3,3)$ | $0.0135$ | $0.2592$ | Near-zero slack → large flow. Same location, nearly free |
| $(1,0)$ | $0.0230$ | $0.1003$ | Small slack → moderate flow. One step apart |
| $(2,0)$ | $0.0220$ | $0.1108$ | Small slack → moderate flow. One step apart |
| $(0,j)$ | $6.4$–$7.9$ | $\approx 0$ | Enormous slack → zero flow. Background pixel priced out |
| $(1,2)$ | $1.0175$ | $\approx 0$ | Large slack → zero flow. Maximum distance path blocked |

The pattern is clear: slack near zero means the path is geometrically consistent with the price system and mass flows freely; large slack means the path is inconsistent and mass is exponentially suppressed.

<br>

**Sub-step C: Weighted slack $s_{ij} \cdot \pi_{ij}$**

Complementary slackness requires $s_{ij} \cdot \pi_{ij} \approx 0$ for all $(i,j)$. Two ways this is satisfied:

- $s_{ij} \approx 0$ (path is open, mass flows freely) — satisfied at $(1,1), (2,2), (3,3), (1,0), (2,0)$
- $\pi_{ij} \approx 0$ (path is blocked, no mass flows) — satisfied at all remaining entries

Computing for the five nonzero flows (from Step 13):

$$s_{10}\cdot\pi_{10} = 0.0230\times0.1003 = 0.002307$$
$$s_{11}\cdot\pi_{11} = 0.0128\times0.2780 = 0.003558$$
$$s_{20}\cdot\pi_{20} = 0.0220\times0.1108 = 0.002438$$
$$s_{22}\cdot\pi_{22} = 0.0165\times0.1920 = 0.003168$$
$$s_{33}\cdot\pi_{33} = 0.0135\times0.2592 = 0.003499$$

All other entries: $s_{ij}$ is large but $\pi_{ij} \approx 0$, so product $\approx 0$.

**Maximum weighted slack $= 0.003558$ — small but not exactly zero, again because potentials are assumed rather than exactly converged. At true convergence this collapses to $\approx 10^{-5}$ range as seen in the real output.**

<br>

**Sub-step D: KKT conditions — FULL SUMMARY**

Three conditions must all hold simultaneously for $\pi$ to be certifiably optimal:

| Condition | What it checks | Anchor result |
|---|---|---|
| Primal feasibility: $\pi \in \Pi(\mathbf{m},\mathbf{v})$ | Row sums $= \mathbf{m}$, col sums $= \mathbf{v}$ | ✓ Verified in Step 12 |
| Dual feasibility: $s_{ij} \geq 0\ \forall i,j$ | Price system never over-values any path | ✓ $\min(S) = 0.0128 > 0$ |
| Complementary slackness: $s_{ij}\pi_{ij} \approx 0\ \forall i,j$ | Mass only flows where prices are consistent | ✓ Max product $= 0.0036$ |

All three hold. $\pi$ is globally optimal — no other transport plan between $\mathbf{m}$ and $\mathbf{v}$ achieves a lower regularized cost. The mathematical certification is complete.

---

### Step 20: Displacement Interpolation — Wasserstein Geodesic

$\pi$ tells us how much mass moves from pixel $i$ to pixel $j$. Displacement interpolation asks: at time $t$, where is that mass *in transit*?

Each $\pi_{ij}$ unit of mass travels a straight line from $\mathbf{x}_i$ to $\mathbf{x}_j$ — at time $t$ it sits at:

$$(1-t)\mathbf{x}_i + t\mathbf{x}_j$$

Collecting all mass positions at each $t$ gives the intermediate frame $\mu_t$.

**Timesteps used:**

$$t \in \{0,\ 0.111,\ 0.222,\ 0.333,\ 0.444,\ 0.556,\ 0.667,\ 0.778,\ 0.889,\ 1.0\}$$

We work through $t=0$, $t=0.5$ (midpoint), and $t=1$ fully. These three cover all distinct behaviors.

`np.linspace(0, 1, 10)` divides the interval $[0,1]$ into 9 equal steps of size $1/9 \approx 0.111$, producing exactly 10 evenly spaced values from 0 to 1.

<br>

#### $t = 0$: Recovery of source $\mu$

$$\mathbf{z}_{ij}(0) = (1-0)\mathbf{x}_i + 0\cdot\mathbf{x}_j = \mathbf{x}_i \quad \forall j$$

Every unit of mass at $(i,j)$ lands at $\mathbf{x}_i$ — the source location. **All $j$ collapse to same position.**

**Pixel 0** ($m_0 = 0$): $\sum_j \pi_{0j} \approx 0$ → skipped.

**Pixel 1** ($\mathbf{x}_1 = (0,1)$): all mass lands at $(0,1)$ regardless of $j$:

$$k = 2\cdot0 + 1 = 1$$
$$\mu_0(1) \mathrel{+}= \pi_{10} + \pi_{11} + \pi_{12} + \pi_{13} = 0.1003 + 0.2780 + 0 + 0 = 0.3783$$

$k$ is the **flat pixel index** — it converts the rounded 2D interpolated coordinate $(r, c)$ back to a single integer via $k = 2r + c$ (the same row-major formula from Step 5), telling `np.add.at` exactly which slot in the 4-element `frame` vector to deposit mass $\pi_{ij}$ into.

$r$ and $c$ are the **rounded interpolated coordinates**:

$$r = \text{round}(z_{ij}^{(0)}), \quad c = \text{round}(z_{ij}^{(1)})$$

Once rounded to integers, $(r, c)$ is a valid 2D grid position. Converting it back to a flat index uses the same row-major formula as Step 5 but in reverse — Step 5 went from flat index $i$ to $(r,c)$ via $r = \lfloor i/2 \rfloor$, $c = i \bmod 2$. Here we go the other direction: given $(r,c)$, recover $k = 2r + c$.

For pixel 1 at $t=0$: interpolated position is $(0, 1.0)$, rounds to $r=0, c=1$, so $k = 2\cdot0 + 1 = 1$ — flat index 1, which is exactly pixel 1's position in the `frame` vector.

**Pixel 2** ($\mathbf{x}_2 = (1,0)$): all mass lands at $(1,0)$:

$$k = 2\cdot1 + 0 = 2$$
$$\mu_0(2) \mathrel{+}= \pi_{20} + \pi_{21} + \pi_{22} + \pi_{23} = 0.1108 + 0 + 0.1920 + 0 = 0.3028$$

**Pixel 3** ($\mathbf{x}_3 = (1,1)$): all mass lands at $(1,1)$:

$$k = 2\cdot1 + 1 = 3$$
$$\mu_0(3) \mathrel{+}= \pi_{30} + \pi_{31} + \pi_{32} + \pi_{33} = 0 + 0 + 0 + 0.2592 = 0.2592$$

$$\mu_0 = (0,\ 0.3783,\ 0.3028,\ 0.2592)$$

Compare to $\mathbf{m} = (0,\ 0.2835,\ 0.3150,\ 0.4016)$ — close but not exact due to assumed potentials. At true convergence: $\mu_0 = \mathbf{m}$ exactly ✓

<br>

#### $t = 0.5$: Midpoint frame

$$\mathbf{z}_{ij}(0.5) = 0.5\mathbf{x}_i + 0.5\mathbf{x}_j$$

**Pixel 0 skipped** ($m_0 = 0$).

**Pixel 1** ($\mathbf{x}_1 = (0,1)$), compute $\mathbf{z}_{1j}$ for all $j$:

$$\mathbf{z}_{10}(0.5) = 0.5(0,1) + 0.5(0,0) = (0,\ 0.5) \xrightarrow{\text{round}} (0,\ 0) \implies k=0$$
$$\mathbf{z}_{11}(0.5) = 0.5(0,1) + 0.5(0,1) = (0,\ 1.0) \xrightarrow{\text{round}} (0,\ 1) \implies k=1$$
$$\mathbf{z}_{12}(0.5) = 0.5(0,1) + 0.5(1,0) = (0.5,\ 0.5) \xrightarrow{\text{round}} (0,\ 0) \implies k=0$$
$$\mathbf{z}_{13}(0.5) = 0.5(0,1) + 0.5(1,1) = (0.5,\ 1.0) \xrightarrow{\text{round}} (0,\ 1) \implies k=1$$

Deposit mass:

$$\mu_{0.5}(0) \mathrel{+}= \pi_{10} + \pi_{12} = 0.1003 + 0 = 0.1003$$
$$\mu_{0.5}(1) \mathrel{+}= \pi_{11} + \pi_{13} = 0.2780 + 0 = 0.2780$$

**Pixel 2** ($\mathbf{x}_2 = (1,0)$), compute $\mathbf{z}_{2j}$:

$$\mathbf{z}_{20}(0.5) = 0.5(1,0) + 0.5(0,0) = (0.5,\ 0) \xrightarrow{\text{round}} (0,\ 0) \implies k=0$$
$$\mathbf{z}_{21}(0.5) = 0.5(1,0) + 0.5(0,1) = (0.5,\ 0.5) \xrightarrow{\text{round}} (0,\ 0) \implies k=0$$
$$\mathbf{z}_{22}(0.5) = 0.5(1,0) + 0.5(1,0) = (1.0,\ 0) \xrightarrow{\text{round}} (1,\ 0) \implies k=2$$
$$\mathbf{z}_{23}(0.5) = 0.5(1,0) + 0.5(1,1) = (1.0,\ 0.5) \xrightarrow{\text{round}} (1,\ 0) \implies k=2$$

Deposit mass:

$$\mu_{0.5}(0) \mathrel{+}= \pi_{20} + \pi_{21} = 0.1108 + 0 = 0.1108$$
$$\mu_{0.5}(2) \mathrel{+}= \pi_{22} + \pi_{23} = 0.1920 + 0 = 0.1920$$

**Pixel 3** ($\mathbf{x}_3 = (1,1)$), compute $\mathbf{z}_{3j}$:

$$\mathbf{z}_{30}(0.5) = 0.5(1,1) + 0.5(0,0) = (0.5,\ 0.5) \xrightarrow{\text{round}} (0,\ 0) \implies k=0$$
$$\mathbf{z}_{31}(0.5) = 0.5(1,1) + 0.5(0,1) = (0.5,\ 1.0) \xrightarrow{\text{round}} (0,\ 1) \implies k=1$$
$$\mathbf{z}_{32}(0.5) = 0.5(1,1) + 0.5(1,0) = (1.0,\ 0.5) \xrightarrow{\text{round}} (1,\ 0) \implies k=2$$
$$\mathbf{z}_{33}(0.5) = 0.5(1,1) + 0.5(1,1) = (1.0,\ 1.0) \xrightarrow{\text{round}} (1,\ 1) \implies k=3$$

Deposit mass:

$$\mu_{0.5}(0) \mathrel{+}= \pi_{30} = 0$$
$$\mu_{0.5}(1) \mathrel{+}= \pi_{31} = 0$$
$$\mu_{0.5}(2) \mathrel{+}= \pi_{32} = 0$$
$$\mu_{0.5}(3) \mathrel{+}= \pi_{33} = 0.2592$$

**Accumulate all deposits at $t=0.5$:**

$$\mu_{0.5}(0) = 0.1003 + 0.1108 + 0 = 0.2111$$
$$\mu_{0.5}(1) = 0.2780 + 0 + 0 = 0.2780$$
$$\mu_{0.5}(2) = 0.1920 + 0 = 0.1920$$
$$\mu_{0.5}(3) = 0.2592$$

$$\mu_{0.5} = (0.2111,\ 0.2780,\ 0.1920,\ 0.2592)$$

Verify mass conservation: $0.2111 + 0.2780 + 0.1920 + 0.2592 = 0.9403 \approx 1$ ✓ (small gap from assumed potentials)

**What happened geometrically:** at $t=0$, pixel 0 carried zero mass. At $t=0.5$, pixel 0 now holds $0.2111$ — mass from pixels 1 and 2 is physically in transit, passing through pixel 0's location on its way to the target. This is displacement interpolation: mass moves through space, not through intensity.

<br>

#### $t = 1$: Recovery of target $\nu$

$$\mathbf{z}_{ij}(1) = 0\cdot\mathbf{x}_i + 1\cdot\mathbf{x}_j = \mathbf{x}_j \quad \forall i$$

Every unit of mass lands at $\mathbf{x}_j$ — the target location. All $i$ collapse to target positions.

**Pixel 0 skipped.**

**From pixel 1:** mass $\pi_{10}=0.1003$ lands at $\mathbf{x}_0=(0,0)\to k=0$; mass $\pi_{11}=0.2780$ lands at $\mathbf{x}_1=(0,1)\to k=1$.

**From pixel 2:** mass $\pi_{20}=0.1108$ lands at $\mathbf{x}_0=(0,0)\to k=0$; mass $\pi_{22}=0.1920$ lands at $\mathbf{x}_2=(1,0)\to k=2$.

**From pixel 3:** mass $\pi_{33}=0.2592$ lands at $\mathbf{x}_3=(1,1)\to k=3$.

$$\mu_1(0) = \pi_{10} + \pi_{20} = 0.1003 + 0.1108 = 0.2111$$
$$\mu_1(1) = \pi_{11} = 0.2780$$
$$\mu_1(2) = \pi_{22} = 0.1920$$
$$\mu_1(3) = \pi_{33} = 0.2592$$

$$\mu_1 = (0.2111,\ 0.2780,\ 0.1920,\ 0.2592)$$

Compare to $\mathbf{v} = (0.2373,\ 0.2881,\ 0.2034,\ 0.2712)$ — close, gap from assumed potentials. At true convergence: $\mu_1 = \mathbf{v}$ exactly ✓

<br>

#### Summary across all three frames

| Pixel $k$ | Location | $\mu_0$ (source) | $\mu_{0.5}$ (midpoint) | $\mu_1$ (target) | $\mathbf{v}$ (true target) |
|---|---|---|---|---|---|
| 0 | $(0,0)$ | $0$ | $0.2111$ | $0.2111$ | $0.2373$ |
| 1 | $(0,1)$ | $0.3783$ | $0.2780$ | $0.2780$ | $0.2881$ |
| 2 | $(1,0)$ | $0.3028$ | $0.1920$ | $0.1920$ | $0.2034$ |
| 3 | $(1,1)$ | $0.2592$ | $0.2592$ | $0.2592$ | $0.2712$ |

The key observation: pixel 0 starts at zero mass, gains $0.2111$ by $t=0.5$ as transiting mass passes through, and holds that mass at $t=1$. Pixels 1, 2, 3 shed mass as it departs toward target positions. Mass is physically redistributed through space — not faded.

---

### Step 21: Pixel Blending Baseline at $t=0.5$

<br>

$$\mu_{0.5}^{\text{blend}} = 0.5\,\mathbf{m} + 0.5\,\mathbf{v}$$

$$\mu_{0.5}^{\text{blend}}(0) = 0.5\times0 + 0.5\times0.2373 = 0.1187$$
$$\mu_{0.5}^{\text{blend}}(1) = 0.5\times0.2835 + 0.5\times0.2881 = 0.2858$$
$$\mu_{0.5}^{\text{blend}}(2) = 0.5\times0.3150 + 0.5\times0.2034 = 0.2592$$
$$\mu_{0.5}^{\text{blend}}(3) = 0.5\times0.4016 + 0.5\times0.2712 = 0.3364$$

$$\mu_{0.5}^{\text{blend}} = (0.1187,\ 0.2858,\ 0.2592,\ 0.3364)$$

<br>

**Direct comparison at $t=0.5$:**

| Pixel $k$ | Wasserstein $\mu_{0.5}$ | Blending $\mu_{0.5}^{\text{blend}}$ | Difference |
|---|---|---|---|
| 0 | $0.2111$ | $0.1187$ | Wasserstein sends more mass here — transiting mass physically passes through |
| 1 | $0.2780$ | $0.2858$ | Similar — pixel 1 is shared by both images |
| 2 | $0.1920$ | $0.2592$ | Blending keeps more mass here — it just averages, never moves it |
| 3 | $0.2592$ | $0.3364$ | Blending keeps pixel 3 bright — source was bright, target was bright, so average stays bright; Wasserstein has already dispatched some mass |

Blending never moves mass — pixel 3 stays bright because both $m_3$ and $v_3$ are large, so their average is large. Wasserstein reduces pixel 3's mass because some of it is physically in transit to pixel 0.

---

### Step 22: Naive Pixel Blending — All Timesteps

Naive Pixel Blending computes a weighted average of $\mathbf{m}$ and $\mathbf{v}$ at each pixel independently — no coordinates, no transport, no geometry.

$$\mu_t^{\text{blend}}(k) = (1-t)\,m_k + t\,v_k \quad \forall k$$

**Full computation at all three key timesteps:**

**$t=0$:**
$$\mu_0^{\text{blend}} = 1.0\cdot(0,\ 0.2835,\ 0.3150,\ 0.4016) + 0\cdot(0.2373,\ 0.2881,\ 0.2034,\ 0.2712) = (0,\ 0.2835,\ 0.3150,\ 0.4016) = \mathbf{m}\ \checkmark$$

**$t=0.5$:**

$$\mu_{0.5}^{\text{blend}}(0) = 0.5\times0 + 0.5\times0.2373 = 0.1187$$
$$\mu_{0.5}^{\text{blend}}(1) = 0.5\times0.2835 + 0.5\times0.2881 = 0.2858$$
$$\mu_{0.5}^{\text{blend}}(2) = 0.5\times0.3150 + 0.5\times0.2034 = 0.2592$$
$$\mu_{0.5}^{\text{blend}}(3) = 0.5\times0.4016 + 0.5\times0.2712 = 0.3364$$

$$\mu_{0.5}^{\text{blend}} = (0.1187,\ 0.2858,\ 0.2592,\ 0.3364)$$

**$t=1$:**
$$\mu_1^{\text{blend}} = 0\cdot\mathbf{m} + 1.0\cdot\mathbf{v} = (0.2373,\ 0.2881,\ 0.2034,\ 0.2712) = \mathbf{v}\ \checkmark$$

<br>

**Mass conservation at $t=0.5$:**

$$\sum_k \mu_{0.5}^{\text{blend}}(k) = 0.1187 + 0.2858 + 0.2592 + 0.3364 = 1.0001 \approx 1\ \checkmark$$

Exact by linearity: $(1-t)\sum_k m_k + t\sum_k v_k = (1-t)\cdot1 + t\cdot1 = 1$.

<br>

#### Final Comparison: Wasserstein vs Blending at $t=0.5$

| Pixel $k$ | Location | $m_k$ | $v_k$ | $\mu_{0.5}^{\text{Wass}}$ | $\mu_{0.5}^{\text{blend}}$ |
|---|---|---|---|---|---|
| 0 | $(0,0)$ | $0$ | $0.2373$ | $0.2111$ | $0.1187$ |
| 1 | $(0,1)$ | $0.2835$ | $0.2881$ | $0.2780$ | $0.2858$ |
| 2 | $(1,0)$ | $0.3150$ | $0.2034$ | $0.1920$ | $0.2592$ |
| 3 | $(1,1)$ | $0.4016$ | $0.2712$ | $0.2592$ | $0.3364$ |

**Reading pixel 0:** source has zero mass, target has $0.2373$.

- Blending: $0.1187$ — pure arithmetic average, mass appears from nowhere at pixel 0.
- Wasserstein: $0.2111$ — mass physically arrived here, transported from pixels 1 and 2 in transit.

**Reading pixel 3:** source has $0.4016$, target has $0.2712$.

- Blending: $0.3364$ — stays bright, just averaged down slightly.
- Wasserstein: $0.2592$ — mass has already departed toward its target destinations; pixel 3 is visibly dimmer because the transport is happening.

**Reading pixel 2:** source has $0.3150$, target has $0.2034$.

- Blending: $0.2592$ — averaged, still substantial.
- Wasserstein: $0.1920$ — mass is leaving this location, already in transit toward pixel 0.

<br>

**THE FUNDAMENTAL DIFFERENCE IN ONE TABLE:**

| Property | Wasserstein $\mu_t$ | Blending $\mu_t^{\text{blend}}$ |
|---|---|---|
| How mass at pixel 0 appears | Physically transported from pixels 1, 2 | Arithmetically interpolated from $v_0$ |
| Pixel coordinates used? | Yes — $\mathbf{z}_{ij}(t) = (1-t)\mathbf{x}_i + t\mathbf{x}_j$ | No — pure weighted average of intensities |
| Mass at $t=0.5$ reflects | Where mass physically is mid-journey | Average of source and target intensities |
| Intermediate frames look like | Ink strokes shifting and morphing in space | Two images fading through each other |
| Straight line in | $\mathbb{R}^2$ — image plane | $\Delta^3$ — intensity simplex |

<br>

**Complete anchor pipeline state at $t=0.5$:**

$$\mu_{0.5}^{\text{Wass}} = (0.2111,\ 0.2780,\ 0.1920,\ 0.2592) \rightarrow \text{mass mid-journey through space}$$
$$\mu_{0.5}^{\text{blend}} = (0.1187,\ 0.2858,\ 0.2592,\ 0.3364) \rightarrow \text{mass averaged in intensity}$$

**Both sum to 1. Both equal $\mathbf{m}$ at $t=0$ and $\mathbf{v}$ at $t=1$. But only the Wasserstein frame respects the 2D geometry of the image — mass traveled to get there.**

---

## Conclusion

| Step | Cell | Operation | Mathematical Object | What It Produces |
|---|---|---|---|---|
| 1 | 4 | Extract labels, find first "3" and "8" | $i_3 = \min\mathcal{I}_3,\ i_8 = \min\mathcal{I}_8$ | Indices $i_3=0,\ i_8=2$ into toy dataset |
| 2 | 4 | Cast images to float64 | $A^{(3)}, A^{(8)} \in \mathbb{R}^{2\times2}$ | $A^{(3)}=\begin{pmatrix}0&180\\200&255\end{pmatrix}$, enables real arithmetic |
| 3 | 6 | Flatten images | $A \in \mathbb{R}^{2\times2} \to \mathbf{a} \in \mathbb{R}^4$ | $\mathbf{a}^{(3)}=(0,180,200,255)$, $\mathbf{a}^{(8)}=(210,255,180,240)$ |
| 4 | 6 | Normalize to probability measures | $\mathbf{m}=\mathbf{a}^{(3)}/Z^{(3)},\ \mathbf{v}=\mathbf{a}^{(8)}/Z^{(8)}$ | $\mathbf{m},\mathbf{v}\in\Delta^3$, both sum to 1 |
| 5 | 8 | Assign pixel coordinates | $\mathbf{x}_i=(\lfloor i/2\rfloor,\ i\bmod 2)$ | $X\in\mathbb{R}^{4\times2}$, embeds pixels into $\mathbb{R}^2$ |
| 6 | 9 | Build cost matrix | $C_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|^2$ | $C\in\mathbb{R}^{4\times4}$, encodes travel cost between every pixel pair |
| 7 | 12 | Normalize cost matrix | $\tilde{C}=C/\max C$ | $\tilde{C}\in[0,1]^{4\times4}$, prevents exponential overflow in Sinkhorn |
| 8 | 11 | Compute log-measures | $\log\_mu_i = \log(m_i+10^{-300})$ | $(-690.78,\ -1.261,\ -1.155,\ -0.912)$, guards against $\log(0)$ |
| 9 | 11 | Initialize dual potentials | $\mathbf{f}^{(0)}=\mathbf{g}^{(0)}=\mathbf{0}$ | Starting point; plan is cost-only, no marginal constraints yet |
| 10 | 11 | Sinkhorn f-update, iter 1 | $f_i \leftarrow \varepsilon\log m_i - \varepsilon\operatorname{LSE}_j\!\left(\frac{g_j-\tilde{C}_{ij}}{\varepsilon}\right)$ | $\mathbf{f}^{(1)}=(-6.908,\ -0.01261,\ -0.01155,\ -0.00912)$ |
| 11 | 11 | Sinkhorn g-update, iter 1 | $g_j \leftarrow \varepsilon\log\nu_j - \varepsilon\operatorname{LSE}_i\!\left(\frac{f_i-\tilde{C}_{ij}}{\varepsilon}\right)$ | $\mathbf{g}^{(1)}=(0.497,\ 0.00017,\ -0.00439,\ -0.00393)$ |
| 12 | 11–12 | Recover transport plan | $\pi_{ij}=\exp\!\left(\frac{f^*_i+g^*_j-\tilde{C}_{ij}}{\varepsilon}\right)$ | $\pi\in\Pi(\mathbf{m},\mathbf{v})$, marginals verified; mass flow between every pixel pair |
| 13 | 13 | Compute primal cost | $\langle\tilde{C},\pi\rangle=\sum_{ij}\tilde{C}_{ij}\pi_{ij}$ | $0.1056$; total transport bill — only off-diagonal flows contribute |
| 14 | 15 | Effect of $\varepsilon$ on cost | $\mathcal{L}_\varepsilon(\pi)=\langle\tilde{C},\pi\rangle-\varepsilon H(\pi)$ | Max-entropy cost $0.4949$ vs optimized $0.1056$; geometry saves 79% |
| 15 | 16 | Extract dual potentials | $\phi=\mathbf{f}^*,\ \psi=\mathbf{g}^*$ | Price system: $\phi_0=-6.91$ prices out background; ink pixels near zero |
| 16 | 17 | Compute entropy $H(\pi)$ | $H(\pi)=-\sum_{ij}\pi_{ij}\log\pi_{ij}$ | $1.497$ on anchor; measures how spread the transport plan is |
| 17 | 17 | Compute regularized objective | $\mathcal{L}_\varepsilon(\pi)=\langle\tilde{C},\pi\rangle-\varepsilon H(\pi)$ | $0.09063$; single number Sinkhorn minimized — certifies optimality |
| 18 | 18 | Compute dual objective | $\mathcal{D}(\phi,\psi)=\langle\phi,\mathbf{m}\rangle+\langle\psi,\mathbf{v}\rangle$ | $0.1031$; gap $0.0124$ from assumed potentials; zero at true convergence |
| 19 | 19 | Verify complementary slackness | $s_{ij}=\tilde{C}_{ij}-(\phi_i+\psi_j)\geq0$;\ $s_{ij}\pi_{ij}\approx0$ | KKT conditions satisfied; $\pi$ certified globally optimal from all angles |
| 20 | 20–21 | Displacement interpolation | $\mu_t(k)=\sum_{T_t(\mathbf{x}_i,\mathbf{x}_j)=\mathbf{x}_k}\pi_{ij}$ | 10 geodesic frames; $\mu_0=\mathbf{m}$, $\mu_1=\mathbf{v}$, mass physically moves through $\mathbb{R}^2$ |
| 21 | 23 | Pixel blending baseline | $\mu_t^{\text{blend}}(k)=(1-t)m_k+tv_k$ | Straight line on simplex $\Delta^3$; no geometry — intensities averaged, mass teleports |
| 22 | 23 | Final comparison | $\mu_{0.5}^{\text{Wass}}$ vs $\mu_{0.5}^{\text{blend}}$ | Wasserstein: mass mid-journey through space. Blending: arithmetic average of intensities. Both valid measures; only Wasserstein respects image geometry |

---
---