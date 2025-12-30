# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm

## Authors
Noah Amsel, David Persson, Christopher Musco, Robert M. Gower

## Abstract
Computing the polar decomposition and the related matrix sign function has been a well-studied problem in numerical analysis for decades. Recently, it has emerged as an important subroutine within the Muon algorithm for training deep neural networks. However, the requirements of this application differ sharply from classical settings: deep learning demands GPU-friendly algorithms that prioritize high throughput over high precision. We introduce Polar Express, a new method for computing the polar decomposition. Like Newton-Schulz and other classical polynomial methods, our approach uses only matrix-matrix multiplications, making it very efficient on GPUs. Inspired by earlier work of Chen &amp; Chow and Nakatsukasa &amp; Freund, Polar Express adapts the update rule at each iteration by solving a minimax optimization problem. We prove that this strategy minimizes error in a worst-case sense, allowing Polar Express to converge as rapidly as possible both in the early iterations and asymptotically. We also address finite-precision issues, making it practical to use in bfloat16. When integrated into the Muon training framework, our method leads to consistent improvements in validation loss when training a GPT-2 model on one billion tokens from the FineWeb dataset, outperforming recent alternatives across a range of learning rates.

# Introduction

Advanced linear algebra is making its way into deep learning. Efficient
algorithms for computing *matrix functions* have found exciting new
applications in training neural networks. In particular, approximations
to the matrix-inverse are used in the full Adagrad method , the matrix
square-root and quarter-root appear as subroutines in the Shampoo and
Soap optimizers , and most recently, the matrix sign function has become
a key ingredient of the `Muon` optimizer .

While the problem of computing these matrix functions has been studied
by numerical analysts for decades, applications in deep learning come
with different requirements than those in computational science. For
deep learning, it is critical to take maximum advantage of GPU-friendly
operations like matrix-matrix products and to avoid less parallel
operations. Moreover, memory overhead must be small to handle large
models. On the other hand, high accuracy is typically less important;
the gold standard of sixteen digits of accuracy is overkill in deep
learning.

Given these considerations, there is a need to develop new matrix
function methods that are tailor-made for deep learning applications. We
take on this challenge by designing a state-of-the-art, GPU-friendly
algorithm for computing the matrix sign function, or more generally, for
computing the *polar decomposition* of a rectangular matrix. We apply
our new `Polar Express` method () to compute the descent direction in
the increasingly popular `Muon` optimizer. In , we show that using
`Polar Express` within `Muon` consistently results in lower validation
loss across all learning rates when training a GPT-2 model, as compared
to other matrix sign methods .



``` python
from itertools import repeat
import torch

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5)
                for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 +1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X  G.size(-1): X = X.mT
    return X
```





## The Muon Method

The `Muon` optimizer has recently gained popularity for training large
language models, often outperforming state-of-the-art adaptive gradient
methods like Adam and AdamW . `Muon` has been used to set records for
the NanoGPT speedrun , to expand the Pareto frontier of performance
versus training FLOPs for large language models , and even to train a
32-billion parameter frontier LLM .

The `Muon` update rule  is defined as follows. Let $\lambda, \beta > 0$
be the learning rate and momentum coefficient hyperparameters. (By
default, $\beta = 0.9$.) Let $\bm{W}_t\in\mathbb{R}^{m\times n}$ be the
weight matrix of a given neural network layer at iteration $t$, and let
$\bm{G}_t\in\mathbb{R}^{m\times n}$ be its (stochastic) gradient. Let
$\bm{M}_t \in \mathbb{R}^{m\times n}$ be the running momentum estimate
of the gradient, where $\bm{M}_0 = \bm{0}$. The `Muon` update is given
by $$\begin{aligned}
\bm{M}_{t} &= \beta \bm{M}_{t-1}+(1-\beta) \bm{G}_{t}   \\
    \bm{W}_{t+1} & = \bm{W}_{t}- \lambda \mathop{\mathrm{polar}}(\bm{M}_t). \label{eq:muon}
\end{aligned}$$ Whereas standard stochastic gradient descent (SGD) with
momentum updates the weight matrix by taking a step in the direction
$-\bm{M}_t$, the `Muon` method steps in the direction
$-\mathop{\mathrm{polar}}(\bm{M}_t)$, where
$\mathop{\mathrm{polar}}(\bm{M})$ denotes the closest semi-orthogonal
matrix to $\bm{M}$ . Concretely, if
$\bm{M}= \bm{U}\bm{\Sigma}\bm{V}^\mathsf{T}$ is the singular value
decomposition (SVD) of $\bm{M}$, then $$\label{eq:matrixsign}
    \mathop{\mathrm{polar}}(\bm{M}) := \bm{U}\bm{V}^\mathsf{T}.$$ The
matrix $\mathop{\mathrm{polar}}(\bm{M})$ can be seen as a generalization
of the matrix sign function to rectangular matrices . Indeed, when
$\bm{M}$ is square symmetric with eigendecomposition
$\bm{M}= \bm{V}\boldsymbol{\Lambda} \bm{V}^\mathsf{T}$,
$\mathop{\mathrm{polar}}(\bm{M})$ exactly coincides with the matrix sign
function
$\mathop{\mathrm{sign}}(\bm{M}) = \bm{V}\mathop{\mathrm{sign}}(\boldsymbol{\Lambda}) \bm{V}^\mathsf{T}$
. Equivalently, $\mathop{\mathrm{polar}}(\bm{M})$ is the left orthogonal
factor of the polar decomposition of $\bm{M}$ . The motivation for
`Muon` is that $-\mathop{\mathrm{polar}}(\bm{M})$ gives the
steepest-descent direction with respect to the *spectral norm* (instead
of the Frobenius norm, as in standard SGD).

Recent work  shows that `Muon` can be viewed as a conditional gradient
(Frank-Wolfe) method with a trust region defined by the spectral norm.
In the same work, the authors also provide a convergence theory for the
smooth and non-convex setting, as well as for the stochastic non-convex
case. The analysis of `Muon` was further refined in , which proves
convergence under a layerwise $(L_0, L_1)$-smoothness assumption, in
both the stochastic non-convex and stochastic Polyak–Łojasiewicz
settings. We also note earlier work that anticipated `Muon`’s use of the
polar factor and its motivation as steepest descent under the spectral
norm . We refer the reader to  and  for further background. In this
paper, we take the `Muon` update rule as given and focus on the problem
of efficiently computing the polar decomposition
$\mathop{\mathrm{polar}}(\bm{M})$.

## Computing the Polar Factor

Although $\mathop{\mathrm{polar}}(\bm{M})$ can be computed directly via
an SVD in $O(\min(mn^2, nm^2))$ time, doing so is prohibitively
expensive in deep learning applications, especially as standard SVD
algorithms fail to take full advantage of the parallelism available on
GPUs. There has been significant work on highly-parallel methods for the
SVD, but the most common approaches actually require computing the
matrix-sign function as a subroutine . Numerical analysts have spent
decades developing iterative methods for computing
$\mathop{\mathrm{polar}}(\bm{M})$. This rich line of work includes
Newton–Schulz , Padé iteration , the Newton and scaled Newton iterations
, the QWHD iteration , and *Zolo-pd* (Zolotarev polar decomposition) .
Unfortunately, as discussed in , most of these methods are based on
rational approximations to the function $\mathop{\mathrm{sign}}(x)$ and
require computing matrix inverses or QR decompositions. Such methods are
ill-suited to GPU acceleration and deep learning applications. In
contrast, the older Newton-Schulz method is based on *polynomial*
approximation of $\mathop{\mathrm{sign}}(x)$ and uses only matrix-matrix
products. Thus, `Muon` initially used Newton-Schulz . Indeed, `Muon`
stands for “MomentUm Orthogonalized by Newton-Schulz” .

#### The Newton-Schulz methods.

Newton-Schulz constructs a sequence of approximations
$\bm{X}_t \approx \mathop{\mathrm{polar}}(\bm{M})$ as follows:
$$\begin{aligned}
\label{eq:NS3}
&\bm{X}_0 = \bm{M}/\|\bm{M}\|_\text{F}&\bm{X}_{t+1} = \frac32 \bm{X}_t - \frac12 \bm{X}_t \bm{X}_t^\top \bm{X}_t
\end{aligned}$$ At each iteration, this rule effectively applies the
cubic polynomial $p(x) = \frac32 x - \frac12 x^3$ to each singular value
of $\bm{X}_t$. The scalar fixed-point iteration $x_{t+1} = p(x_t)$
converges to $\mathop{\mathrm{sign}}(x_0)$ as $t \to \infty$, provided
$|x_0| \leq 1$. As a result, the matrix iteration satisfies
$\lim\limits_{t\to\infty} \bm{X}_t = \bm{U}\bm{V}^\top = \mathop{\mathrm{polar}}(\bm{X}_0)$.
Higher-degree versions of Newton-Schulz follow the same principle. For
example, the degree-5 polynomial $p(x) = (15x - 10 x^3 + 3 x^5)/8$
converges even faster. The Newton-Schulz iterations converge
super-exponentially when $\bm{X}_t$ is sufficiently close to
$\mathop{\mathrm{polar}}(\bm{M})$, but they suffer from slow initial
convergence; when $\bm{X}_0$ is far from
$\mathop{\mathrm{polar}}(\bm{M})$, the approximation improves slowly
over the first few iterations.

#### The Jordan and You methods.

In `Muon`, high accuracy approximations to
$\mathop{\mathrm{polar}}(\bm{M})$ are usually not necessary. The primary
goal is instead to compute a coarse approximation in as few iterations
as possible. To accelerate convergence in the low-accuracy regime,
Jordan recently proposed a fixed-point iteration based on the polynomial
$p(x)=3.4445x - 4.7750x^3 + 2.0315x^5$ , which was found using a
heuristic numerical search. Unlike Newton-Schulz, the scheme that Jordan
proposed does not converge to $\mathop{\mathrm{polar}}(\bm{M})$.
Instead, it plateaus at an error of $\approx 0.3$. However, it reaches
this level of accuracy rapidly. As a result, when the number of
iterations is smaller than about $10$, Jordan’s method outperforms the
Newton-Schulz iteration. Building on this idea, You  proposed a method
that applies six different polynomial updates in succession, which were
again found by heuristic search. This method achieves better accuracy
than Jordan’s but still fails to converge.

We introduce a new method. In particular, we derive polynomial update
rules that are *optimal* at every iteration, outperforming all previous
polynomial methods in our setting.

## Contributions

We present `Polar Express`[^6], an iterative method for approximating
$\mathop{\mathrm{polar}}(\bm{M})$. Our method dynamically adapts the
polynomial update rule at each iteration, prioritizing rapid progress in
the initial stage and high accuracy in the later stage. `Polar Express`
constructs polynomials $p_1, \ldots, p_T$ so that the resulting
composition is the optimal approximation to the sign function with
respect to the supremum ($L^{\infty}$) norm (). By iteratively applying
these polynomials to $\bm{M}$, `Polar Express` computes an approximation
to $\mathop{\mathrm{polar}}(\bm{M})$ that is optimal in the worst-case
at every iteration. Our method converges to
$\mathop{\mathrm{polar}}(\bm{M})$ super-exponentially (), and it quickly
reaches a good approximation within just five to ten iterations. This
early-stage acceleration is especially valuable in deep learning
applications, where runtime efficiency takes precedence over high
accuracy. In contrast, classical methods like Newton-Schulz suffer from
a slow initial convergence, while recent heuristic proposals  fail to
converge. Our method is efficient to run on GPUs, using only a few
matrix-matrix products per iteration.[^7]

We give an explicit instantiation of `Polar Express` in , which
incorporates minor modifications to make it compatible with
half-precision arithmetic (see ). can be used as a drop-in replacement
for previous methods. In numerical experiments, `Polar Express`
outperforms previous methods on synthetic matrices and gradient matrices
from a GPT-2 transformer (). We demonstrate the effectiveness of using
`Polar Express` within the `Muon` optimizer in , showing that it
consistently improves the training of GPT-2 language models on 1 billion
tokens of the FineWeb dataset .

#### Notation.

We let $\|\bm{M}\|_\text{F}$ and $\|\bm{M}\|_2$ denote the Frobenius
norm and spectral norm (largest singular value) of a matrix $\bm{M}$,
respectively. We denote the spectrum (set of singular values) by
$\sigma(\bm{M})$.

Let $\mathbb{P}_d$ be the set of polynomials of degree at most $d$. For
odd $d$, $\mathbb{P}_{d}^{\mathop{\mathrm{odd}}}$ denotes the set of
polynomials of degree at most $d$ containing only odd-degree monomials.
For a polynomial $p$, $\deg(p)$ is its degree. Let
$\mathop{\mathrm{sign}}(x)$ be the scalar sign function, which satisfies
$\mathop{\mathrm{sign}}(0) = 0$, $\mathop{\mathrm{sign}}(x) = 1$ if
$x > 0$ and $\mathop{\mathrm{sign}}(x) = -1$ if $x 0$,
we define $p(\bm{M}) := \bm{U} p(\bm{\Sigma})\bm{V}^\mathsf{T}$, where
$p(\bm{\Sigma})$ is the diagonal matrix with diagonal entries
$p(\sigma_i)$ for $i = 1,\ldots,\mathop{\mathrm{rank}}(\bm{M})$.

# Related Work

Computing $\mathop{\mathrm{polar}}(\bm{M})$ is an important and
longstanding problem in numerical linear algebra, with applications
spanning electronic structure calculations, lattice quantum
chromodynamics, orthogonal Procrustes analysis, parallel algorithms for
computing the SVD, and beyond; see e.g. .

#### Newton-Schulz and polynomial Padé methods.

The earliest methods in the literature are polynomial iterations like
[eq:NS3]. Several nearly simultaneous papers
introduced the family of polynomial Padé iterations, comprising
Newton-Schulz and its higher-degree analogues . These higher-degree
methods are also sometimes called “Newton-Schulz”; when doing so, we
will specify the degree for clarity. In these methods, each iteration
refines the current approximation $\bm{X}_t$ by applying a low-degree
odd matrix polynomial, where any odd monomial $x \mapsto x^{2q+1}$ is
defined for rectangular matrices by the formula
$\bm{X}_t \mapsto \bm{X}_t\left( \bm{X}_t^\top \bm{X}_t\right)^q$. Our
`Polar Express` method also takes this form, though unlike
Newton-Schulz, it changes the polynomial at each iteration.

The polynomials used in Padé methods are chosen to match the value and
first few derivatives of $\mathop{\mathrm{sign}}(x)$ at the points
$x = \pm 1$. For instance, the update rule of the third method in this
family is defined by
$p(x) = \frac1{16}\left(35x - 35x^3 + 21x^5 - 5x^7\right)$, which is the
unique degree-7 polynomial satisfying $p(\pm 1) = \pm 1$ and
$p'(\pm1) = p''(\pm 1) = p'''(\pm 1) = 0$. These methods converge so
long as all singular values of $\bm{X}_0$ lie in $(0, 1]$, a condition
guaranteed by the initialization of
[eq:NS3]. Furthermore, the order of
convergence of the degree $2q+1$ method is $q+1$ . In particular, the
Newton-Schulz method ($q=1$) converges quadratically.

#### Newton’s method and rational Padé.

In the numerical analysis literature, polynomial methods were succeeded
by rational iterations like Newton’s method , defined as follows[^8]:
$$\begin{aligned}
&\bm{X}_0 = \bm{M}& \bm{X}_{t+1} = \frac12\left(\bm{X}_t + \bm{X}_t^{-\top}\right)
\end{aligned}$$ Newton’s method also converges quadratically. Like
Newton-Schulz, it works because the rational function
$r(x) = \frac12(x + x^{-1})$ has a stable fixed point at $1$; unlike for
Newton-Schulz, this point is a global attractor for the whole positive
real line. At first glance, Newton’s method has nothing to do with the
Padé iterations discussed above. However, after a change of variables
$\bm{Y}_t = \bm{X}_t^{-1}$, it can be reinterpreted as
$\bm{Y}_{t+1} = 2\bm{Y}_t(\bm{I}+ \bm{Y}_t^\top \bm{Y}_t)^{-1}$, which
is sometimes called inverse Newton. Observing that
$r(x) = \frac{2x}{1+x^2}$ satisfies $r(\pm 1) = \pm 1$ and
$r'(\pm 1) = 0$, we see that (inverse) Newton is also a Padé method,
though a rational rather than polynomial one. In fact, given a odd
degree $2q_n+1$ for the numerator and an even degree $2q_d$ for the
denominator, there is a unique rational function that matches the value
and first $q_n + q_d$ derivatives of $\mathop{\mathrm{sign}}(x)$ at
$x = \pm 1$. This directly yields a Padé method for computing
$\mathop{\mathrm{polar}}(\bm{M})$ whose order of convergence is
$q_n + q_d + 1$. For instance, $r(x) = \frac{3x+x^3}{1+3x^2}$ is called
Halley’s method, which converges cubically. When $q_d = 0$, we recover
the polynomial Padé methods.

There are two main weakness of Newton’s method and the Padé iterations:
slow convergence in the initial phase and the need to compute explicit
inverses. To accelerate initial convergence, Higham popularized the
technique of rescaling the matrix after every Newton iteration .
Intuitively, rescaling $\bm{X}_t$ so that
$\sigma_{\max} = 1/\sigma_{\min}$ centers the spectrum around $1$, where
convergence is fastest. Several easily-computable choices of scaling
factor exist to accomplish this approximately. Note that this rescaling
scheme would fail for Newton-Schulz, which likewise suffers from slow
initial convergence but which would diverge if $\sigma_{\max} \gg 1$.

Computing matrix inverses is difficult to parallelize and to implement
stably in low precision arithmetic. However, a trick was developed for
stably computing many rational methods *without* explicit inverses; QR
decompositions can be used instead . Applying this trick to Halley’s
method and combining with a special rescaling scheme yields the QDWH
(QR-based dynamically weighted Halley) method, which converges in just
six iterations for any reasonably conditioned matrix .

#### Adaptive rational methods from optimal approximations.

A landmark 2016 paper introduced a new paradigm to design iterative
methods for computing $\mathop{\mathrm{polar}}(\bm{M})$ . We describe
this paradigm in more detail in , but the main insight is as follows.
Padé methods choose the update rule to be an approximation to
$\mathop{\mathrm{sign}}(x)$ of a given degree that is optimally accurate
in the neighborhood of $x=1$. Instead, we should choose the
approximation to $\mathop{\mathrm{sign}}(x)$ that is optimal over an
*interval* $[\ell, 1] \subset \mathbb{R}_{\geq 0}$ that contains the
singular values. Moreover, after each step of the algorithm, the range
of the singular values changes; therefore, we adapt the update rule at
each iteration to match the new interval. When the range of the singular
values is large, this approach ensures that the update rule shrinks it
as quickly as possible. As the algorithm proceeds and the interval
shrinks to a small neighborhood of $1$, the update rule approaches that
of a Padé method, maintaining the same high order of convergence as it
has.

Within the class of odd rational functions whose numerators and
denominators have degree $2q+1$ and $2q$, respectively, an explicit
formula for this optimal approximation to $\mathop{\mathrm{sign}}(x)$ on
any interval $[\ell, 1]$ was found by Zolotarev. It was shown that these
rationals have remarkable convergence properties for any $q$ . For
$q=1$, this optimal approximation coincides exactly with the dynamically
weighted Halley’s method (QDWH) referenced above. For even faster
convergence than QDWH, proposed the Zolo-pd method, which uses $q=17$.
Finally, these methods all admit the same QR-based implementation trick
as QDWH.

#### Adaptive polynomial methods.

In this paper, we adopt the paradigm of Zolo-pd but with polynomials
rather than rationals of degree $(2q+1, 2q)$. This choice avoids the
need for QR factorizations, relying solely on GPU-friendly matrix-matrix
multiplications in low-precision arithmetic. While this class of methods
has not been fully developed in the numerical analysis literature,
similar ideas have been rediscovered in different guises. In an
unpublished manuscript that predates Zolo-pd, Chen and Chow describe a
rescaling strategy for Newton-Schulz. Though motivated differently,
their method is equivalent to ours for degree-3 polynomials (unlike our
work, they do not consider general odd degree). They also observe
numerical instability that prevents the method from converging to all
the way to machine precision. Using the insights of , they propose a
simple mitigation for this issue that we adopt in . Our work gives the
approach from a stronger theoretical foundation that connects to the
paradigm of Zolo-pd. Concretely, we prove that choosing an optimal
polynomial at each iteration leads to a composed polynomial that is
*globally* optimal in the sense of
[eq:matrix_minimax_problem].

Independently, a group of cryptographers developed a similar method for
approximating the scalar function $\mathop{\mathrm{sign}}(x)$ in the
context of homomorphic encryption schemes . Their focus is mainly on
tuning the analogues in their setting of the polynomial degree and
number of iterations, whereas we focus on demonstrating optimality and
efficiently constructing the update polynomials for degree $3$ and $5$.
In addition, we consider matrix-valued inputs in low-precision
arithmetic—not scalars in exact arithmetic—and we demonstrate our
method’s effectiveness within the `Muon` algorithm for training deep
neural networks.

#### Application within `Muon`.

The designers of `Muon` realized that, due to the extreme efficiency
requirements and lax accuracy requirements of their setting,
rational-based methods from the numerical analysis literature are
inapplicable. However, polynomial-based iteration schemes can take full
advantage of GPUs because they use only matrix-matrix products in
half-precision arithmetic, not inverses or QR decompositions. The
preference for speed over accuracy motivates methods that aim to quickly
produce coarse approximations, even at the cost of asymptotic
convergence. Examples include the proposals of Jordan  and You , as
discussed in . Like Chen and Chow , Jordan found that convergence in the
initial phase can be accelerated by choosing update rules that have a
large derivative near zero, so as to increase the small singular values
as much as possible at each iteration. You furthermore chose to use
different update rules at each iteration, allowing extra flexibility to
tune the trade-off between speed and accuracy. Both used degree-5
polynomials that were found through gradient descent on heuristic
objective functions. These proposals were previously compared to
Newton-Schultz[^9], but never to Chen and Chow’s method from . We find
that our method (which generalizes ) outperforms them all.

Finally, we remark that concurrent work of Grishina, Smirnov, and
Rakhuba also proposes an adaptive polynomial method that generalizes and
applies it to accelerating Muon . Like , this work does not establish
global optimality of the composed polynomial as we do in or address
finite precision considerations.

# Approximations by Compositions of Polynomials

To design a GPU-friendly method for computing
$\mathop{\mathrm{polar}}(\bm{M})$, we limit ourselves to the following
GPU-friendly operations:

1.  Linear combinations: given scalars $\beta, \gamma \in \mathbb{R}$
    and matrices $\bm{B}$ and $\bm{C}$, compute
    $\beta \bm{B} + \gamma \bm{C}$,

2.  Matrix-matrix products: compute $\bm{B} \bm{C}$.

While both these computational primitives are well-suited for parallel
computing environments, matrix-matrix products come at a higher
computational cost than linear combinations. Therefore, our method
attempts to minimize the number of matrix-matrix products. A key
observation is that we can compute *odd* monomials of
$\bm{M}= \bm{U}\bm{\Sigma}\bm{V}^\mathsf{T}$ using the following
formula:
$$\bm{M}^{2q+1} := \bm{U}\bm{\Sigma}^{2q+1} \bm{V}^\mathsf{T}= \bm{M}(\bm{M}^\mathsf{T}\bm{M})^{q}.$$
Hence, for an odd polynomial
$p(x) = a_0 x + a_1 x^3 + \cdots + a_qx^{2q+1}$ we can compute
$$p(\bm{M}) := a_0\bm{M}+ a_1 \bm{M}(\bm{M}^\mathsf{T}\bm{M}) + \cdots + a_q \bm{M}(\bm{M}^\mathsf{T}\bm{M})^{q}.$$

It has been shown that for an arbitrary polynomial $p$, one requires
$\Theta(\deg(p)^{1/2})$ products to compute $p(\bm{M})$ ; see also for
related work. This compares favorably to the naive approach that forms
all monomials in $p$ and then sums them together, which requires
$\Omega(\deg(p))$ products. However, if $p$ can be expressed as a
composition of $T$ polynomials, each of degree $d$
$$\label{eq:composition}
    p = p_T \circ p_{T-1} \circ \cdots \circ p_1,$$ then the degree of
$p$ is $d^T$, and $p(\bm{M})$ can be efficiently computed recursively by
$$\label{eq:iteration}
    \bm{X}_0 = \bm{M}, \quad \bm{X}_{t} = p_t(\bm{X}_{t-1}) \text{ for } t = 1,2,\ldots,T.$$
The final iterate is $\bm{X}_T = p(\bm{M})$, which we compute with just
$O(Td)$ matrix-matrix products.

Iterative methods for $\mathop{\mathrm{polar}}(\bm{M})$ can be seen in
this light. For instance, the degree-5 Newton-Schulz method uses the
polynomial update
$p_t(x) = \frac{15}{8}x-\frac{10}{8}x^3 + \frac{3}{8}x^5$ for each
$t = 1,\ldots,T$. The composition $p = p_T \circ \cdots \circ p_1$
approximates $\mathop{\mathrm{sign}}(x)$, and the approximation error
goes to $0$ as $T$ grows. In this paper, we ask the following question:
what choice of $p_T \circ \cdots \circ p_1$ gives the *best*
approximation to $\mathop{\mathrm{sign}}(x)$?

The method we will present is optimal in the following sense: given
lower and upper bounds $\ell$ and $u$ on the singular values of
$\bm{M}$, an odd degree $d \in \mathbb{N}$, and the number of iterations
$T \in \mathbb{N}$, our method computes the composition
$p^{\star}(\bm{M})$ that minimizes the worst-case error in the spectral
norm. That is, $$\label{eq:matrix_minimax_problem}
p^{\star} = \mathop{\mathrm{argmin}}_{\substack{p = p_T \circ p_{T-1} \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_{d}^{\mathop{\mathrm{odd}}}}}
\max_{\substack{\bm{M}\in \mathbb{R}^{m \times n} \\ \sigma(\bm{M}) \subset [\ell, u]}}
\left\|\mathop{\mathrm{polar}}(\bm{M})-p(\bm{M})\right\|_2.$$ Given that
$\mathop{\mathrm{polar}}(\bm{M}) -p(\bm{M}) = \bm{U}(\bm{I}-p(\boldsymbol{\Sigma}) )\bm{V}^\mathsf{T}$,
and by the unitary invariance of the spectral norm, we have that
[eq:matrix_minimax_problem]
is equivalent to $$\label{eq:scalar_minimax_problem}
p^{\star} \; = \; \mathop{\mathrm{argmin}}_{\substack{p = p_T \circ p_{T-1} \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}}} \,
\max_{x \in [\ell, u]}
\left|1-p(x)\right|.$$ For completeness, the equivalence
between [eq:matrix_minimax_problem]
and [eq:scalar_minimax_problem]
is proven in .






In other words, the problem given in
[eq:matrix_minimax_problem]
reduces to that of finding a “uniform” or “minimax” approximation to the
constant function $x \mapsto 1$ over the interval $[\ell, u]$, as given
in [eq:scalar_minimax_problem].
Uniform approximation on an interval by polynomials or rational
functions of a given degree is a central topic in approximation theory;
see e.g. . Here, we seek an approximation of a particular form—a
*composition* of odd polynomials of fixed degrees. In the next section,
we solve the optimization problem
of [eq:scalar_minimax_problem]
and use the solution to create `Polar Express`. (a) shows the resulting
$p^*$ polynomial labeled as `PolarExp`, as compared to the `Jordan`’s
method in , and the six iterations of `You`’s method in .

# The Polar Express

## Greedy is optimal

The key observation is that the polynomial used in each iteration can be
chosen greedily, given the choice of polynomials from the previous
iterations. For the first iteration, we choose $p_1$ so as to map the
interval $[\ell,u]$ as close to $1$ as possible. That is, it minimizes
$\max_{x \in [\ell, u]} |1 - p_1(x)|$. The image of $p_1$ will be a new
interval $[\ell_2, u_2]$, where
$$\ell_2 = \min_{x \in [\ell, u]} p_1(x) \qquad \qquad u_2 = \max_{x \in [\ell, u]} p_1(x)$$
We now pick $p_2$ to map the interval $[\ell_2, u_2]$ as close to $1$ as
possible, obtaining a new interval $[\ell_3, u_3]$ that is the image of
$[\ell, u]$ through $p_2 \circ p_1$. We continue this process for as
many iterations as desired.

The following theorem guarantees that this process finds the solution to
[eq:scalar_minimax_problem],
and thereby also
[eq:matrix_minimax_problem].
The scheme is also outlined in (b), which demonstrates the evolution of
the lower bounds $\ell_t$, the upper bounds $u_t$, and the polynomials
$p_t$ across iterations.



theoremgreedy
Let $d$ be odd and define $\ell_1 = \ell$ and $u_1 = u$. For
$t = 1,\ldots,T$ define $$\begin{aligned}
p_t &= \; \mathop{\mathrm{argmin}}_{\substack{p \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}}} \, \max_{x \in [\ell_t, u_t]} |1-p(x)| \addtocounter{equation}{1}\tag{\theequation}\label{eq:greedy}\\
\ell_{t+1} &=\; \min_{x \in [\ell_t, u_t]} p_t(x) \\
u_{t+1} &= \;\max_{x \in [\ell_t, u_t]} p_t(x)
\end{aligned}$$ The resulting composition
$p^{\star}:=p_T \circ p_{T-1} \circ \cdots \circ p_1$ is optimal and the
error is given by: $$\label{eq:optimal}
\max\limits_{x \in [\ell,u]}|1-p^{\star}(x)| \quad =\quad \min_{\substack{p = p_T \circ p_{T-1} \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}}} \,
\max_{x \in [\ell, u]}
\left|1-p(x)\right| = 1-\ell_{T+1}.$$ Furthermore the new error, lower
and upper bounds can be computed through $$\label{eq:newbounds}
    \ell_{t+1} = p_t(\ell_t),  \quad u_{t+1} = 2-\ell_{t+1},\quad \text{ and }\quad \max\limits_{x \in [\ell_t,u_t]} |1-p_t(x)| = 1-\ell_{t+1}.$$





*Proof.* See . ◻





We note that choice of the degree of each $p_1, p_2, \ldots, p_{T}$ need
not be the same for to hold. More generally, one may specify a sequence
of degrees $d_1,\ldots,d_T$ and define each $p_t$ as
$$p_t = \mathop{\mathrm{argmin}}_{\substack{p \in \mathbb{P}_{d_t}^{\mathop{\mathrm{odd}}}}} \, \max_{x \in [\ell_t, u_t]} |p(x) - 1|, \qquad \mbox{for } t = 1,\ldots, T.$$
Our theory translates entirely to this more general case. However, for
simplicity we assume $d = d_t$ for all $t = 1,\ldots, T$. Our setting is
similar to that of , which considers the closely related problem of
choosing the depth $T$ and degrees $d_1,\ldots,d_{T}$ such that $p$
approximates $\mathop{\mathrm{sign}}$ up to a prescribed error tolerance
while minimizing the number of scalar multiplications. Interestingly,
from the optimal choice of degrees is $d_t = 5$ for *almost* all
iterations. This justifies choosing $d$ to be a constant and our use of
$d=5$ in particular.



Fortunately, [eq:newbounds] shows that once $p_t$
has been found, we can compute the new lower and upper bounds
$\ell_{t+1}$ and $u_{t+1}$ and the approximation error simply by
evaluating $p_t(\ell_t)$. Hence, for any *fixed* upper and lower bounds
on the singular values of $\bm{M}$, we can *precompute* the polynomials
$p_1,\ldots,p_T$ and the bounds
$[\ell_1, u_1],\ldots,[\ell_{T+1},u_{T+1}]$. Then, applying the
iterative procedure of
[eq:iteration], the final iterate
$\bm{X}_T$ will satisfy the following error bound $$\label{eq:error}
\|\mathop{\mathrm{polar}}(\bm{M}) - \bm{X}_{T}\|_2
= \|\mathop{\mathrm{polar}}(\bm{M}) - p^{\star}(\bm{M})\|_2
\leq 1-\ell_{T+1}.$$

From the optimality guarantee of , we know that our method converges at
least as fast as the Newton-Schulz iteration of the same degree.
Combining this fact with an existing analysis of Newton-Schulz, we
immediately get the following convergence guarantee showing that our
method enjoys faster than exponential convergence.



theoremconvergence Let $\bm{M}$ be a matrix normalized
so that $\sigma(\bm{M}) \subset [\ell,1]$. Let
$\bm{X}_T = p^{\star}(\bm{M})$, where $p^{\star}$ is the polynomial from
with $d = 2q+1$. Then, we have $$\label{eq:ns_convergence}
        \|\mathop{\mathrm{polar}}(\bm{M}) - \bm{X}_T\|_2 \leq |1-\ell^2|^{(q+1)^T}.$$
Hence, for $d = 3$ the method converges quadratically and for $d = 5$
the method converges cubically.





*Proof.* See . ◻



In fact, underestimates how fast our method converges. For degree $d=5$,
our method converges about twice as fast as Newton-Schulz (compare with
). Furthermore, the same analysis applies even if $p^*$ is constructed
using a “lower bound” $\ell$ that was too high. That is, replacing
$\ell$ on the right-hand side of
[eq:ns_convergence] by
$\sigma_{\min}$, the theorem holds even if $p^*$ is constructed to be
optimal on the interval $[\ell, 1]$ for $\ell > \sigma_{\min}$.
Intuitively, when $\ell = u = 1$, the polynomial $p^*$ coincides exactly
with the Newton-Schulz method. Mistakenly setting
$\ell > \sigma_{\min}$, we obtain a polynomial that converges slower
than the optimal polynomial but faster than Newton-Schulz, so the
guarantee of still holds (cf. ).

## Finding the optimal polynomial for each iteration

shows that we can solve
[eq:scalar_minimax_problem]
by greedily choosing the optimal approximation
$p_t \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}$ for each interval
$[\ell_t,u_t]$ for $t = 1,\ldots,T$. In this section, we show how to
find each $p_t$. Since we are now focused on just one iteration, we drop
the subscripts. Given $\ell$ and $u$, we wish to solve the following
optimization problem: $$\label{eq:remez_goal}
\mathop{\mathrm{argmin}}_{\substack{p \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}}} \, \max_{x \in [\ell, u]} |1-p(x)|$$
That is, we seek a minimax or uniform approximation of the function
$x \mapsto 1$ on $[\ell, u]$ from the set of odd polynomials.
(Equivalently, we seek a minimax optimal approximation to
$\mathop{\mathrm{sign}}(x)$ on $[-u, -\ell] \cup [\ell, u]$.)

Problems of the
form [eq:remez_goal] are well-studied in
approximation theory and numerical analysis. The key mathematical
insight underlying the solution is the Equioscillation Theorem, which we
state formally for our setting in . This theorem gives a surprising
characterization of the optimal solution
of [eq:remez_goal]: an odd $p$ is
optimal for degree $2q+1$ if and only if there is a set of $q+2$
equioscillating points. This is a set of points at which $p$ achieves
its maximum approximation error $\pm E$, and for which the sign of the
error alternates. Even if the optimal approximation error $E$ is not
known in advance, finding a set of $q+2$ equioscillating points for a
given $E$ serves as a certificate that no better approximation error is
achievable. The Equioscillation Theorem is the basis of the Remez
algorithm , a general tool that can be used to find (nearly) optimal
polynomial approximations of a given degree to *any* function on any
interval. With very minor modifications to handle the constraint that
$p$ be odd, Remez can be used to directly solve
[eq:remez_goal].

However, the Remez algorithm is opaque, complex, and difficult to
implement correctly. Fortunately, we do not need the Remez algorithm in
its full generality to solve our problem. We seek only low degree
polynomials, and the function we wish to approximate is the constant
function $f(x) \equiv 1$. For $d=3$, we can derive an explicit, closed
form solution to [eq:remez_goal] using the
Equioscillation Theorem. Up to rescaling, the optimal polynomial turns
out to be the same one derived in Chen and Chow by different means . For
degree $d=5$, we present , a much simpler way of solving
[eq:remez_goal] that is
mathematically equivalent to Remez in our setting. This algorithm is
implemented in its entirety in .

We briefly describe the solution for $d=3$. We seek a polynomial of the
form $p(x) = ax + bx^3$. The Equioscillation Theorem stipulates that $p$
must have an equioscillating set of size 3. For $p$ to achieve its
maximum error at a point $x$, $x$ must be a local extremum of $p(x) - 1$
on the interval $[\ell, u]$. Thus, for $x$ to be eligible for membership
in the equioscillating set, it must either be a true local extremum of
$p(x) - 1$ that happens to lie in $[\ell, u]$, or else one of the
endpoints $\ell, u$. However, because $p$ is an odd cubic, it has at
most one true local extremum on $\mathbb{R}_{\geq 0}$. Thus, to build an
equioscillating set of three points, we must include $p$’s unique
positive local extremum *and* both endpoints. This local extremum of $p$
occurs at $\sqrt{\frac{-a}{3b}}$. Therefore, we seek $a, b$ such that
$$\label{eq:equioscillation_deg_3}
p(\ell) = 1-E, \qquad \qquad p\left(\sqrt{\frac{-a}{3b}}\right) = 1 + E, \qquad \qquad p(u) = 1-E$$
for some $E$. This is a system of three equations in three variables.
The solution $p(x) = ax + bx^3$ is most easily expressed as follows. Let
$p_{\mathop{\mathrm{NS}}}(x) = \frac{3}{2} x - \frac{1}{2} x^3$. Then
$$p(x) = \beta p_{\mathop{\mathrm{NS}}}(\alpha x), \quad \text{ where } \alpha = \sqrt{\frac{3}{u^2 + lu + \ell^2}} \quad \text{ and } \quad \beta = \frac{4}{2 + \ell u(\ell + u) \alpha^3}.$$

We now turn to the degree-5 case. The intuition of is as follows. For
any fixed set of four points $\ell 

R0.5 



When working in finite-precision arithmetic, especially the
half-precision `bfloat16` format used in deep learning, we must take
some care to avoid blowups and other problems due to numerical error. To
this end, we make three small changes to the method. These adjustments
stabilize the algorithm with a negligible effect on accuracy.
Furthermore, these adjustments can be made in the offline stage by
modifying the coefficients of our optimal polynomials.

The first issue arises when numerical round-off creates singular values
that are slightly larger than our current upper bound $u_t$. Our optimal
polynomials converge only when the singular values of $\bm{X}_t$ are
less than $u_t$. In some cases we have
$$p_t(u_t + \epsilon) > u_{t+1} + \epsilon,$$ so over many iterations, a
singular value that is slightly larger than $u_t$ large could grow to
$\infty$ instead of converging to $1$.

To fix this issue, we simply replace each polynomial $x \mapsto p_t(x)$
by $x \mapsto p_t(x / 1.01)$. This safety factor corrects for round-off
errors in previous iterations while only slightly changing the behavior
of the polynomial on the interval $[\ell_t, u_t]$, though it does cause
the singular values to converge to $0.999998$ instead of to $1$. To
correct for this, the safety factor can be omitted in the final
iteration.

The second issue was identified in and addressed in the context of
polynomial iterations by Chen and Chow . In general, iterative methods
for $\mathop{\mathrm{polar}}(\bm{M})$ aim to increase each singular
value relative to the largest singular value; while
$\sigma_{\min}(\bm{X}_0) \ll \sigma_{\max}(\bm{X}_0)$, after enough
iterations,
$\sigma_{\min}(\bm{X}_t) \approx \sigma_{\max}(\bm{X}_t) \approx 1$.
However, the convergence of each singular value to $\sigma_{\max}$ may
not be monotonic. Over the domain $[\ell_t, u_t]$, our optimal
polynomial $p_t$ oscillates repeatedly between $\ell_{t+1}$ and
$u_{t+1}$, so some singular values that are near $u_t$ may get mapped
down to $\ell_{t+1}$. It so happens that this non-monotonicity—even at a
single iteration—can cause loss of precision. That is, problems occur if
$$\frac{p_t(\sigma_i)}{\sigma_i} \ll \frac{\max\limits_{x \in [\sigma_{\min},\sigma_{\max}]}p_t(x)}{\sigma_{\max}},$$
where $0 \leq \sigma_{\min} \leq \sigma_i \leq \sigma_{\max}$ are
singular values of $\bm{X}_t$ . In the extreme case $p_t(\sigma_i) 

**input:** Matrix $\bm{M}$, iteration count $T$, degree $d$, approximate
lower bound $\ell$.

**output:** An approximation $\bm{X}_T$ to
$\operatorname{polar}(\bm{M})$.



$\ell_1 = \ell$, $u_1 = 1$. Solve using Remez ():
$p_t = \mathop{\mathrm{argmin}}\limits_{p \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}}\max\limits_{x \in \left[\max(\ell_t, u_t/10),\, u_t \right]} |1-p(x)|$
 $p_t \gets p_t(\cdot / 1.01)$

$\ell_{t+1} \gets p_t(\ell_t)$,$u_{t+1} \gets 2-\ell_{t+1}$ Set
$\bm{X}_0 = \bm{M}/(\|\bm{M}\|_\text{F}+ 10^{-2})$. $\bm{X}_{t} = p_t(\bm{X}_{t-1})$
**return** $\bm{X}_T$.





We give the pseudocode of our proposed method for any degree in  and
provide a complete implementation in [our
repository](https://github.com/NoahAmsel/PolarExpress). We give the
specific version of the `Polar Express` with degree $d=5$ and
$\ell = 10^{-3}$ used in our GPT experiments in . Our algorithm first
computes the polynomials $p_1,\ldots,p_{T}$ of in full precision using
the results of (or the Remez algorithm for $d > 5$). This stage is
offline because the coefficients of the polynomials are only computed
and stored once. For every subsequent call to the algorithm, these
coefficients are reused and the offline stage is skipped. For instance,
in these polynomials have been precomputed and stored in the variable
`coeffs_list`.

The polynomial $p^{\star} := p_{T} \circ \cdots \circ p_1$ is then
applied to the input matrix $\bm{M}$ in the online stage. The online
stage can be performed in lower precision (`bfloat16`) for greater speed
on a GPU. Horner’s rule can be used to carry out each iteration. For
instance, if $p_t = ax + bx^3 + cx^5$, then
$$\bm{X}_t = \bm{X}_{t-1}\left(a\bm{I}+ \bm{Y}_{t-1} \left(b\bm{I}+ c\bm{Y}_{t-1}\right)\right)$$
where $\bm{Y}_{t-1} = \bm{X}_{t-1}^\top\bm{X}_{t-1}$.

A simple implementation of the offline stage of is given in . For deep
learning applications, we recommend using $d = 5$ and $T = 5$ or $6$
with $\ell_1 = 10^{-3}$. With these parameters, the offline stage as
implemented in gives the polynomials encoded in `coeffs_list` in . All
told, our proposal for `Muon` is to apply the composition of these
polynomials to $\bm{M}/ (\|\bm{M}\|_F + 10^{-2})$.

# Numerical Experiments

## Convergence of `Polar Express`



We compare the performance of `Polar Express` against degree-5
Newton-Schulz and the methods of Chen and Chow , Jordan , and You .

We first study an idealized scenario where the spectrum of the input
matrix is known exactly. We generate a random matrix whose singular
values are evenly spaced on a logarithmic scale between $10^{-6}$ and
$1$. The right and left singular vectors are chosen at random. The left
panel of shows the results. Since all the methods in this plot use
degree-5 polynomials, their computational and runtime costs are all
proportional to the number of iterations. As expected, Newton-Schulz
converges but makes almost no progress for the first 17 iterations.
Jordan’s method rapidly achieves an error of $\approx 0.3$ after just 11
iterations, but ceases to converge further. You’s method, which is
difficult to see on the plot because it is only defined for six
iterations, converges at a similar rate as Jordan’s method. When
`Polar Express` is instantiated with $\ell = \sigma_{\min}$, it
dominates the other methods at every iteration, achieving excellent
accuracy after just 11 iterations and converging about twice as fast as
Newton-Schulz to any given error. Even when the lower bound on
$\sigma_{\min}$ is wrong by two orders of magnitude in either direction,
the method remains competitive, though it does not actually outperform
Jordan’s method until iteration 13 or 14.



Next we test the methods’ performance on a matrix from a real-world
application, namely, the gradient of a weight matrix from the fourth
transformer block of a GPT-2 architecture with respect to a language
modeling objective on a batch of text from the Tiny Shakespeare dataset
. The right panel of shows the results. Once again, the best-tuned
version of `Polar Express` outperforms the other methods. This time, we
see that setting $\ell$ to be many orders of magnitude too small can
delay convergence significantly, and make `Polar Express` less
competitive as compared to Jordan’s method.

For most other weight matrices in this GPT-2 model, the methods all take
more than 10 iterations to converge in the spectral norm. The spectral
error is large if there is even one outlying singular value that is far
from $1$. However, for some applications, we may be satisfied with a
weaker notion of convergence, like the relative Frobenius norm. shows
the performance of various methods on this metric. We use gradient
matrices of the same model, but from two different layers. In addition,
we compare the degree-5 methods to Chen and Chow’s degree-3 method. To
make this comparison fair, we measure the number of matrix-matrix
products performed by each method instead the number of iterations. We
find that `Polar Express` can once again dominate the other methods
across iterations. Chen and Chow’s method is also quite competitive, and
the remaining methods behave much as in .





## Training GPT-2

In our final experiment, we compare the performance of using our
`Polar Express` method given in  inside the `Muon` algorithm versus
Jordan’s  and You’s  methods.[^10] We train two different GPT-2 models:
$$\begin{aligned}
    \texttt{GPT-Small}: & \quad  n_{\text{embd}} = 768, \quad n_{\text{layer}} = 12, \quad n_{\text{head}} = 12 \\
   \texttt{GPT-Large}: & \quad n_{\text{embd}} = 1280,  \quad n_{\text{layer}} = 36, \quad n_{\text{head}} = 20 
\end{aligned}$$ and a vocabulary size of $50{,}257$, using a context
length of $1024$. Training is performed on 1B tokens from the
FineWeb dataset , using a batch size of
32 and a single epoch. All models are trained with mixed precision
(`bfloat16`) on 4 H100 GPUs. For all methods we use the learning rate
schedule proposed in , consisting of a constant phase for the first 40%
of training steps followed by a linear decay. All methods for the matrix
sign computations are performed in `float16b` precision and use five
iterations.

We apply `Muon` selectively to certain layers of the model. Following
the nano-gpt implementation , we assign `Muon` to all parameters with at
least two dimensions (typically weight matrices, and excluding RMS norm
parameters), excluding the embeddings, unembeddings, and the positional
encodings. These excluded parameters are instead optimized with AdamW.

and shows the resulting runs of each method in terms of validation loss
and training loss on the `GPT-Large` and `GPT-Small` models,
respectively. In both figures we can see that `muon-PolarExp` achieves a
better validation and training loss than `muon-Jordan` or `muon-You` for
every learning rate. Since each iteration of the different matrix sign
methods are equally expensive (since they all apply a degree 5
polynomial), improved validation loss in terms of epochs also translates
to an improved loss in terms of wall clock time (see bottom right of ).
The advantage is remarkably consistent across all learning rates and
epochs.

We also experimented with adding weight decay $0.1$ to the model,
keeping all else the same, in . Here we find again that `muon-PolarExp`
achieves a better validation and training loss for every learning rate,
except one ($lr = 10^{-2}$) where its performance matches that of
`muon-Jordan`.

# Acknowledgements

This work was partially supported by NSF awards 2045590 and 2234660.
Computations were run at facilities supported by the Scientific
Computing Core at the Flatiron Institute, a division of the Simons
Foundation.

# Proof of 

The aim of this section is to prove . We begin with a result that
provides a few essential properties for the the polynomial solving
[eq:scalar_minimax_problem]
when $T = 1$. This result is known as Chebyshev’s theorem or the
equioscillation theorem .



 Let $d = 2q+1$ and
$u, \ell > 0$. Consider the problem $$\label{eq:oneiteration}
        \min\limits_{p \in \mathbb{P}_{d}^{\mathop{\mathrm{odd}}}} \max\limits_{x \in  [\ell,u]} |1 - p(x)|.$$
There exists a unique polynomial
$p^{\star} \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}$ solving
[eq:oneiteration]. Furthermore,
$p^{\star}$ is the unique solution to the above problem if and only if
there exist $q+2$ distinct points
$\{x_0,\ldots,x_{q+1}\} \subset [\ell,u]$ such that
$$1 - p^{\star}(x_i) \;=\; \eta(-1)^i \max\limits_{x \in [\ell,u]} |1 - p^{\star}(x)|, \quad \mbox{for} \; i = 0,\ldots,q+1,$$
for $\eta = 1$ or $\eta = -1$.





*Proof.* A discussion can be found in . Here we include a formal proof
for completeness.

By Chebyshev’s Theorem it is sufficient to show that
$\mathbb{P}_d^{\mathop{\mathrm{odd}}}$ satisfies the Haar condition: any
non-zero
$p \in \mathbb{P}_d^{\mathop{\mathrm{odd}}} = \mbox{span}\{x, \ldots, x^3, \ldots, x^{2q+1}\}$
can have at most $q$ roots in $[\ell,u]$.

Since $\deg(p) = d = 2q+1$ we know that $p$ can have at most $2q+1$
roots in $\mathbb{R}$. However, since $p(0) = 0$ and $p(x) = -p(-x)$ we
know that $p$ has one root at zero, and the remaining roots come in
symmetric pairs $(x,-x)$. Because of this, $p$ can have at most $q$
roots in the positive orthant, and thus it can have at most $q$ roots in
$[\ell,u] \subset (0, \infty)$. Hence,
$\mathbb{P}_d^{\mathop{\mathrm{odd}}}$ satisfies the Haar condition,
which yields the desired result. ◻



The proof of will be by induction on $T$. We begin by establishing the
base case, $T=1$, which is handled by the following result.




Let $u, \ell > 0$ and define
$$p^{\star} := \mathop{\mathrm{argmin}}\limits_{p \in \mathbb{P}_d^*} \max\limits_{x \in [\ell,u]}|1-p(x)|.$$
Then
$$p^{\star}(\ell) = \min\limits_{x \in [\ell,u]} p^{\star}(x), \quad \max\limits_{x \in [\ell,u]} p^{\star}(x) = 2- p^{\star}(\ell), \text{ and }\max\limits_{x \in [\ell,u]}|1-p^{\star}(x)| = 1-p^{\star}(\ell).$$





*Proof.* Throughout the proof we assume $d = 2q+1$. We begin with
proving $$p^{\star}(\ell) = \min\limits_{x \in [\ell,u]} p^{\star}(x).$$
Consider the polynomial $e(x) := 1-p^{\star}(x)$. The proof will contain
three steps. We first rule out the trivial case that $p^{\star} \neq 0$,
since $p(x) = \frac{2}{\ell + u}x$ would then be a better approximation.
Hence, $p^{\star}$ cannot be the zero polynomial.

*Step 1: $e(x)$ has exactly $q$ stationary points inside the open
interval $(\ell,u)$.*

Note that $e(x)$ has at most $2q$ stationary points in $\mathbb{R}$,
since its derivative $e'(x)$ is a polynomial of degree $2q$.
Furthermore, since $p^{\star}$ is odd, we have that $e'(x) = -p'(x)$ is
even of degree $2q$, and thus can have at most $q$ stationary points
contained in $(0,+\infty)$. Hence, there can be at *most* $q$ stationary
points of $e(x)$ inside the interval $[\ell,u]$.

By there are $q + 2$ points $x_0,\ldots,x_{q+1} \in [\ell,u]$ where
$e(x)$ is maximized or minimized in $[\ell,u]$. These points are either
stationary points or they are endpoints of the interval $[\ell,u]$. Let
$n_{\text{ext}}$ be the number of stationary points and
$n_{\text{stat}}$ be the number of endpoints in the set
$\{x_0,\ldots,x_{q+1}\}$. Since a point can be both a stationary point
and an endpoint we have $q+2 \leq n_{\text{end}} + n_{\text{stat}}$.
However, $n_{\text{end}} \leq 2$ and $n_{\text{stat}} \leq q$, which
follows from the previous paragraph where we showed that there are at
most $q$ stationary points of $e(x)$ in $[\ell,u]$. So
$n_{\text{end}} + n_{\text{stat}} \leq q + 2$, and consequently we must
have $n_{\text{end}} = 2$ and $n_{\text{stat}} = q$, as required.

*Step 2: $x = \ell$ is a maximum of $e(x)$ on the interval
$[\ell,u]$*

By and the discussion from Step 1, we know that $|e(x)|$ is maximized at
$q+2$ points inside $[\ell,u]$ and $q$ of these points are contained
inside the open interval $(\ell,u)$. Hence, $x = \ell$ must either be a
maximum or a minimum of $e(x)$. We will show that $x = \ell$ must be a
maximum by contradiction.

Suppose $x = \ell$ was a minimum of $e(x)$ on $[\ell,u]$. First note
that $p^{\star}$ is trivially non-negative on $[\ell,u]$, or else
$p(x) = 0$ would be a better polynomial. Hence, since $p^{\star}(0) = 0$
we must have ${p^{*}}'(\delta) > 0$ for some $\delta \in [0,\ell]$, or
else the zero polynomial $p(x) = 0$ would be a better approximation.
Hence, for some $\delta \in [0,\ell]$ we have $e'(\delta)*Step 3: Obtaining the desired equalities*

Since $e(x)$ has a maximum in $[\ell,u]$ at $x = \ell$, we have
$p^{\star}(\ell) = \min\limits_{x \in [\ell,u]} p^{\star}(x)$. The other
two equalities are immediate consequences of the equioscillation
property of $p^{\star}$ and that $x = \ell$ is a minimum of $p^{\star}$
over the set $[\ell,u]$. ◻



With the above-mentioned result in hand, we are ready to prove .



*Proof.* The proof of
[eq:newbounds] is an immediate
consequence of , since for each $t = 1,\ldots,T$, $p_t$ is the optimal
approximation in $\mathbb{P}_d^{\mathop{\mathrm{odd}}}$ to
$x \mapsto 1$.

We now proceed with the proof of
[eq:optimal], which will be by
induction. The proof for $T = 1$ is an immediate consequence of and we
also have $p^{\star}(\ell) = \ell_2$ by
[eq:newbounds]. Now suppose the result
is true for all $t \leq T-1$. For $t = 1,\ldots,T-1$, note that the
image of $p_t$ on $[\ell_t,u_t]$ is exactly $[\ell_{t+1},u_{t+1}]$ by
i). Hence, if we define $g(x) := p_{T-1} \circ \cdots \circ p_1(x)$,
then the image of $g$ on $[\ell,u]$ is $[\ell_{T},u_{T}]$. Furthermore,
by i) we also have $g(\ell) = \ell_T$. Pick any $f$ such that $f \neq g$
and $$f = \widetilde{p}_{T-1} \circ \cdots \circ \widetilde{p}_1,$$ for
some
$\widetilde{p}_1,\ldots,\widetilde{p}_{T-1} \in \mathbb{P}_{d}^{\mathop{\mathrm{odd}}}$.
Let the image of $f$ on $[\ell,u]$ be $[a,b]$. We will prove that
$\frac{a}{b} \leq \frac{\ell_T}{u_T}$ by contradiction.

Suppose $\frac{a}{b} > \frac{\ell_T}{u_T}$. Define $c = \frac{2}{a+b}$.
Then, the image of the scaled function $c f$ on $[\ell,u]$ is $[ca,cb]$
and $cf$ satisfies
$$\max\limits_{x \in [\ell,u]}|1-cf(x)| = \max\left\{1-ca,cb-1\right\} = \frac{b-a}{a+b}.$$
Recall by our inductive hypothesis, we have
$\max\limits_{x \in [\ell,u]}|1-g(x)| = 1-\ell_{T} = u_T-1$ where the
second equality holds by
[eq:newbounds]. It follows that
$$\begin{aligned}
        \frac{a}{b} &> \frac{\ell_T}{u_T} \\
        \Leftrightarrow \frac{a}{b} &> \frac{\ell_T}{2-\ell_T}\\
        \Leftrightarrow \ell_T & \frac{b-a}{a+b}\\
        \Leftrightarrow \max\limits_{x \in [\ell,u]}|1-g(x)|&> \max\limits_{x \in [\ell,u]}|1-cf(x)|,
    
\end{aligned}$$ which leads to a contradiction to our inductive
hypothesis that $g$ is optimal. Hence, we must have
$\frac{a}{b} \leq \frac{\ell_T}{u_T}$.

Consequently, using that $\frac{a}{b} \leq \frac{\ell_T}{u_T}$, we will
show that for any
$\widetilde{p}_T \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}$ and for any
$f = \widetilde{p}_{T-1} \circ \cdots \circ \widetilde{p}_1$
$\widetilde{p}_T \circ f$ cannot be a better approximation than
$p_T \circ g$. In particular, we have

$$\begin{aligned}
        \max\limits_{x \in [\ell,u]} |1-\widetilde{p}_{T}(f(x))|&\geq\min\limits_{p \in \mathbb{P}_{d}^*}\max\limits_{x \in [\ell,u]}|1-p(f(x))| \\
        &= \min\limits_{p \in \mathbb{P}_{d}^*}\max\limits_{x \in [a,b]}|1-p(x)| \\
        &= \min\limits_{p \in \mathbb{P}_{d}^*}\max\limits_{x \in [a/b,1]}|1-p(x)|\\
        &\geq \min\limits_{p \in \mathbb{P}_{d}^*} \max\limits_{x \in [\ell_T/u_T,1]}|1-p(x)| \\
        &= \min\limits_{p \in \mathbb{P}_{d}^*} \max\limits_{x \in [\ell_T,u_T]}|1-p(x)| \\
        & = \min\limits_{p \in \mathbb{P}_{d}^*} \max\limits_{x \in [\ell,u]}|1-p(g(x))|\\
        & =  \max\limits_{x \in [\ell_T,u_T]}|1-p_{T}(g(x))| = 1-p_T(\ell_{T}) = 1-\ell_{T+1},
    
\end{aligned}$$ where the second and third equality follow by changing
variables $y = x/b$ so that
$$\min\limits_{p \in \mathbb{P}_{d}^*}\max\limits_{x \in [a,b]}|1-p(x)| = \min\limits_{p \in \mathbb{P}_{d}^*}\max\limits_{y \in [a/b,1]}|1-p(by)| =\min\limits_{p \in \mathbb{P}_{d}^*}\max\limits_{y \in [a/b,1]}|1-p(y)|$$
and this last equality follows because the space $\mathbb{P}_{d}^*$ is
invariant under input rescaling; that is, for any $b \neq 0$, the map
$x \mapsto  b x$ preserves the space
$\mathrm{span}\{x, x^3, \dots, x^d\}$. This concludes the proof. ◻



# Proof of 

In this section we provide the proof of the convergence guarantee stated
in .



*Proof.* Define
$$p^{\star} = \mathop{\mathrm{argmin}}_{\substack{p = p_T \circ p_{T-1} \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_d^*}} \,
\max_{x \in [\ell, u]}
\left|1-p(x)\right|.$$ Then returns $\bm{X}_T = p^{\star}(\bm{M})$. Let
$h \in \mathbb{P}_q$ be $[q/0]$ Padé-approximant to $(1-x)^{-1/2}$ and
define $p(x) = xh(1-x^2) \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}$.
Define $f = p \circ \cdots \circ p$ as the composition of $p$ with
itself $T$ times. Then, by , , and $f(x) \geq 0$ for $x \geq 0$ we have
$$\begin{aligned}
        \|\mathop{\mathrm{sign}}(\bm{M}) - \bm{X}_T\|_2 &\leq \max\limits_{x \in [\ell,1]}|1-p^{\star}(x)| \\
        &\leq \max\limits_{x \in [\ell,1]}|1-f(x)|\\
        & \leq \max\limits_{x \in [\ell,1]} \left[\frac{|1-x^2|^{(d+1)^T}}{1+f(x)}\right]\\
        & \leq |1-\ell^2|^{(d+1)^T},
    
\end{aligned}$$ as required. ◻



# Proof of equivalence between [eq:matrix_minimax_problem] and [eq:scalar_minimax_problem]

In this section we provide a proof for the equivalence between
[eq:matrix_minimax_problem]
and [eq:scalar_minimax_problem].
It is sufficient to show that for any fixed polynomial $p$ we have
$$\varepsilon_1 := \max_{\substack{\bm{M}\in \mathbb{R}^{m \times n} \\ \sigma(\bm{M}) \subset [\ell, u]}}
\left\|\mathop{\mathrm{polar}}(\bm{M})-p(\bm{M})\right\|_2 =
\max_{x \in [\ell, u]}
\left|1-p(x)\right| := \varepsilon_2.$$ For any fixed $\bm{M}$, by the
unitary invariance of the spectral norm we immediately have
$$\left\|\mathop{\mathrm{polar}}(\bm{M})-p(\bm{M})\right\|_2 = \max\limits_{\sigma_i \in \sigma(\bm{M})}|1-p(\sigma_i)|
\leq\max\limits_{x \in [\ell, u]}
\left|1-p(x)\right|.$$ Consequently, $\varepsilon_1 \leq \varepsilon_2$.

Suppose that $x^* \in [\ell,u]$ is chosen so that
$|1-p(x^*)| = \max_{x \in [\ell, u]}
\left|1-p(x)\right|.$ Without loss of generality, assume $m \geq n$.
Letting $\bm{M} = x^* \bm{U}\bm{V}^\mathsf{T}$, for any matrix
$\bm{U} \in \mathbb{R}^{m \times n}$ and
$\bm{V} \in \mathbb{R}^{n \times n}$ with orthonormal columns, and
noting $\mathop{\mathrm{polar}}(\bm{M}) = \bm{U}\bm{V}^\mathsf{T}$
yields $$\begin{aligned}
  \varepsilon_1 &\geq  \|\mathop{\mathrm{polar}}(\bm{M})-p(\bm{M})\|_2  \\ & = \|\bm{I}_n - p(x^*) \bm{I}_n\|_2\\
    &= |1-p(x^*)|\\
    &= \max_{x \in [\ell, u]} 
\left|1-p(x)\right| \; = \varepsilon_2
\end{aligned}$$ Consequently, $\varepsilon_1 \geq \varepsilon_2$. Hence,
$\varepsilon_1 = \varepsilon_2$, as desired.

# Remez algorithm

In this section, we show in detail how to solve
[eq:remez_goal]. By , this also gives
a solution to
[eq:scalar_minimax_problem].
We give a closed form solution for $d=3$. We then describe how the Remez
algorithm can be used to approximate $p_t$ for arbitrary $d$. We finish
with , a simplified version of Remez for solving
[eq:remez_goal] with $d=5$. Recall
[eq:remez_goal]:
$$\mathop{\mathrm{argmin}}_{\substack{p \in \mathbb{P}_d^{\mathop{\mathrm{odd}}}}} \, \max_{x \in [\ell, u]} |1-p(x)|$$

We begin with the case when $d = 3$. In this case, there is a simple
closed form for the optimal odd polynomial
$p^{\star} \in \mathbb{P}_3^{\mathop{\mathrm{odd}}}$ as described in .
On a given interval $[\ell,u]$, the optimal approximation to the
constant function $x \mapsto 1$ is given by the scaled and shifted
Newton-Schulz polynomial
$p_{\mathop{\mathrm{NS}}}(x) = \frac{3}{2} x - \frac{1}{2} x^3$:
$$p^{\star}(x) = \beta p_{\mathop{\mathrm{NS}}}(\alpha x), \text{ where } \alpha = \sqrt{\frac{3}{u^2 + \ell u + \ell^2}} \text{ and } \beta = \frac{4}{2 + \ell u(\ell + u) \alpha^3}.$$
One can verify that this polynomial satisfies the equioscillation
condition from at $x = \ell, \frac{1}{\alpha}, u$ as given in
[eq:equioscillation_deg_3],
with $\sqrt{-a/(3b)} = 1/\alpha$ and $E = \beta - 1$. Therefore, it must
necessarily be the optimal approximation from
$\mathbb{P}_3^{\mathop{\mathrm{odd}}}$. Note that when $u=1$, the
function $x \mapsto p_{\mathrm{NS}}(\alpha x)$ is the same polynomial
derived by Chen and Chow .

Unfortunately, for larger $d$, finding closed form expressions for
optimal approximations from $\mathbb{P}_d^{\mathop{\mathrm{odd}}}$
becomes challenging, and we know of no closed form solution. However, we
can approximate the optimal polynomial using the Remez algorithm. Let
$d = 2q+1$. Again recalling , the optimal polynomial must satisfy the
equioscillation property at a set of $q+2$ points, as in
[eq:equioscillation_deg_3].
The Remez algorithm finds the equioscillation points
$A = \{x_0,\ldots,x_{q+1}\}$ from by iteratively refining a sequence of
trial points $A^{(k)} = \{x_0^{(k)}, \ldots, x_{q+1}^{(k)}\}$ so that
$A^{(k)}$ converges to $A$. From the sequence of trial points $A^{(k)}$
the algorithm also finds a sequence of polynomials $p^{(k)}$ so that
$p^{(k)}$ converges to the optimal polynomial. The convergence is very
fast, and usually 10 iterations is sufficient to converge to the optimal
polynomial up to double precision machine epsilon . More commonly, the
Remez algorithm is used to find optimal polynomial approximations to
general continuous functions where $d \approx 100$ or even
$d\approx 1000$. However, because the polynomial we build to approximate
$\mathop{\mathrm{sign}}(x)$ is a composition of polynomials, each of
which has a low degree, in our setting the degree $d$ is small, usually
$d = 5$. For $d=5$ the Remez algorithm simplifies significantly. We now
describe this simplified algorithm.

We first choose an initial set of trial points $A^{(1)}$, which ideally
should come close to satisfying the equioscillation property. From , the
unique optimal approximation
$p^{\star} \in \mathbb{P}_{5}^{\mathop{\mathrm{odd}}}$ satisfies the
equioscillation property at four points in $[\ell, u]$. Since the
function we wish to approximate is constant, the equioscillation points
must be extrema of $p^{\star}$ on $[\ell, u]$. Because $p^{\star}$ is a
odd quintic, it can have at most two local extrema on the positive real
line, and thus at most two local extrema on $[\ell, u]$. The other two
equioscillation points must therefore be the endpoints $\ell$ and $u$.
Since we know that $\ell$ and $u$ must be among the true equioscillation
points, we always include them in our set of trial points. For
notational simplicity, we call the other two points $q$ and $r$. We
initialize $q_1 = \frac14 \ell + \frac34 u$ and
$r_1 = \frac34 \ell + \frac14 u$, since we observe that as $\ell \to u$
these are approximately the other two equioscillation points.

We now show how to refine a candidate set of trial points
$A^{(k)} = \{\ell, q_k, r_k, u\}$ to produce
$A^{(k+1)} = \{\ell, q_{k+1}, r_{k+1}, u\}$ as well as an approximately
equioscillating polynomial $p_k$. For any fixed set of trial points, we
can find a degree-5 odd polynomial $p_k(x) = a_k x + b_k x^3 + c_k x^5$
that satisfies $$\label{eq:remez_step_one}
p_k(\ell) = 1-E_k, \quad p_k(q_{k}) = 1+E_k, \quad p_k(r_k) = 1-E_k, \quad p_k(u) = 1+E_k$$
for some $E_k$ by solving a linear system in $a_k, b_k, c_k$ and $E_k$.
This can be rewritten as follows: $$\label{eq:linearsystem}
     \begin{bmatrix} 
    \ell & \ell^3 & \ell^5 & 1 \\
    q_{k} & q_{k}^3 & q_{k}^5 & -1 \\
    r_k & r_k^3 & r_k^5 & 1\\
    u & u^3 & u^5 & -1 
    \end{bmatrix} \begin{bmatrix} a_k \\ b_k \\ c_k \\ E_k \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1\end{bmatrix}.$$
If $A^{(k)}$ were the extrema of the error function
$e_k(x) = 1 - p_k(x)$ on $[\ell, u]$, then they would be an
equioscillating set for $p_k$, and $p_k$ would be the solution.
Therefore, to refine $A^{(k)}$, we find the extrema of
$e_k(x) = 1 - p_k(x)$. These can occur at $\ell, u$ and the roots of
$e_k'(x)$. Setting $e_k'(x) = 0$ yields the quartic equation
$5c_kx^4 + 3b_kx^2 + a_k = 0$, whose two solutions are given explicitly
by the *quadratic* formula after the substitution $y = x^2$. We set
$q_{k+1}$ and $r_{k+1}$ to be the solutions to this equation and let
$A^{(k+1)} = \{\ell,q_{k+1},r_{k+1},u\}$. We repeat the procedure until
$|E_k| := \max\limits_{x \in [\ell,u]} |1-p_k(x)| \approx \max\limits_{x \in [\ell,u]} |1-p_{k+1}(x)|=:|E_{k+1}|$.

We note that the matrix appearing in
[eq:linearsystem] is a Vandermonde
matrix. Vandermonde matrices become notoriously ill-conditioned as the
degree grows large . However, since in our setting we choose $d$ to be
small, there is no ill-conditioning due to large degrees. Instead, we
observe ill-conditioning when $\ell \approx u$. However, as
$\ell/u \to 1$ the optimal polynomial will converge to the polynomial
$\frac{x/u}{8}\left(15-10(x/u)^2 + 3(x/u)^4\right)$, which can be
verified by noting that as $\ell/u \to 1$ all equioscillation points
$x_0,x_1,x_2,x_3$ must converge to $u$. For general $d = 2q+1$, the
polynomial will converge to $(x/\ell)h(1-(x/\ell)^2)$ where
$h \in \mathbb{P}_q$ is the $[q/0]$ Padé approximant to $(1-x)^{1/2}$ .
In fact, this polynomial is extremely close to the optimal polynomial
for sufficiently large $\ell$. To see this, let $p^{\star}$ be the
optimal approximation from $\mathbb{P}_{5}^{\mathop{\mathrm{odd}}}$ and
let $p(x) = \frac{x/u}{8}\left(15-10(x/u)^2 + 3(x/u)^4\right)$. Then,
$$\begin{aligned}
    \max\limits_{x \in [\ell,u]}|p^{\star}(x) - p(x)| &\leq \max\limits_{x \in [\ell,u]}|1 - p(x)| + \max\limits_{x \in [\ell,u]}|1-p^{\star}(x)| \\
    &\leq2\max\limits_{x \in [\ell,u]}|1 - p(x)| \\
    &\leq 2 \left(1-\ell/u\right)^3.
\end{aligned}$$ where we invoked and the fact that $p^{\star}$ is the
optimal approximation to $x\mapsto 1$ from
$\mathbb{P}_{5}^{\mathop{\mathrm{odd}}}$. Hence, when
$\ell/u \geq 1-\epsilon_d^{1/3}$, where
$\epsilon_{\text{double}} \approx 1.1 \times 10^{-16}$ is the double
precision machine epsilon, then
$|p^{\star}(x) - p(x)| \leq 2 \epsilon_{\text{double}}$. In other words,
up to double precision machine epsilon, $p^{\star}$ is equal to $p$.
Therefore, whenever $\ell/u \geq 1-\epsilon_{\text{double}}^{1/3}$ the
algorithm simply returns the Padé approximant (that is, the scaled
Newton-Schulz polynomial).

The full algorithm is given in . In our experiments, we never observed
taking more than five iterations to converge. This algorithm is
implemented in full in .



**input:** interval $[\ell,u]$ for $u > \ell > 0$.  
**output:** Approximation $p \in \mathbb{P}_5^{\mathop{\mathrm{odd}}}$
to
$p^{\star} = \mathop{\mathrm{argmin}}\limits_{p \in \mathbb{P}_5^{\mathop{\mathrm{odd}}}}\max\limits_{x\in[\ell,u]}|1-p(x)|$.



**define**  $\epsilon_{\text{double}}= 1.11 \times 10^{-16}$ Return
$p(x) = \frac{x/u}{8}\left(15-10(x/u)^2 + 3(x/u)^4\right)$
$q_1 = \frac14 \ell + \frac34 u, \quad r_1 = \frac34 \ell + \frac14 u$.
$E_0 = \infty, \quad E_{-1} = -\infty$ $k \gets 0$ $k \gets k+1$
$\begin{bmatrix}a_k \\ b_k \\ c_k \\ E_k\end{bmatrix} = \begin{bmatrix} 
    \ell & \ell^3 & \ell^5 & 1 \\
    q_{k} & q_{k}^3 & q_{k}^5 & -1 \\
    r_k & r_k^3 & r_1^5 & 1\\
    u & u^3 & u^5 & -1 
    \end{bmatrix}^{-1}\begin{bmatrix}1 \\ 1 \\ 1 \\ 1\end{bmatrix}$
$q_{k+1} = \sqrt{\frac{-3b_k - \sqrt{9b_k^2 - 20a_kc_k}}{10c_k}}, \quad r_{k+1} = \sqrt{\frac{-3b_k + \sqrt{9b_k^2 - 20a_kc_k}}{10c_k}}$
Return $p(x) = a_kx + b_kx^3 + c_kx^5$





# Initialization for Matrices with Large Spectral Gaps

In , we constructed a sequence of polynomials that is adapted to the
range of the singular values $[\ell, u]$. Assuming nothing else about
the input, these polynomials are optimal since they provide a good
approximation to $1$ across the entire interval. However, in many
applications, the spectrum has large gaps; that is, there are several
large outlying singular values that are well-separated from the rest.
For these matrices, it is not necessary for the polynomial to be
accurate on the entire interval $[\ell, u]$, only on the range of the
small singular values plus a few other isolated points. In this section,
we take advantage of this structure to accelerate our method by
preprocessing the matrix to eliminate the largest singular values.

The first step is to find small intervals containing each of these large
singular values. To find lower bounds, we use subspace iteration, which
is a generalization of the power method that approximates multiple
singular values simultaneously. Fix $k$, the number of singular values
we wish to eliminate. Letting $\sigma_1 \geq \cdots \geq \sigma_n$
denote the singular values of $\bm M$, subspace iteration produces
estimates $\tilde \sigma_1 \geq 
\cdots \geq \tilde \sigma_k$ satisfying $\sigma_i \geq \tilde \sigma_i$
for all $i \in 1,\ldots,k$.[^11] To find upper bounds on each
$\sigma_i$, we can use the fact that
$\|\bm M\|_\text{F}^2 = \sum_{j=1}^n \sigma_j^2$ as follows:
$$\label{eq:sigma_bound}
\sigma_i^2
= \|\bm M\|_\text{F}^2 - \sum\limits_{\substack{j  = 1 \\ j \neq i}}^n \sigma_j^2
\leq \|\bm M\|_\text{F}^2 - \sum\limits_{\substack{j = 1 \\ j \neq i}}^k \sigma_j^2
\leq \|\bm M\|_\text{F}^2 - \sum\limits_{\substack{j = 1 \\ j \neq i}}^k \tilde \sigma_j^2$$
That is, for each $i \in [n]$,
$$\sigma_i \in \left[\tilde \sigma_i, \,\, \sqrt{\|\bm M\|_\text{F}^2 - \sum\limits_{\substack{j = 1 \\ j \neq i}}^k \tilde \sigma_j^2}\right]$$
Setting $i=k+1$, the above also provides an upper bound for the tail of
the spectrum, $\sigma_{k+1}, \ldots, \sigma_n$.

The second step is to find an odd polynomial that well-approximates the
constant function on each of these intervals and on the tail
simultaneously. For simplicity, we treat only the $k=1$ case here.
Assume that $\bm{M}$ is normalized to $\|\bm{M}\|_\text{F}= 1$ and let
$z = \tilde \sigma_1$ be the lower bound produced by subspace iteration
(which reduces to the power method in this case). Then
[eq:sigma_bound] gives
$\sigma_1 \in [z, 1]$ and
$\sigma_2, \ldots, \sigma_n \leq \sqrt{1-z^2}$. Assume that these
intervals do not overlap, that is,
$\sqrt{1- z^2} \leq z \Leftrightarrow z \geq 1/\sqrt{2}$. Then we
construct the unique odd cubic polynomial $p(x) = ax + bx^3$ that
satisfies $p(\sqrt{1-z^2})=1$ and $p(z) = 1$ by setting
$$\label{eq:init_poly}
a = \frac{z^2 (z + \sqrt{1-z^2}) - \sqrt{1-z^2}}{z \sqrt{1-z^2} (2 z^2 - 1)}
\qquad b = \frac{\sqrt{1-z^2}-z}{z \sqrt{1-z^2} (2 z^2 - 1)}$$ Because
$p(0)=0$ and $p$ has at most one local extremum on
$\mathbb{R}_{\geq 0}$, these conditions immediately guarantee that $p$
is concave-increasing on $[0, \sqrt{1-z^2}]$, so it must lie above the
line $x \mapsto x/\sqrt{1-z^2}$. Furthermore, $p$ is decreasing on
$[\sigma_1, 1]$, so it maps $\sigma_1 \in [z, 1]$ to $[p(1), 1]$. By
minimizing $p(1)$ over all valid $z$ (that is, over the interval
$z \in [1/\sqrt{2}, 1]$), one can further show that $p(1) > 1/\sqrt{2}$,
so $\sigma_1$ cannot be decreased very much by applying $p$. Thus, the
largest singular value of $p(\bm{M})$ is still at most $1$, while the
smaller singular values have increased by a potentially large factor of
$1/\sqrt{1-z^2}$. When there is a large outlying singular value, $z$ is
close to $1$ and this initialization scheme makes much more progress
than a standard iteration of `PolarExpress` would have.

In , we demonstrate the benefit of using the $p$ given by
[eq:init_poly] on a synthetic matrix
whose spectrum follows a power law decay. That is,
$\sigma_j(\bm{M}) = j^{-5}$, so this matrix has a large outlying
singular value $\sigma_1 \gg \sigma_2$. Applying
[eq:init_poly] costs almost as much as
performing an iteration of a degree-5 polynomial method, so for fair
comparison, we count it as an additional iteration in this plot. For
both Newton-Schulz and `Polar Express`, performing the extra
spectrum-aware initialization step described in this section leads to
significant speedups in convergence.



# Fast Polynomial Iteration for Rectangular Matrices

In this section, we describe a simple method for applying an iterative
polynomial method to a rectangular matrix. For matrices with a large
aspect ratio, this method yields significant computational savings. We
emphasize that this method is applicable to *any* computation of the
form $(p_T \circ \cdots \circ p_1)(\bm{X})$, where each $p_t$ is an odd
polynomial. Thus, it can be used to apply Newton-Schulz or Jordan’s
polynomials in addition to our own.

As a preliminary, we first describe the baseline approach. Let
$\bm X \in \mathbb{R}^{m \times n}$ with $m \geq n$, where
$\alpha := m / n \geq 1$ is called the aspect ratio. Any odd polynomial
$p$ of degree $d = 2q+1$ can be represented as $p(x) = xh(x^2)$, where
$h$ is a polynomial of degree $q$. Thus,
$p(\bm X) = \bm X h(\bm X^\top \bm X)$. Furthermore, $h$ can be written
in a factored form called Horner’s rule to reduce the number of
multiplications. For instance, if $h(y) = a + by + cy^2 + dy^3$,
Horner’s rule gives $h(y) = a + y\left(b + y\left(c + dy\right)\right)$.
For a matrix,
$h(\bm Y) = a\bm{I} + \bm{Y}\left(b\bm{I} + \bm{Y}\left(c\bm{I} + d\bm{Y}\right)\right)$.
Thus for $\bm Y \in \mathbb{R}^{n \times n}$, computing $h(\bm Y)$ costs
about $\left(\deg(h) - 1\right)\cdot n^3$ operations, and computing
$p(\bm X) = \bm X h(\bm X^\top \bm X)$ costs
$2mn^2 + \left(\frac{d-1}2 - 1\right)\cdot n^3 = \left(\frac{d-3}2 + 2\alpha\right)\cdot n^3$
operations. This process could be repeated for each iteration
$p_1, \ldots, p_T$. Notice that if we instead computed
$h(\bm X \bm X^\top) \bm X$, the result would be the same but the cost
would be higher.

A major drawback of this naive approach is that it has a strong
dependence on $\alpha$, since two rectangular matrix multiplications
must be performed in *each* of the $T$ iterations. When $m \gg n$, these
two multiplications dominate the cost. In , we introduce a simple trick
that dramatically reduces this cost, using just two rectangular matrix
multiplications to compute *all* $T$ iterations.



**input:** $\bm{X} \in \mathbb{R}^{m \times n}$ with $m > 1.5 n$, odd
polynomials $p_1(x) = x h_1(x^2), \ldots, p_T(x) = xh_T(x^2)$.  
**output:** The matrix $(p_T \circ \cdots \circ p_1)(\bm{X})$.



$\bm Y = \bm X^\top \bm X$ Let $\bm Q_0 = \bm I$
$\bm R_t = \bm Q_{t-1}^\top \bm Y \bm Q_{t-1}$
$\bm Q_t = \bm Q_{t-1} h_t(\bm R_t)$ **return** $\bm X \bm Q_T$





To see why this works, define $q_0(x) = x$, $$\begin{aligned}
q_t(x)
&= \frac{(p_t \circ \cdots \circ p_1)(x)}x
= \frac{p_t\left((p_{t-1} \circ \cdots \circ p_1)(x)\right)}x
= \frac{p_t\left(x q_{t-1}(x)\right)}x \\
&= \frac{x q_{t-1}(x)\cdot h_t\left((x q_{t-1}(x))^2\right)}x
= q_{t-1}(x)\cdot h_t\left(x^2 \cdot q_{t-1}(x)^2\right)
\end{aligned}$$ and $r_t(x) = x^2 \cdot q_{t-1}(x)^2$. It is clear by
induction that $\bm R_t = r_t(\bm X), \bm Q_t = q_t(\bm X)$, and
$\bm X \bm Q_T = (p_t \circ \cdots \circ p_1)(\bm X)$. As promised, this
algorithm uses no rectangular multiplications in the for-loop. If each
$p_t$ is degree $d$, then the total cost is
$\left(\frac{d+3}2 T + 2\alpha\right)\cdot n^3$. When
$\alpha > 1.5 \frac{T}{T-1}$, this is smaller than the naive method. We
can use this criterion to select either or the baseline method at
runtime.

can introduce numerical errors, especially when working in a low
precision format like `bfloat16`. We identify two sources of numerical
trouble and propose remedies for each. The first is due to the
ill-conditioning of $\bm{X}$. Let $\bm{X}= \bm{U}\bm{\Sigma}\bm{V}^\top$
be the SVD. For large $T$,
$(p_T \circ \cdots p_1)(\bm{X}) = \bm{X}\bm{Q}_T \approx \mathop{\mathrm{polar}}(\bm{X}) = \bm{U}\bm{V}^\top$.
Thus, $\bm{Q}_T \approx \bm{V}^\top \bm{\Sigma}^{-1} \bm{V}$. When
$\bm{X}$ has very small singular values and the floating point precision
is very low, instantiating $\bm{Q}_T$ may be unstable. To mitigate this
issue, we use a restarting strategy. Notice that the issue arises only
for large $T$, for which
$(p_T \circ \cdots \circ p_1)(\epsilon) \approx 1$. Limiting ourselves
to $T=3$ iterations improves the conditioning of $\bm{Q}_T$ because
$(p_T \circ \cdots \circ p_1)(\epsilon) \ll 1$. Thus, to compute $T>3$
iterations, we begin with $\bm{X}_0$ and apply with the first three
polynomials, producing $\bm{X}_3$. When then apply again with the next
three polynomials to $\bm{X}_3$, producing $\bm{X}_6$, and so on. As
$\bm{X}_t$ approaches convergence, its conditioning improves and we may
no longer need to restart at all. Note that restarting after every
iteration is exactly the same as the baseline method.

Second, while the matrix $\bm{Y}$ is positive definite in exact
arithmetic, numerical round-off can introduce spurious negative
eigenvalues that cause the method to diverge to infinity. To combat this
issue, we instead set $\bm{Y}= \bm{X}^\top \bm{X}+ 10^{-3}\bm{I}$ during
the first application of . (We also normalize by
$\|\bm{X}\|_\text{F}+ 10^{-3}$ instead of $\|\bm{X}\|_\text{F}$.) In
subsequent restarts of , we set $\bm{Y}= \bm{X}^\top \bm{X}$ as before.
This is akin to slightly increasing each of the singular values of
$\bm{X}$, but it does *not* change the polar factor of $\bm{X}$. Thus,
while the output will be slightly different in the early iterations, the
algorithm still converges to the correct answer.

shows that using can significantly improve runtime on the GPU when the
aspect ratio is large enough. As expected, using for many iterations
significantly reduces the dependence of the runtime on the aspect ratio.
Running six iterations of a degree-5 polynomial method when $\alpha = 4$
(as with the linear transformations in each MLP block of a transformer)
we obtain almost a 2x speedup, and when $\alpha = 32$, we obtain a 5x
speedup. If we restart every three iterations, the trend is the same but
the runtime savings are somewhat smaller.



## Application to `Muon`

If these problems can be mitigated, the speed afforded by suggests an
improvement in the way `Muon` is applied to transformers. In sum, the
idea is to replace one large matrix with a small aspect ratio by many
smaller matrices with large aspect ratios and apply to all of them in
parallel. Each multi-head attention layer contains four square weight
matrices $\bm W_Q, \bm W_K, \bm W_V$ and
$\bm W_O \in \mathbb{R}^{d \times d}$. The orthogonalization step of
`Muon` is either applied separately to these four matrices or else to
$[\bm W_Q \mid \bm W_K \mid \bm W_V]$ and $\bm W_O$, since typical
implementations of multi-head attention store the weights in this
concatenated form. However, we believe it is natural to consider each of
these four weight matrices to be a concatenation of many smaller linear
transformations, each corresponding to a single attention head. If $H$
is the number of heads, each of these smaller matrices has size
$d \times \frac{d}H$; that is, they have aspect ratio $\alpha = H$. The
gradient matrices of $[\bm W_Q \mid \bm W_K \mid \bm W_V]$ and $\bm W_O$
can be reshaped into 3-tensors in which each slice is one of these
smaller matrices. Since typical transformers like GPT-3 can have as many
as $96$ heads, this variation of `Muon` has the potential to reduce the
runtime.

We use this idea to train a GPT-Small model on FineWeb1B. We compare
four conditions:

1.  The baseline approach used in the rest of this paper

2.  Splitting up the gradient matrices of
    $[\bm W_Q \mid \bm W_K \mid \bm W_V]$ and $\bm W_O$ by head and
    applying Muon to each piece, as described above

3.  Using , restarted after three iterations

4.  Splitting by head *and* using

We used `Polar Express` with weight decay of $0.1$ for all conditions
and swept learning rates $0.003, 0.005, 0.01$. Otherwise, all
hyperparameters were the same as in .

Our results showed that these changes had a negligible effect in this
setting. They did not affect the optimization quality. Compared to the
baseline, splitting by heads actually reduced the final loss slightly
from 3.59 to 3.55; using increased the loss very slightly, from 3.59 to
3.60 when not splitting by head, and from 3.55 to 3.56 when we did
split. However, the runtimes of all 12 runs were nearly identical,
showing that at this scale, the FLOP savings of is not beneficial. The
embedding size of GPT-Small is just $768$. These techniques may be more
impactful when using a larger model. It may also have more impact
outside of deep learning, where `Polar Express` would be run for more
than the $5$ iterations used in our experiments. We leave exploration of
these settings to future work.

# Code for Constructing Polynomials of `Polar Express`

The following code gives a Python implementation of the offline stage of
. This code was used to construct the coefficients of the polynomials
given in [alg:polar-express], which in
turn were used in our `Muon` experiments (). It uses $\ell = 10^{-3}$
and $u=1$ by default. It incorporates and the finite precision
considerations described in .

``` python
from math import inf, sqrt
import numpy as np


def optimal_quintic(l, u):
    assert 0  u
    q = (3*l + 1) / 4
    r = (l + 3) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3*b + np.array([-1, 1]) * 
                        sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, cushion=0.02407327424182761):
    u = 1
    coefficients = []
    for _ in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion*u), u)
        # Due to cushioning, this may be centered around 1 with 
        # respect to 0.024*u, u. Recenter it around 1 with respect 
        # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
        pl = a*l + b*l**3 + c*l**5
        pu = a*u + b*u**3 + c*u**5
        rescalar = 2/(pl + pu)
        a *= rescalar; b *= rescalar; c *= rescalar
        # Optionally incorporate safety factor here:
        # a /= 1.01; b /= 1.01**3; c /= 1.01**5
        coefficients.append((a, b, c))
        l = a*l + b*l**3 + c*l**5
        u = 2 - l
    return coefficients


print(*optimal_composition(1e-3, 10), sep="\n")
```

[^1]: 

[^2]: New York University. `noah.amsel@nyu.edu`

[^3]: New York University and Flatiron Institute. `dup210@nyu.edu`,
    `dpersson@flatironinstitute.org`

[^4]: New York University. `cmusco@nyu.edu`

[^5]: Flatiron Institute. `rgower@flatironinstitute.org`

[^6]: 

[^7]: In , we describe two further algorithmic ideas that can be
    incorporated into `Polar Express`. They are not used in our `Muon`
    experiments but they may be beneficial in other settings, and we
    believe they merit further study.

[^8]: Our description of Newton’s method and other rational methods
    assumes square non-singular $\bm{M}$. Non-square problems can be
    reduced to the square case by an initial QR decomposition, but this
    is not an option for purely polynomial methods like ours.

[^9]: Jordan actually compares to $2x - \frac32 x^3 + \frac12 x^5$,
    whereas the true degree-5 Newton-Schulz polynomial is
    $(15x - 10x^3 + 3x^5)/8$. However, the difference in performance is
    negligible for the first few iterations.

[^10]: Our code is available at
    , in the `polar`
    branch.

[^11]: Let $\bm{Q}_0 \in \mathbb{R}^{n \times k}$ be a random matrix
    with orthonormal columns and define $\bm{Q}_{t+1},
    \bm{R}_{t+1} = \mathtt{qr}\left(\bm{M}^\top \bm{M}\bm Q_t\right)$,
    where $\mathtt{qr}$ is the QR decomposition. Subspace iteration
    outputs the singular values
    $\tilde \sigma_1, \ldots, \tilde \sigma_k$ of $\bm{M}\bm{Q}_T$,
    $\tilde \sigma_1, \ldots, \tilde \sigma_k$. By the Cauchy
    interlacing theorem, $\tilde \sigma_k \leq \sigma_k$.