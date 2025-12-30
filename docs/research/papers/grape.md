# Group Representational Position Encoding

## Authors
Yifan Zhang, Zixiang Chen, Yifeng Liu, Zhen Qin, Huizhuo Yuan, Kangping Xu, Yang Yuan, Quanquan Gu, Andrew Chi-Chih Yao

## Abstract
We present GRAPE (Group RepresentAtional Position Encoding), a unified framework for positional encoding based on group actions. GRAPE brings together two families of mechanisms: (i) multiplicative rotations (Multiplicative GRAPE) in $\mathrm{SO}(d)$ and (ii) additive logit biases (Additive GRAPE) arising from unipotent actions in the general linear group $\mathrm{GL}$. In Multiplicative GRAPE, a position $n \in \mathbb{Z}$ (or $t \in \mathbb{R}$) acts as $\mathbf{G}(n)=\exp(n\,ω\,\mathbf{L})$ with a rank-2 skew generator $\mathbf{L} \in \mathbb{R}^{d \times d}$, yielding a relative, compositional, norm-preserving map with a closed-form matrix exponential. RoPE is recovered exactly when the $d/2$ planes are the canonical coordinate pairs with log-uniform spectrum. Learned commuting subspaces and compact non-commuting mixtures strictly extend this geometry to capture cross-subspace feature coupling at $O(d)$ and $O(r d)$ cost per head, respectively. In Additive GRAPE, additive logits arise as rank-1 (or low-rank) unipotent actions, recovering ALiBi and the Forgetting Transformer (FoX) as exact special cases while preserving an exact relative law and streaming cacheability. Altogether, GRAPE supplies a principled design space for positional geometry in long-context models, subsuming RoPE and ALiBi as special cases. Project Page: https://github.com/model-architectures/GRAPE.



`Project Page`: 



# Introduction

Positional information is essential for sequence modeling with
Transformers , whose self‑attention is otherwise permutation‑invariant.
Early work injected absolute positional codes (sinusoidal or learned)
into token representations . Later, relative encodings depending on
offsets  and linear logit biases such as ALiBi  were introduced, the
latter offering strong length extrapolation with negligible overhead.



Rotary Position Embedding (RoPE)  realizes relative positions as
orthogonal planar rotations of queries and keys, preserving norms and
yielding exact origin invariance of attention scores. Despite its
appeal, RoPE fixes coordinate planes and typically a log‑uniform
spectrum, limiting cross‑subspace coupling and contextual warping of
phase. More broadly, absolute codes break translation equivariance;
table‑based relatives add window‑dependent overhead. A new formulation
is needed because current methods isolate the essential properties of
stability, monotonic distance penalty, and expressivity. These
observations motivate a unified formulation that (i) preserves RoPE’s
orthogonality and exact relativity when desired, (ii) *also* covers
additive/forgetting mechanisms such as ALiBi  and Forgetting Transformer
(FoX) , and (iii) admits learned and contextual generalizations with
clean streaming.

We therefore propose **G**roup **R**epresent**A**tional **P**osition
**E**ncoding (**GRAPE**), a group‑theoretic framework that unifies two
complementary families of positional mechanisms (see
Figure 1 for an overview). The
multiplicative family (Multiplicative GRAPE) models positions as
norm‑preserving rotations in $\mathop{\mathrm{SO}}(d)$ acting on
$(\qb,\kb)$; the additive family (Additive GRAPE/Path-Integral Additive
GRAPE) models positions as unipotent actions in the general linear group
$\mathrm{GL}$ that yield linear‑in‑offset logit biases (including
content‑gated and path‑integral forms). This perspective recovers RoPE
and ALiBi as exact special cases, proves that FoX is an exact instance
of Additive GRAPE, and supplies principled, streaming‑friendly
contextual extensions on both sides.

Concretely: *(a)* Multiplicative GRAPE (GRAPE-M) encodes
$n\in\mathbb{Z}$ (or $t\in\mathbb{R}$) as an element of
$\mathop{\mathrm{SO}}(d)$ via a rank‑2 skew generator; and *(b)*
Additive GRAPE (GRAPE-A) and Path‑Integral Additive GRAPE (GRAPE-AP)
lifts to the general linear group $\mathrm{GL}$ using homogeneous
coordinates to produce linear‑in‑offset logit biases (recovering ALiBi
and FoX).

For Multiplicative GRAPE, positions are mapped as $$\begin{aligned}
\Gb(n)=\exp\big(n\,\omega\, \Lb\big)\in\mathop{\mathrm{SO}}(d), \qquad \Lb=\ab \bbb^\top-\bbb\ab^\top\in\mathfrak{so}(d),
\end{aligned}$$ where $\ab,\bbb\in\mathbb{R}^d$ define a rank‑2 skew
generator $\Lb$ and $\omega>0$ is a frequency. The action is an
isometry, and $\Gb(n+m)=\Gb(n)\Gb(m)$ guarantees exact origin invariance
of attention logits. We derive a closed‑form Rodrigues‑type formula ,
enabling fast linear-time application with stable derivatives and no
explicit matrix materialization. RoPE is recovered when $d/2$ commuting
rank‑2 generators act on disjoint coordinate planes with prescribed
frequencies.

For Additive GRAPE, positions are mapped via the matrix exponential
$\Gb_{\mathrm{add}}(n)=\exp(n\omega
\Ab)=
\Ib+n\omega
\Ab$ in a lifted homogeneous space. Here, the generator
$\Ab \in \mathfrak{gl}(d+1)$ is a nilpotent matrix of rank one. While
this additive transformation is not an isometry, it preserves the exact
relative law, ensuring attention scores depend only on position offsets.
This formulation provides a rigorous group-theoretic foundation for
additive biases, recovering ALiBi and FoX as exact instances.

Our contributions are highlighted as follows:

1.  We propose **GRAPE** as a unified group‑theoretic view that subsumes
    *multiplicative* orthogonal rotations in $\mathop{\mathrm{SO}}(d)$
    and *additive* unipotent (all eigenvalues equal to $1$) mechanisms
    in general linear group $\mathrm{GL}$, recovering RoPE and ALiBi as
    exact special cases and proving FoX is an exact instance
    (Appendix 10).

2.  **Multiplicative GRAPE.** We derive a closed‑form rank‑2 matrix
    exponential with fast application and stable differentiation; we
    show RoPE is a special multiplicative GRAPE in a possibly learned
    orthogonal basis.

3.  **Additive GRAPE.** We show that linear‑in‑offset logit biases arise
    from rank‑1 (or low‑rank) unipotent actions in the general linear
    group $\mathrm{GL}$ with an exact relative law and streaming
    cacheability. This includes query‑ or key‑gated slopes, a commuting
    dictionary of additive components, and exact recoveries of ALiBi and
    FoX in closed form
    (Sections 4,
    4.2,
    Appendix 10). We also formalize
    path‑integral additive biases that remain causal and support
    efficient training.
    (Section 5).

# Multiplicative Group Representational Position Encoding

We propose the **Multiplicative GRAPE**, as a Lie‑group positional map
with a closed‑form rank‑2 matrix exponential, an exact relative law, and
a streaming/cache methodology. The core intuition is to encode position
as a norm‑preserving rotation in the special orthogonal group
$\mathop{\mathrm{SO}}(d)$ [^2]. A single skew‑symmetric generator
$\Lb\in\mathfrak{so}(d)$ produces the entire family of rotations via the
matrix exponential. We begin with notation and the rank‑2 generator.

## Preliminaries and Rank-2 Generator

The generator $\Lb$ is formally defined as an element of the
corresponding Lie algebra, $\mathfrak{so}(d)$. Let
$\mathfrak{so}(d)=\{\Lb\in\mathbb{R}^{d\times d}: \Lb^\top = -\Lb\}$
denote the Lie algebra of $\mathop{\mathrm{SO}}(d)$. The simplest
non-trivial generator defines a rotation within a single 2D plane. We
construct such a rank-2 generator from two vectors, $\ab$ and $\bbb$,
that span this plane of action. For $\ab, \bbb\in\mathbb{R}^d$, define
the rank-2 generator $\Lb \equiv \Lb(\ab, \bbb)$ as $$\begin{aligned}
\Lb(\ab, \bbb) = \ab \bbb^\top - \bbb \ab^\top,
\alpha=\|\ab\|^2,\ \beta=\|\bbb\|^2,\ \gamma = \ab^\top \bbb,
\Delta=\alpha\beta-\gamma^2\ge 0,\ s=\sqrt{\Delta} \label{eq:rank2generator}.
\end{aligned}$$

**Rank-2 structure.** Let $\mathcal{U}=\mathrm{span}\{\ab,\bbb\}$. The
rank‑2 generator $\Lb$ has a useful geometric property: applying it
twice projects onto the action plane $\mathcal{U}$ and scales. A direct
calculation shows $$\begin{aligned}
\Lb^2 = -\,s^2\,\mathbf{P}_{\mathcal{U}},
\end{aligned}$$ where $\mathbf{P}_{\mathcal{U}}$ is the orthogonal
projector to the space $\mathcal{U}$. Hence spectrum of $\Lb$ (the set
of its eigenvalues), denoted $\sigma(\Lb)$, is $\{\pm i s,0,\ldots,0\}$
and the minimal polynomial is $\lambda(\lambda^2+s^2)$. A detailed
derivation is given in
Appendix 16.

**Initialization.** Write
$\Ab\triangleq[\ab\ \bbb]\in\mathbb{R}^{d\times 2}$ and
$\Jb = \begin{psmallmatrix}0&-1\\[0.2ex]1&0\end{psmallmatrix}$ so that
$\Lb = \Ab \Jb \Ab^\top$. For any $\Mb\in \mathrm{SL}(2)$ (the
$2\times 2$ real matrices with determinant $1$, see
Table [tab:notation_summary]),
$\Mb \Jb \Mb^\top = \Jb$ and thus $\Ab \mapsto \Ab \Mb$ leaves $\Lb$
invariant; for general $\Mb\in \mathrm{GL}(2)$ the group of invertible
$2\times 2$ matrices), $\Lb$ scales by $\det(\Mb)$. Therefore the
oriented plane $\mathcal U=\mathrm{span}\{\ab, \bbb\}$ and the scalar
$s=\sqrt{\alpha\beta-\gamma^2}$ determine the action. We fix a gauge at
initialization by $\|\ab\|=\|\bbb\|=1$ and $\ab^\top \bbb = 0$
(absorbing scale into $\omega$).

**Canonical $90^\circ$ rotation operator.** Fix a block‑diagonal complex
structure $\mathcal{J}\in\mathfrak{so}(d)$ with
$\mathcal{J}^\top=-\mathcal{J}$ and $\mathcal{J}^2=-\mathbf{I}$ (for odd
$d$, act on the top‑left $2\lfloor d/2\rfloor$ coordinates and leave the
final coordinate unchanged). Concretely,
$\mathcal{J}=\bigoplus_{i=1}^{\lfloor d/2\rfloor}\begin{psmallmatrix}0&-1\\[0.2ex]1&0\end{psmallmatrix}$.
For any $\ab\in\mathbb{R}^d$, write $\ab_\perp := \mathcal{J} \ab$,
which equals “$\ab$ rotated by $90^\circ$” within the canonical $2$D
blocks and satisfies $\ab^\top \ab_\perp=0$ and $\|\ab_\perp\|=\|\ab\|$.

## Exact relative law

For a fixed $\Lb\in\mathfrak{so}(d)$, define
$\Gb(n)=\exp(n \Lb)\in\mathop{\mathrm{SO}}(d)$, which forms a
one‑parameter subgroup. The exact relative law property for positional
encoding implies: $$\begin{aligned}
\Gb(t{-}s) = \Gb(s)^\top \Gb(t),\qquad \Gb(n)^\top \Gb(n) = \mathbf{I}.
\end{aligned}$$ Here $\Gb(n)\in \mathop{\mathrm{SO}}(d)$, so the
transpose coincides with the group inverse, $\Gb(n)^\top=\Gb(n)^{-1}$;
the identity above is exactly the **relative-position law for a
one-parameter subgroup**. A concise summary of
$\mathop{\mathrm{SO}}(d)$, $\mathrm{GL}(d)$ and $\mathrm{SL}(d)$ is
collected in
Table [tab:notation_summary]. This
algebraic property enables relative positional encoding: interactions
depend only on offsets. $$\begin{aligned}
\Gb(n)=\exp(n\omega \Lb), \quad \Gb(n+m)=\Gb(n)\Gb(m), \quad \Gb(0)= \mathbf{I}, \quad \text{and} \quad \Gb(-n)=\Gb(n)^\top.
\end{aligned}$$ Crucially, this exact relative property relies solely on
the one-parameter subgroup structure ($G(n+m)=G(n)G(m)$), holding true
regardless of whether the generator implies commuting or coupled
non-commuting subspaces.

## Closed‑form fast matrix exponential

Based on the minimal polynomial mentioned in
Section 2.1, the exponential map
$\exp(\Lb)$ for a rank‑2 generator can be expressed as a quadratic in
$\Lb$. This yields a convenient closed‑form solution, often referred to
as a Rodrigues‑type formula : $$\begin{aligned}
\exp(\Lb) = \mathbf{I} + \frac{\sin s}{s}\, \Lb + \frac{1-\cos s}{s^2}\, \Lb^2.
\end{aligned}$$ Geometrically, the formula is best understood via
$\Lb^2$ as a projector onto $\mathcal{U}$. Since
$\Lb^2=-s^2 \mathbf{P}_{\mathcal{U}}$, the exponential can be written as
$$\exp(\Lb) = \mathbf{I}-(1-\cos s)\, \mathbf{P}_{\mathcal{U}} + \frac{\sin s}{s}\, \Lb,$$
which reveals its action explicitly: it is a rotation by angle $s$
within the plane $\mathcal{U}=\mathrm{span}\{\ab,\bbb\}$ and the
identity on the orthogonal complement $\mathcal{U}^\perp$. The vectors
$\ab$ and $\bbb$ thus define the plane of action for the positional
rotation.

**Cost of application.** For a single rank‑2 plane, computing
$\mathbf{y} = \Gb(n)\mathbf{x}$ requires two inner products
$\mathbf{u} = \langle \ab, \mathbf{x}\rangle$,
$\mathbf{v} = \langle \bbb, \mathbf{x}\rangle$, followed by
$\mathbf{y} = \mathbf{x} + f_1(n) (\ab v - \bbb u) + f_2(n)\left[\gamma(\ab v + \bbb u)-\beta \ab u-\alpha \bbb v\right]$,
where $(\alpha,\beta,\gamma)$ are plane scalars and $f_{1,2}$ are
trigonometric scalars (with series guards as $s\to0$). This is $O(d)$
flops with a small constant and no materialization of $\Gb(n)$;
derivative expressions are in
Appendix 16.

## The $\bbb=\mathcal{J}\ab$ constraint

We now consider an important special case by setting
$\bbb = \mathcal{J}\ab$. This constraint, which makes the plane vectors
$\ab$ and $\bbb$ orthogonal and equal in norm, significantly simplifies
the generator’s structure and reveals a direct connection to the
canonical RoPE formulation. With this constraint, the scalars simplify:
$\gamma = \ab^\top \bbb = \ab^\top\mathcal{J} \ab = 0$,
$\beta = \|\bbb\|^2 = \|\ab\|^2 = \alpha$, and hence
$s = \sqrt{\alpha\beta-\gamma^2} = \alpha$. Moreover, on the $2$D
subspace $\mathcal{U}=\mathrm{span}\{\ab, \mathcal{J}\ab\}$ one has
$$\begin{aligned}
\Lb(\ab, \mathcal{J} \ab) \ab =-(\mathcal{J}\ab) \alpha,\qquad \Lb(\ab,\mathcal{J}\ab)\,\mathcal{J}\ab = \alpha\,\ab,
\end{aligned}$$ so
$\Lb(\ab,\mathcal{J}\ab)|_{\mathcal{U}}=-\,\alpha\,\mathcal{J}|_{\mathcal{U}}$
and $\Lb(\ab, \mathcal{J} \ab)|_{\mathcal{U}^\perp}=0$. Therefore
$$\begin{aligned}
\exp\big(n\omega \Lb(\ab,\mathcal{J}\ab)\big)
= \mathbf{I} - \big(1-\cos(n\omega\alpha)\big)\mathbf{P}_{\mathcal{U}} - \sin(n\omega\alpha)\,\mathcal{J}\mathbf{P}_{\mathcal{U}},
\end{aligned}$$ This expression follows by substituting
$\Lb|_{\mathcal{U}}=-\alpha\,\mathcal{J}|_{\mathcal{U}}$ and
$\Lb^2=-\alpha^2\mathbf{P}_{\mathcal{U}}$ into the Rodrigues formula
$\exp(n\omega\Lb)=\mathbf{I}+\frac{\sin(n\omega s)}{s}\Lb+\frac{1-\cos(n\omega s)}{s^2}\Lb^2$
with $s=\alpha$; see
Appendix 16 for the algebraic steps. It is
a pure planar rotation by angle $n\omega\alpha$ on $\mathcal{U}$ and the
identity on $\mathcal{U}^\perp$.



If $\|\ab\|=1$, the rotation angle reduces to $n\omega$. Without
normalization, the effective frequency is
$\omega_{\mathrm{eff}}=\omega\|\ab\|^2$, so the scale of $a$ can be
absorbed into $\omega$.



## Application to relative encoding and equivariance

We now demonstrate how the **GRAPE-M** operator $\Gb(n)$ is applied in
practice. As established in
Section 2.2, the operator’s group structure
guarantees the exact relative law. We first transform the query and key
vectors, $\qb_i$ and $\kb_j$, into position‑aware representations,
$\tilde{\qb}_i$ and $\tilde{\kb}_j$: $$\begin{aligned}
\tilde{\qb}_i := \Gb(i)\qb_i, \qquad \tilde \kb_j := \Gb(j)\kb_j.
\end{aligned}$$ It follows from the exact relative law established in
Section 2.2 that the attention score
between these position-aware vectors simplifies to: $$\begin{aligned}
\tilde \qb_i^\top \tilde \kb_j = \qb_i^\top \Gb(i)^\top \Gb(j)\kb_j = \qb_i^\top \Gb(j-i)\kb_j.
\end{aligned}$$ Hence, the attention score depends solely on the
relative offset $j-i$, not on the absolute positions.

**Streaming and caching.** At inference, cache
$\kb_j^\star = \Gb(j) \kb_j$ once when token $j$ arrives. At step $t$,
form $\tilde \qb_t = \Gb(t) \qb_t$ and compute logits
$\tilde \qb_t^\top \kb_j^\star$. No cache rotation is needed when $t$
increments; complexity matches RoPE. A full integration into multi-head
attention (per-head formulation, logits, and streaming) is detailed in
Section 9.

# Multi‑Subspace Multiplicative GRAPE

A single rank‑2 generator acts on a 2D subspace, leaving the rest of the
$d$‑dimensional space untouched. To encode position across the entire
hidden dimension, we can combine multiple generators. This leads to the
Multi‑Subspace (MS) **Multiplicative GRAPE (GRAPE-M)** model, which
forms the basis for both RoPE and more expressive types. Detailed rank‑2
algebra appears in
Appendix 16.

## Multi‑Subspace GRAPE-M and RoPE as a Special Case

The simplest way to combine generators is to ensure they act on mutually
orthogonal subspaces, which guarantees they commute. Let $d$ be even.
For $i=1,\ldots,d/2$, we can define a set of rank-2 generators
$\{\Lb_i\}$, each acting on a distinct 2D plane. RoPE is the canonical
example of this construction. We further discussed non-commuting
multiplicative GRAPE in
Appendix 11.

Let the $2\times 2$ canonical skew matrix be
$\Jb=\begin{psmallmatrix}0&-1\\[0.2ex]1&0\end{psmallmatrix}$ and the
coordinate selector be
$\Ub_i=[\eb_{2i-1}\ \eb_{2i}]\in\mathbb{R}^{d\times 2}$. We set the
rank‑2 generators as
$\Lb_i = \Ub_i \Jb \Ub_i^\top = \Lb(\eb_{2i-1}, \eb_{2i})$ and assign
per‑plane frequencies $\theta_i>0$. The total generator is the commuting
sum:
$$\Lb_{\mathrm{RoPE}}=\sum_{i=1}^{d/2}\theta_i \Lb_i\qquad\text{with}\qquad [\Lb_i, \Lb_j]=0\ \text{ for }i\neq j.$$
Then $$\begin{aligned}
\Gb(n)
&=\exp\big(n \Lb_{\mathrm{RoPE}}\big)
=\prod_{i=1}^{d/2} \exp(n\theta_i \Lb_i)
=\mathop{\mathrm{blockdiag}}\big(\Rb_2(n\theta_1),\ldots, \Rb_{2}(n\theta_{d/2})\big), \label{eq:commuting_ms}
\end{aligned}$$ where $\Rb_2(\theta)$ denotes the standard $2\times 2$
rotation matrix introduced in
Table [tab:notation_summary], and
the last equality holds because each term $\exp(n\theta_i \Lb_i)$ is
identity except for a single $2{\times}2$ rotation block on its
diagonal. Eq. [eq:commuting_ms] is precisely the
RoPE mapping: a block‑diagonal product of planar rotations with
per‑subspace angles $n\theta_i$.

Equality holds when the planes $\{\Ub_i\}$ are the coordinate 2D blocks
and $\{\theta_i\}$ follow the canonical log-uniform spectrum.



 Choose $d/2$ mutually orthogonal
vectors $\{\ab_i\}$ and set $\bbb_i=\mathcal{J}\ab_i$ with per-plane
angles $\theta_i$. Then the commuting MS-GRAPE
$\Gb(n) = \prod_{i=1}^{d/2}\exp(n\theta_i \Lb(\ab_i, \mathcal{J}\ab_i))$
equals the standard RoPE map in a (possibly learned) orthogonal basis.
If the planes are the canonical coordinate pairs and $\{\theta_i\}$
follow the log-uniform spectrum, we recover the canonical RoPE exactly.



**Spectral parameterization.** Classical RoPE chooses $\theta_i$ on a
log‑uniform grid across $i$. In GRAPE, $\theta_i$ can be learned or
shared/tied across heads or layers. The MS‑GRAPE view also allows
replacing the coordinate selectors $\Ub_i$ by a learned orthogonal basis
$\Bb \in \mathop{\mathrm{SO}}(d)$ so that
$\Lb = \sum_i \theta_i \Bb \Ub_i \Jb \Ub_i^\top \Bb^\top$, preserving
commutativity while learning subspaces.

**Multimodal GRAPE.** Please refer to
Appendix 14 for 2D and 3D GRAPE for Vision
and Multimodal Position Encoding.

# Additive Group Representational Position Encoding

This section shows that additive positional mechanisms (absolute shifts
of features and additive logit biases, including ALiBi ) also admit a
group-theoretic formulation. The key is a homogeneous lift to an
augmented space and a one-parameter subgroup of the general linear group
$\mathrm{GL}$ that acts by unipotent (all eigenvalues equal to $1$)
transformations. This yields an exact relative law and streaming/cache
rules analogous to
Section 2.5.

## Homogeneous lift and a unipotent action

To produce additive biases from a multiplicative group action, we employ
the homogeneous lift. This is a standard method in linear algebra for
representing affine transformations (such as translations) as linear
transformations in a higher-dimensional space. Let
$\widehat{\xb}:=[\xb;1]\in\mathbb{R}^{d+1}$ denote a homogeneous
augmentation of $\xb\in\mathbb{R}^d$. We now work within the general
linear group $\mathrm{GL}(d+1)$ and its corresponding Lie algebra
$\mathfrak{gl}(d+1)$, which is the set of all $(d+1)\times(d+1)$ real
matrices. Fix a generator $$\label{eq:add_generator}
\Ab \;=\;
\begin{bmatrix}
\mathbf{0}_{d\times d} & \ub \\
\mathbf{0}_{1\times d} & 0
\end{bmatrix}
\in \mathfrak{gl}(d{+}1),
\qquad
\Ab^2=\mathbf{0},$$ where $\ub\in\mathbb{R}^d$. Its exponential is
unipotent: $$\begin{aligned}
\Gb_{\mathrm{add}}(n)
:= \exp(n\,\omega\, \Ab)
= \Ib_{d+1} + n\,\omega\, \Ab
=
\begin{bmatrix}
\Ib_d & n\,\omega\,\ub\\
\mathbf{0}^\top & 1
\end{bmatrix}\in \mathrm{GL}(d{+}1),
\end{aligned}$$ $$\begin{aligned}
\Gb_{\mathrm{add}}(n{+}m)=\Gb_{\mathrm{add}}(n)\Gb_{\mathrm{add}}(m).
\end{aligned}$$

**Application and exact relative law in $\mathrm{GL}$.** For
queries/keys augmented as $\widehat{\qb}_i=[\qb_i;1]$ and
$\widehat{\kb}_j=[\kb_j;1]$, define $$\begin{aligned}
\label{eq:add_transform}
\widetilde{\qb}_i := \Gb_{\mathrm{add}}(i)\,\widehat{\qb}_i,
\qquad
\widetilde{\kb}_j := \Gb_{\mathrm{add}}(j)^{-{\top}}\,\widehat{\kb}_j,
\end{aligned}$$ We use the shorthand
$\Gb_{\mathrm{add}}(j)^{-{\top}} := (\Gb_{\mathrm{add}}(j)^{-1})^\top$
to emphasize that we first take the group inverse in
$\mathrm{GL}(d{+}1)$ and then transpose it. and score with the standard
inner product on $\mathbb{R}^{d+1}$. The key is transformed using the
inverse transpose ($\Gb_{\mathrm{add}}(j)^{-{\top}}$). This is necessary
because for a general linear group $\mathrm{GL}$, the simple transpose
is no longer the inverse (unlike in $\mathop{\mathrm{SO}}(d)$), and the
inverse transpose is required to recover the exact relative law:
$\Gb_{\mathrm{add}}(i)^\top \Gb_{\mathrm{add}}(j)^{-{\top}} = \Gb_{\mathrm{add}}(j{-}i)^{-{\top}}$
for any one-parameter subgroup in $\mathrm{GL}$. This composition
results in the final form: $$\label{eq:add_relative_law}
\widetilde{\qb}_i^\top \widetilde{\kb}_j
= \widehat{\qb}_i^\top \Gb_{\mathrm{add}}(j{-}i)^{-{\top}} \widehat{\kb}_j,
\quad\text{depending only on } j{-}i.$$ Streaming matches
Section 2.5: cache
$\widehat{\kb}_j^\star=\Gb_{\mathrm{add}}(j)^{-{\top}}\widehat{\kb}_j$
once; at step $t$ form
$\widetilde{\qb}_t=\Gb_{\mathrm{add}}(t)\widehat{\qb}_t$ and compute
$\widetilde{\qb}_t^\top \widehat{\kb}_j^\star$.

**Closed form and content-gated additive term.** Since
$\Ab^\top=\begin{psmallmatrix}\mathbf{0}&\mathbf{0}\\ \ub^\top&0\end{psmallmatrix}$
and $(\Ab^\top)^2=\mathbf{0}$, $$\label{eq:add_closed_form}
\Gb_{\mathrm{add}}(m)^{-{\top}}
= \Ib_{d+1} - m\,\omega\, \Ab^\top
=
\begin{bmatrix}
\Ib_d & \mathbf{0}\\
-\,m\,\omega\,\ub^\top & 1
\end{bmatrix},
\qquad m=j{-}i,$$ whence $$\begin{aligned}
\label{eq:add_bias_keygated}
\widetilde{\qb}_i^\top \widetilde{\kb}_j
= \qb_i^\top \kb_j \;+\; 1 \;-\; (j{-}i)\,\omega\, \ub^\top \kb_j.
\end{aligned}$$ The constant “$+1$” is softmax‑shift invariant; the
final term is an additive, linear‑in‑offset bias whose slope is
key‑gated by $\ub^\top \kb_j$. A symmetric generator for the query,
$\Ab_{\mathrm{qry}}=\begin{psmallmatrix}\mathbf{0}&\mathbf{0}\\ \vb^\top&0\end{psmallmatrix}$
applied analogously produces a query‑gated slope
$(j{-}i)\,\omega\,\vb^\top \qb_i$. Using both the key-gated and
query-gated components yields a combined bias of the form
$(j{-}i)\,\omega\,( \vb^\top \qb_i - \ub^\top \kb_j)$, still obeying the
exact relative law
Eq. [eq:add_relative_law].

## Exact ALiBi as a Rank-1 unipotent in $\mathrm{GL}(d{+}2)$

ALiBi adds a head-specific scalar slope $\beta_h (j{-}i)$ to the logits
that is independent of content. This is captured exactly by augmenting
with two constant coordinates: $$\begin{aligned}
\widehat{\qb}_i=[\qb_i; \;1;\;0]\in\mathbb{R}^{d+2},\qquad
\widehat{\kb}_j=[\kb_j; \;0;\;1]\in\mathbb{R}^{d+2},
\end{aligned}$$ and choosing the rank-1 nilpotent generator
$$\begin{aligned}
\label{eq:alibi_generator}
\Ab_h^\top \;=\; \beta_h\, \eb_{d+1}\, \eb_{d+2}^\top
\quad\Longleftrightarrow\quad
\Ab_h \;=\; \beta_h\, \eb_{d+2}\, \eb_{d+1}^\top,
\qquad
(\Ab_h^\top)^2=\mathbf{0}.
\end{aligned}$$ Then
$\Gb_{\mathrm{add},h}(m)^{-{\top}}=\Ib - m\,\Ab_h^\top$ and
$$\begin{aligned}
\widehat{\qb}_i^\top \Gb_{\mathrm{add},h}(j{-}i)^{-{\top}} \widehat{\kb}_j
= \qb_i^\top \kb_j \;-\; (j{-}i)\,\beta_h,
\end{aligned}$$ i.e., the ALiBi term emerges as a unipotent
$\mathrm{GL}(d{+}2)$ action with exact relative composition.

**FoX as GRAPE-A.** Let $f_t\in(0,1]$ be per‑token forget scalars and
set $\omega_t:=\log f_t$. Using the rank‑1 generator of
Section 4.2, the resulting additive
bias is $b(t,j)=\sum_{\ell=j+1}^{t}\omega_\ell$, which coincides with
FoX’s forgetting bias $D_{ij}$. A full derivation and the unipotent path
product are given in
Appendix 10.

# Path Integral Additive GRAPE

Additive GRAPE (GRAPE-A) realizes exactly relative additive logits via a
one‑parameter unipotent action in the general linear group
$\mathrm{GL}$; the bias depends only on an offset $m=j{-}i$ (or a
contextual phase difference $\Phi_j{-}\Phi_i$ when using cumulative
phases). Here the “phase” $\Phi_t$ is a scalar path variable, typically
defined as a cumulative sum $\Phi_t=\sum_{\ell 0$. For each time $u$, let $\mathbf{p}_{u,h}\in\mathbb{R}^d$
be a positional embedding obtained from token-local features (a linear
projection followed by RMS normalization in our implementation). Let
$\mathcal{J}$ be the canonical block-diagonal $90^\circ$ operator
(Section 2.4), and define
$\Rb_\ell:=\exp(\ell\,\mathcal{J})$ (a fixed commuting rotation). For a
link function $g:\mathbb{R}\to(-\infty,0)$ that is monotone increasing
and $1$-Lipschitz[^3], define the *edge potential* $$\label{eq:pa-edge}
\psi_h(t,\ell)
:= \alpha_h\, g\left(\frac{1}{d}\,\big\langle \mathbf{p}_{t,h},\, \Rb_\ell\,\mathbf{p}_{\ell,h}\big\rangle\right)
\ \le\ 0,
\qquad \ell4.2). For each fixed endpoint
$t$, define endpoint-indexed unipotent factors
$$\Hb_h^{(t)}(\ell)\ :=\ \Ib + \psi_h(t,\ell)\, \Eb.$$ Since $\Eb^2=0$,
the path product along $(j,t]$ collapses additively:
$$\label{eq:pa-unipotent}
\prod_{\ell=j+1}^{t} \Hb_h^{(t)}(\ell)
\ =\ \Ib + \bigg(\sum_{\ell=j+1}^{t}\psi_h(t,\ell)\bigg)\,\Eb
\ =\ \Ib + b_h(t,j)\, \Eb.$$ Scoring in homogeneous coordinates as in
Section 4 with the paired
inverse-transpose removes multiplicative anisotropy and yields exactly
the additive term $b_h(t,j)$,
cf. Eq. [eq:add_relative_law]. The
*rowwise* semigroup law is preserved
(Eq. [eq:pa-unipotent]), while the
$t$-dependence of the factors intentionally relaxes the global
one-parameter group law.

 **Relation to
GRAPE-A.** GRAPE-AP strictly contains GRAPE-A as the special case in
which edge potentials do not depend on the endpoint:
$$\psi_h(t,\ell)\equiv \theta_h\,a_\ell
\ \Longrightarrow\
b_h(t,j)=\theta_h\sum_{\ell=j+1}^t a_\ell
=\theta_h\big(A_t-A_j\big),\ \ \ A_u:=\sum_{\ell4.2.

- **Phase-modulated Additive GRAPE.** If $a_\ell=\omega_\ell$ with
  $\omega_\ell=g(x_\ell)\ge 0$, then $b_h(t,j)=\theta_h(\Phi_t-\Phi_j)$
  with $\Phi_u=\sum_{\ell4. Outside these
endpoint-independent regimes, GRAPE-AP provides strictly more
expressive, path‑integral biases while preserving row-wise path
composition (Eq. [eq:pa-unipotent]).

**Computation and streaming.** For each head $h$ and decoding step $t$,
compute the row $\{\psi_h(t,\ell)\}_{\ell\le t}$ by a single similarity
sweep
$\ell\mapsto \langle \mathbf{p}_{t,h},\,\Rb_\ell \mathbf{p}_{\ell,h}\rangle$
(the rotated probes $\Rb_\ell \mathbf{p}_{\ell,h}$ can be cached on
arrival), apply the link $g$, and take a prefix sum to obtain
$j\mapsto b_h(t,j)$. This yields $O(t)$ per-step overhead with $O(1)$
recomputation per cached key; memory is $O(L)$ per head for the cached
probes (or $O(d)$ if the per-$\ell$ rotations are recomputed on the
fly).

**Spectral and stability.** Each factor
$\Hb_h^{(t)}(\ell)=\Ib+\psi_h(t,\ell)\Eb$ is unipotent with all
eigenvalues $1$ and at most two singular values deviating from $1$; the
full path product equals $\Ib+b_h(t,j)\Eb$
(Eq. [eq:pa-unipotent]). As in
Appendix 17.3, the paired
inverse-transpose used for scoring cancels multiplicative distortions
and delivers exactly the additive bias $b_h(t,j)$; operator norms remain
controlled linearly in $|b_h(t,j)|$.

A more extensive spectral analysis, including eigenvalue structure and
singular-value behavior across GRAPE variants, is provided in
Appendix 17. There, we also give an
explicit comparison to PaTH Attention , which is shown to be contractive
and near singular. These properties may impair PaTH’s effectiveness in
long-context modeling.

# Experiments

In this section, we evaluate the performance of GRAPE on the language
modeling task in comparison with baseline positional encoding
mechanisms, including RoPE , AliBi , as well as Forgetting Transformer
(FoX) .

## Implementation Details

Based on the nanoGPT codebase , our experiments are implemented based on
the Llama model . We only change the positional encoding mechanism and
keep the rest of the model architecture the same as Llama. We choose
FineWeb-Edu 100B dataset , which contains 100 billion training tokens
and 0.1 billion validation tokens, and we randomly choose 50B tokens for
training. Our models are with 36 layers and 10 heads, with a hidden size
of 1280 and head dimension of 128. We applied QK RMSNorm for training
stability . The context length is set to 4,096, and the batch size is
480. All the models are optimized by AdamW optimizer , with a maximum
learning rate of $2\times 10^{-4}$, $(\beta_1,\beta_2)=0.9,0.95$, and a
weight decay of 0.1. We use a cosine learning rate scheduler with 2,000
warm-up iterations, and the minimum learning rate is $1\times 10^{-5}$.
We also clip the gradient to 1.0 for stabler training. The frequency of
RoPE is set to 10,000. Moreover, for fair comparison, we do not use
FoX-Pro and disabled the KV-shift module within it.

## Result Analysis

The curves for training and validation loss of models with a variant
positional encoding mechanism are displayed in
Figures [fig:medium] and
[fig:large]. This analysis provides
specific insight into the source of the framework’s stability and
performance. It can be observed that GRAPE can keep a persistent edge
over other mechanisms, including RoPE and FoX. Moreover, the model with
RoPE suffers from training instability shown in Figure 3 (a), while the
model with GRAPE embedding steadily improves during the training
process.

























# Related Work

Positional information in Transformers mainly can be categorized into
these classes: (a) absolute encodings (sinusoidal or learned) ; (b)
relative encodings that depend on offsets ; and (c) linear logit biases
with strong length extrapolation , all shaping recency/extrapolation
behavior .

**Multiplicative position encoding.** RoPE realizes offsets as
block‑diagonal planar rotations of queries/keys, preserving norms and
exact origin invariance; it is widely deployed across LLMs and
modalities . Angle/spectrum designs improve long‑context fidelity (e.g.,
xPos) ; LRPE formalizes separable relative transforms for linear
attention models ; mechanistic work analyzes frequency usage . These
methods are also compatible with sparse/linear attentions and with
context‑scaling procedures . Beyond 1D language modeling, 2D RoPE and
variants adapt rotary encodings to 2D grids by applying rotations along
spatial axes, and have been shown to improve high‑resolution
extrapolation in Vision Transformers and related vision models .
Recently, LieRE  learns dense skew‑symmetric generators whose
exponentials produce high‑dimensional rotations for multi‑modal,
$n$‑dimensional inputs, while STRING  designs separable,
translation‑invariant RoPE‑style encodings that scale to 2D and 3D
coordinates in vision and robotics settings . **GRAPE-M** identifies
RoPE as commuting rank‑2 exponentials in $\mathop{\mathrm{SO}}(d)$ and
extends it to learned subspaces and compact non‑commuting mixtures in
closed form and a much faster way. Compared with LieRE, which
parameterizes a dense skew-symmetric generator and applies a numerical
matrix exponential (e.g., `torch.matrix_exp`) with $\mathcal{O}(d^3)$
time and $\mathcal{O}(d^2)$ parameters per head, Multiplicative GRAPE
decomposes the action into rank-2 subspaces and uses the closed-form
Rodrigues-type formulas from
Section 2.3, so we only need vector–vector
operations with $\mathcal{O}(d)$ cost per head (a detailed comparison
between LieRE and GRAPE is presented in
Appendix 13.)

**Additive position encoding and forgetting mechanisms.** Additive
schemes such as ALiBi and related kernelized/randomized forms are
captured exactly by GRAPE-A as unipotent actions in the general linear
group $\mathrm{GL}$ that preserve the same relative law and streaming
cacheability. Importantly, *forgetting mechanisms are additive*: the
Forgetting Transformer (FoX) implements a learnable per‑head exponential
decay in the attention logits and is a specific GRAPE-A / GRAPE-AP
instance imposing distance‑dependent attenuation . FoX’s data‑dependent
forget gates yield a path‑additive bias $\Db$ that we show is exactly
the endpoint‑independent GRAPE-AP case; see
Appendix 10 for a constructive
equivalence and its streaming implementation .

**Contextual position encoding.** Content‑adaptive position modulates
effective phase or distance via token features through gating/scaling
and algebraic parameterizations , and contextual counting (CoPE) . GRAPE
introduces phase‑modulated and dictionary‑based contextual variants that
replace a linear phase with cumulative token‑adaptive phases (single or
multi‑subspace) while retaining exact headwise relativity and streaming
caches. Finally, models can length‑generalize without explicit encodings
(“NoPE”) under suitable training , which corresponds to the trivial
generator $\Lb=0$ in our view.

# Conclusion

GRAPE provides a general framework for positional encoding based on
group actions, unifying *multiplicative* and *additive* mechanisms.
Multiplicative GRAPE offers a closed‑form, rank‑2 exponential that is
relative, compositional, and norm‑preserving; it recovers RoPE and
yields learned‑basis and non‑commuting extensions at controlled cost.
Additive GRAPE realizes ALiBi and FoX exactly via unipotent general
linear group $\mathrm{GL}$ lifts with the same streaming/cache policy.
The GRAPE framework integrates seamlessly with existing Transformer
models and offers a principled, extensible design space for future
architectures.



| **Symbol**         | **Definition**                                                                                                                 |
|:-------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| $\mathrm{GL}(d)$   | **General Linear Group**: The group of all $d \times d$ invertible matrices.                                                   |
| $\mathrm{SO}(d)$   | **Special Orthogonal Group**: The group of $d \times d$ orthogonal matrices                                                    |
|                    | with determinant 1 ($\mathbf{R}^\top \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R})=1$).                                          |
| $\mathrm{SL}(d)$   | **Special Linear Group**: The group of $d \times d$ matrices with determinant 1.                                               |
| $\mathfrak{gl}(d)$ | **general linear algebra**: The Lie algebra of $\mathrm{GL}(d)$, consisting of all $d \times d$ matrices.                      |
| $\mathfrak{so}(d)$ | **special orthogonal algebra**: The Lie algebra of $\mathrm{SO}(d)$, consisting of all $d \times d$                            |
|                    | skew-symmetric matrices ($\Lb^\top = -\Lb$).                                                                                   |
| $\exp(\cdot)$      | **Exponential Map**: A map from a Lie algebra (generator) to a Lie group (operator).                                           |
| $\Rb_2(\theta)$    | **2D Rotation Matrix**: The matrix $\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$.          |
| $\Gb(n)^\top$      | **Transpose (in $\mathrm{SO}(d)$)**: For $\Gb \in \mathrm{SO}(d)$, the transpose is the group inverse ($\Gb^\top = \Gb^{-1}$). |
| $\Gb(n)^{-{\top}}$ | **Inverse Transpose (in $\mathrm{GL}(d)$)**: The transpose of the matrix inverse, $(\Gb^{-1})^\top$.                           |
| Unipotent          | **Unipotent Transform**: A linear transformation whose eigenvalues are all 1.                                                  |
| $\mathbf{p}_{u,h}$ | **Positional Embedding/Representation**: A vector derived from token-local features,                                           |
|                    | obtained via a linear projection followed by RMS normalization.                                                                |



# Application in Multi‑Head Attention

Building upon the algebraic foundation for relative encoding established
in Section 2.5, this section details the
concrete integration of the rotational map $\Gb(n)$ into the full
Multi-Head Attention (MHA) architecture, covering the per-head
formulation, streaming policy, and implementation complexity.

**Per‑head formulation.** Let $H$ be the number of heads and $d$ the
per‑head width. For head $h\in[H]$, let
$(\qb_{t,h}, \kb_{t,h}, \vb_{t,h}) \in \mathbb{R}^d$ denote the
query/key/value at position $t$. A **GRAPE-M** position map is realized
as an orthogonal operator $\Gb_{h,t} \in \mathop{\mathrm{SO}}(d)$
applied to $(\qb_{t,h}, \kb_{t,h})$: $$\begin{aligned}
\tilde \qb_{t,h} = \Gb_{h,t}\, \qb_{t,h},\qquad
\tilde \kb_{t,h} = \Gb_{h,t}\, \kb_{t,h},\qquad
\tilde \vb_{t,h} = \vb_{t,h}.    
\end{aligned}$$ The headwise attention logits and outputs are then
$$\label{eq:mha_headwise_logits}
\ell_{t,j,h} = \frac{\tilde \qb_{t,h}^\top \tilde \kb_{j,h}}{\sqrt{d}} = \frac{\qb_{t,h}^\top \big(\Gb_{h,t}^\top \Gb_{h,j}\big) \kb_{j,h}}{\sqrt{d}},
\qquad
\yb_{t,h} = \sum_{j\le t}\mathrm{softmax}\big(\ell_{t,\cdot,h}\big)_j\,\tilde \vb_{j,h},$$
with the usual output projection applied after concatenation across
heads.

**Exact relative law.** If $\Gb_{h,t}$ arises from a one-parameter
subgroup $\Gb_{h}(n)=\exp(n\, \Lb_h)$ (commuting MS‑GRAPE-M, including
RoPE and learned commuting bases), then $$\begin{aligned}
\Gb_{h,t}^\top \Gb_{h,j} = \Gb_h(j{-}t)\qquad\Longrightarrow\qquad
\ell_{t,j,h} = \frac{\qb_{t,h}^\top \Gb_h(j{-}t)\, \kb_{j,h}}{\sqrt{d}},    
\end{aligned}$$ so logits depend only on the offset $j{-}t$ (exact
origin invariance).

**Streaming cache.** Applying the rotational map $\Gb(t)$ independently
to each query and key vector is the core property that enables an
efficient streaming cache policy. For any type where $\Gb_{t}$ is known
at token arrival (non-contextual and phase-modulated), cache
$\tilde \kb_{j,h} = \Gb_{h,j} \kb_{j,h}$ once and never rewrite it; at
step $t$, compute $\tilde \qb_{t,h} = \Gb_{h,t} q_{t,h}$ and use logits
$\ell_{t,j,h}=\tilde \qb_{t,h}^\top \tilde \kb_{j,h}/\sqrt{d}$.

# Forgetting Transformer as a Special Additive GRAPE

The Forgetting Transformer (FoX) introduces a scalar forget gate
$f_t\in(0,1]$ per head and timestep and adds the cumulative log‑gate as
an additive bias in the attention logits. Concretely, for a head $h$,
$$f_{t,h}=\sigma(\wb_{f,h}^\top \xb_t + b_{f,h}),\qquad
F_{ij,h}=\prod_{\ell=j+1}^{i} f_{\ell,h},\qquad
D_{ij,h}=\log F_{ij,h}=\sum_{\ell=j+1}^{i}\log f_{\ell,h},$$ and the
attention is
$$\Ob_h=\mathrm{softmax}\Big(\tfrac{1}{\sqrt d}\Qb\Kb^\top + \Db_h\Big)\Vb .
\tag{FoX}
\label{eq:fox}$$ We now show that
Eq. [eq:fox] is exactly realized by our GRAPE-A
framework using the endpoint‑independent path‑additive specialization of
Section 5.

**FoX as GRAPE-AP with endpoint‑independent edges.** In GRAPE-AP
(Section 5), a head‑wise additive logit
$b_h(t,j)$ arises as a causal path sum
$$b_h(t,j)\;=\;\sum_{\ell=j+1}^{t}\psi_h(t,\ell).$$ If the edge
potentials do not depend on the endpoint,
i.e. $\psi_h(t,\ell)\equiv a_{\ell,h}$, then $b_h(t,j)$ reduces to a
difference of per‑time potentials:
$$b_h(t,j)\;=\;\sum_{\ell=j+1}^{t} a_{\ell,h}
\;=\;U_{t,h}-U_{j,h},\qquad U_{u,h}:=\sum_{\ell4.2. For a fixed head $h$ and
endpoint $t$, define per‑link unipotent factors
$$\Hb^{(t)}_h(\ell)\;=\;\Ib + \psi_h(t,\ell)\,\Eb,\qquad \psi_h(t,\ell)=\log f_{\ell,h}.$$
Since $\Eb^2=\mathbf{0}$, the path product collapses:
$$\prod_{\ell=j+1}^{t}\Hb^{(t)}_h(\ell)\;=\;\Ib + \Big(\sum_{\ell=j+1}^{t}\log f_{\ell,h}\Big) \Eb
\;=\;\Ib + D_{ij,h}\, \Eb.$$ Scoring in homogeneous coordinates as in
Section 4 with the paired
inverse‑transpose, $$\widetilde{\qb}_{t,h}^\top\,\widetilde{\kb}_{j,h}
\;=\;\widehat{\qb}_{t,h}^\top\Big(\Ib + D_{ij,h}\,E\Big)^{-{\top}}\widehat{\kb}_{j,h}
\;=\;\qb_{t,h}^\top \kb_{j,h}\;+\;D_{ij,h},$$ recovers
Eq. [eq:fox] exactly (up to the standard
$1/\sqrt d$ factor we include throughout). Hence *FoX is an exact
GRAPE-A / GRAPE-AP instance* realized by a rank‑1 unipotent path with
endpoint‑independent edges.

**Streaming and complexity.** Compute prefix sums
$U_{t,h}=\sum_{\ell4–Section 5. The headwise gates $f_{t,h}$
add $O(1)$ parameters and negligible computation.

**Special cases and composition.** If $f_{t,h}\equiv e^{-\beta_h}$
(constant per head), then $D_{ij,h}=-\beta_h(i{-}j)$ and FoX reduces to
exact ALiBi
(Section 4.2). More generally, FoX
composes additively with the multiplicative (orthogonal) GRAPE acting on
$(\qb,\kb)$ as in Eq. [eq:pa-logit], preserving
norm‑preservation of the rotational part while adding bounded,
non‑positive, content‑adaptive path biases.

# Non-Commuting Multiplicative GRAPE

Consider the thin compression $\Lb = \Eb \Lb_r \Eb^\top$ with
$\Eb\in\mathbb{R}^{d\times r}$ orthonormal and
$\Lb_r\in\mathfrak{so}(r)$. Then
$$\sigma(\Lb) = \sigma(\Lb_r)\cup\{0\}^{d-r},\qquad
\sigma\big(\exp(n \Lb)\big)=\sigma\big(\exp(n \Lb_r)\big)\cup\{1\}^{d-r}.$$
If $\Lb_r = \Tb(\bigoplus_{t=1}^{r/2}\theta_t \Jb) \Tb^\top$ is the
real-Schur form, then the nontrivial eigenvalues are
$\{\pm i\theta_t\}_{t=1}^{r/2}$ and $e^{\pm i n\theta_t}$ for the
exponential. Thus, the expressive power of non-contextual non-commuting
MS-GRAPE is fully captured by the $r/2$ mode angles $\{\theta_t\}$; the
ambient lifting via $\Eb$ preserves the spectrum.

# Composition of Additive GRAPE and Multiplicative GRAPE

For the unipotent forms of Additive GRAPE, applying
$\Gb_{\mathrm{add}}(m)^{-{\top}}$ requires one inner product and one
scalar-vector multiplication per active component. Thus, the per-head
overhead is $O(d)$ and typically negligible relative to attention
matmuls. Multiplicative GRAPE
(Section 3) and Additive GRAPE
(Section 4) compose naturally, either
additively at the logit level $$\begin{aligned}
\ell_{t,j,h}
= \tfrac{1}{\sqrt{d}}\,\qb_{t,h}^\top \Gb_h(j{-}t) \kb_{j,h}
\;+\; \Big[\widehat{\qb}_{t,h}^\top \Gb_{\mathrm{add},h}(j{-}t)^{-{\top}} \widehat{\kb}_{j,h} - \qb_{t,h}^\top \kb_{j,h}\Big],
\end{aligned}$$ or as a single block‑upper‑triangular $\mathrm{GL}$
action in homogeneous coordinates. Concretely, define the joint lift
$$\widehat{\qb}=[\qb;1],\quad
\widehat{\kb}=[\kb;1],\qquad
\widehat{\Gb}(m)
\;=\;
\begin{bmatrix}
\exp(m\,\Lb) & m\,\omega\,\ub\\
\mathbf{0}^\top & 1
\end{bmatrix}
\in \mathrm{GL}(d{+}1),$$ which combines the orthogonal rotation
$\exp(m\Lb)$ on features with a unipotent translation along the
homogeneous axis. Scoring with the paired inverse‑transpose as in
Eq. [eq:add_transform] yields
$$\widehat{\qb}^\top\,\widehat{\Gb}(m)^{-{\top}}\widehat{\kb}
+\;=\; \qb^\top \exp(m\Lb)\kb \;-\; m\,\omega\,\ub^\top \kb \;+\; \text{const},$$
exactly reproducing the sum of multiplicative (rotary) and additive
(bias) components up to a softmax‑invariant constant. In both
formulations, exact relativity and streaming caches are retained.

# Comparison with LieRE

Lie Rotational Position Encodings (LieRE)  encode positional information
by learning a skew-symmetric generator in $\mathop{\mathrm{SO}}(d)$. The
method then applies the matrix exponential of this generator to get a
rotational position map. For each attention head, the method learns one
skew matrix. Its exponential gives a dense orthogonal operator on
queries and keys. Positions then match elements of a one-parameter
subgroup on the rotation manifold. This picture is a compact Lie
theoretic version of RoPE style encodings. Different heads can learn
distinct rotational geometries and the map keeps the norm and an exact
relative position law.

Formally, for head $h$ the generator is $G_h\in\mathfrak{so}(d)$. The
positional map is $x\mapsto \exp(n\omega_h G_h)x$. A direct
implementation has cost $T_{\mathrm{LieRE}}(d)=\Theta(d^3)$ per head for
the matrix exponential and needs $\Theta(d^2)$ parameters and the same
order of memory.

Multiplicative GRAPE and LieRE both use rotations in
$\mathop{\mathrm{SO}}(d)$ that come from skew-symmetric generators.
LieRE gives each head a dense or block skew matrix. It forms the
positional operator with the full matrix exponential $\exp(G)$. This
creates very rich rotations but needs $\mathcal{O}(d^3)$ time for the
exponential and $\mathcal{O}(d^2)$ parameters and memory per head.
GRAPE-M restricts the generator to a sum of rank 2 planes and uses a
closed form Rodrigues-type formula for the exponential
(Section 2). For one token, the positional
mapping then reduces to a few inner products and vector updates. So the
cost is $\mathcal{O}(d)$ time and $\mathcal{O}(d)$ memory per head.

This choice of parametrization has two main effects in practice. First,
the GRAPE-M scale cleanly translates to contextual versions where
frequencies or phases depend on the token content. The closed-form
expression can be computed quickly for each token, and there is no large
matrix exponential. In the LieRE setup, one needs a new dense matrix
exponential for each content-dependent generator. This step is much more
costly and makes such contextual use harder to deploy in real models.
Second, GRAPE gives a single group-theoretic picture for multiplicative
and additive mechanisms. The multiplicative part lives in
$\mathop{\mathrm{SO}}(d)$ and additive or forgetting style terms (ALiBi,
FoX, GRAPE-A, GRAPE-AP) come from unipotent actions in $\mathrm{GL}$
with the same relative law and the same streaming cacheability
(Sections 4-5). LieRE only targets rotational
encodings and does not model additive logit biases or forgetting terms.

# 2D and 3D GRAPE for Vision and Multimodal Position Encoding

Extending GRAPE beyond one-dimensional token positions is easy. The
construction only needs a chosen group action on coordinates.

For images with integer pixel coordinates $(u,v)\in\mathbb{Z}^2$ we pick
two generators $\Lb^{(x)}$ and $\Lb^{(y)}$. A token at $(u,v)$ then gets
the encoding $$\Gb_{\mathrm{2D}}(u,v)
  = \exp\!\big(u\,\omega_x \Lb^{(x)}\big)\,
    \exp\!\big(v\,\omega_y \Lb^{(y)}\big)
  \in \mathop{\mathrm{SO}}(d).$$ The two generators act on 2D planes
that can be disjoint in the base design. In that case, the map reduces
to a RoPE-style separable encoding. A learned choice of planes inside
$\mathbb{R}^d$ gives the GRAPE-M variant again.

For 3D coordinates $(u,v,w)$ that mark video space time tokens or point
clouds, we follow the same pattern. We introduce three commuting
generators and define $$\Gb_{\mathrm{3D}}(u,v,w)
  = \exp\!\big(u\,\omega_x \Lb^{(x)}\big)\,
    \exp\!\big(v\,\omega_y \Lb^{(y)}\big)\,
    \exp\!\big(w\,\omega_z \Lb^{(z)}\big).$$ In the non-commuting case,
we use the thin Schur mode compression from
Appendix 11. The closed-form rank 2
matrix exponential from the main text still applies. The per token cost
stays $\mathcal{O}(d)$ even for higher-dimensional coordinate spaces.

On the additive side, GRAPE-A and GRAPE-AP handle 2D or 3D structures
through the scalar offset $m$. The value $m$ can be any function of
coordinate differences. For an image, we can take
$$m = \alpha_x(u_t - u_j) + \alpha_y(v_t - v_j),$$ and this keeps the
same algebraic template. For 3D settings, we can set
$$m = \|{\bf r}_t - {\bf r}_j\|$$ with ${\bf r}_t$ and ${\bf r}_j$ in
$\mathbb{R}^3$. The update matrix then stays unipotent, and the exact
relative composition law still holds. This gives a clear way to impose
axis-aligned or radial recency bias in vision and multimodal models.

# Algorithmic Details and Pseudo Code

This appendix contains the detailed pseudocode.





$\Qb, \Kb\in\mathbb{R}^{B\times L\times H\times d}$, orthogonal
$\Eb \in \mathbb{R}^{d\times d}$, frequencies
$\{\omega_{h,j}\}_{j=1}^{d/2}$, positions $n\in\mathbb{Z}^L$
$\Qb'[:, :, h, :]\gets \Qb[:, :, h, :]\, \Eb$;$\Kb'[:, :, h, :]\gets \Kb[:, :, h, :]\, \Eb$
$\theta\gets n_\ell \,\omega_{h,j}$; apply $2\times 2$ rotation
$\Gb_2(\theta)$ to coords $(2j{-}1,2j)$ of $\Qb'[:,\ell,h,:]$ and
$\Kb'[:,\ell,h,:]$
$\tilde \Qb[:, :, h, :]\gets \Qb'\, \Eb^\top$;$\tilde \Kb[:, :, h, :]\gets \Kb'\, \Eb^\top$
$(\tilde \Qb, \tilde \Kb)$









$\Qb, \Kb\in\mathbb{R}^{B\times L\times H\times d}$; planes
$\{(\ab_{h,j}, \bbb_{h,j},\omega_{h,j})\}_{j=1}^{m}$; positions $n$
Build $\Ub_h=\mathrm{span}\{\ab_{h,j}, \bbb_{h,j}\}$; orthonormalize
$\bbb_h\in\mathbb{R}^{d\times r_h}$
$\Lb_{\Ub,h}\gets \bbb_h^\top \Big(\sum_{j=1}^{m}\omega_{h,j} \Lb(\ab_{h,j}, \bbb_{h,j})\Big) \bbb_h\in\mathfrak{so}(r_h)$
Orthogonally Schur-decompose:
$\Lb_{\Ub,h} = \Tb_h\left(\bigoplus_{t=1}^{r_h/2}\theta_{h,t} \Jb\right) \Tb_h^\top$
$\Eb_h\gets \bbb_h \Tb_h \in \mathbb{R}^{d\times r_h}$; precompute
$(c_{h,t}, s_{h,t})=(\cos\theta_{h,t},\sin\theta_{h,t})$
$y_\Qb \gets \Eb_h^\top \Qb[:,\ell,h,:]$;$y_\Kb \gets \Eb_h^\top \Kb[:,\ell,h,:]$
$(\Cb_{h,t}, \Sbb_{h,t}) \gets \textsc{PhaseTo}(n_\ell;\, c_{h,t}, s_{h,t})$
Apply
$\begin{psmallmatrix}\Cb_{h,t}&- \Sbb_{h,t}\\ \Sbb_{h,t}& \Cb_{h,t}\end{psmallmatrix}$
to coordinates $(2t{-}1,2t)$ of $y_Q, y_K$
$\tilde \Qb[:,\ell,h,:]\gets \Qb[:,\ell,h,:] + \Eb_h(y_Q - \Eb_h^\top \Qb[:,\ell,h,:])$
$\tilde \Kb[:,\ell,h,:]\gets \Kb[:,\ell,h,:] + \Eb_h(y_K - \Eb_h^\top \Kb[:,\ell,h,:])$
$(\tilde \Qb,\tilde \Kb)$









$\Qb,\Kb\in\mathbb{R}^{B\times L\times H\times d}$; per-head additive
generators $\{\Ab_h\}$ with $\Ab_h^2=\mathbf{0}$; positions
$n\in\mathbb{Z}^L$ Augment:
$\widehat{\Qb}\gets[\Qb;\mathbf{1};\mathbf{0}]$,
$\widehat{\Kb}\gets[\Kb;\mathbf{0};\mathbf{1}]$ as needed
(Section 4.2)
$\widehat{\Kb}^\star[:,j,h,:]\gets \big(\Ib - n_j\,\Ab_h^\top\big)\,\widehat{\Kb}[:,j,h,:]$
$\widetilde{\Qb}[:,t,h,:]\gets \big(\Ib + n_t\,\Ab_h\big)\,\widehat{\Qb}[:,t,h,:]$
Compute additive logits:
$\lambda_{t,j,h}\gets \widetilde{\Qb}[:,t,h,:]^\top \widehat{\Kb}^\star[:,j,h,:]$
$\{\lambda_{t,j,h}\}$ (to be added to orthogonal GRAPE/RoPE logits)





# Differentiation and Fast Application of Rank-2 Matrix Exponential

**Differentiation and stability.** Let $f_1(z)=\frac{\sin z}{z}$ and
$f_2(z)=\frac{1-\cos z}{z^2}$ with $z=n\omega s$. Then
$$\exp(n\omega \Lb) = \Ib + f_1(z) \Lb + f_2(z) \Lb^2.$$ For any scalar
parameter $\theta\in\{\omega\}\cup\{\text{entries of }a,b\}$,
$$\begin{aligned}
\partial_\theta \exp(n\omega \Lb)
&= f_1(z)\,\partial_\theta \Lb + f_2(z)\,(\Lb\,\partial_\theta \Lb + \partial_\theta \Lb\, \Lb)
+ \partial_\theta z\,\big(f_1'(z) \Lb + f_2'(z) \Lb^2\big),\\
\partial_\theta z&= n\omega\,\partial_\theta s + n s\,\partial_\theta \omega,\qquad
\partial_\theta s=\tfrac{1}{2}s^{-1}\partial_\theta(\alpha\beta-\gamma^2).
\end{aligned}$$ Use series for $|z|2.

## Rank-2 Plane: Exact Spectrum and Geometric Interpretation



 For
$\Lb = \Lb(\ab, \bbb)$, the eigenvalues are
$\{\pm i s\}\cup\{0\}^{d-2}$, and there exists
$\Bb\in\mathop{\mathrm{SO}}(d)$ such that
$$\Bb^\top \Lb \Bb = \begin{bmatrix} s \Jb & \mathbf{0}\\ \mathbf{0} & \mathbf{0}_{d-2}\end{bmatrix},\qquad \Jb = \begin{psmallmatrix}0&-1\\[0.2ex]1&0\end{psmallmatrix}.$$
Moreover, $s=\|\ab\|\|\bbb\|\sin\phi$, where $\phi\in[0,\pi]$ is the
angle between $a$ and $b$.





*Proof.* From
Section 2, $\Lb^2=-s^2 \Pb_{\mathcal U}$
with $\mathcal U=\mathrm{span}\{\ab, \bbb\}$, whence the minimal
polynomial is $\lambda(\lambda^2+s^2)$ and $\sigma(\Ub)=\{\pm i s,0\}$.
Choosing an orthonormal basis aligned with
$\mathcal U\oplus\mathcal U^\perp$ yields the claimed form. Finally,
$\Delta=\alpha\beta-\gamma^2=\|\ab\|^2\|\bbb\|^2(1-\cos^2\phi)=(\|\ab\|\|\bbb\|\sin\phi)^2$. ◻





 The
per-step rotation angle of $\exp(\eta \Kb)$ on $\mathcal U$ equals
$\theta=\eta s$ and satisfies $0\le \theta\le \eta\|\ab\|\|\bbb\|$, with
equality when $\ab\perp \bbb$. If $\bbb=\mathcal{J} \ab$
(Section 2.4) and $\|\ab\|=1$, then $s=1$ and
$\theta=\eta$.



**Exponential spectrum.** For any $n\in\mathbb{Z}$,
$$\sigma\big(\exp(n \Lb)\big)=\{e^{\pm i n s}\}\cup\{1\}^{d-2}.$$ Hence
$\rho(\exp(n \Lb))=1$, the map is unitary (orthogonal), and all Lyapunov
exponents are zero. Periodicity holds with fundamental period $T=2\pi/s$
when $s/\pi\in\mathbb{Q}$; otherwise, the trajectory is quasi-periodic
on the unit circle.

## Multi-subspace GRAPE-M and RoPE

Let $\Lb = \sum_{j=1}^{m}\theta_j \Lb_j$ with mutually orthogonal planes
(hence $[\Lb_i, \Lb_j]=0$ for $i\neq j$) and
$\Lb_j = \Ub_j \Jb \Ub_j^\top$. Then
$$\Bb^\top \Lb \Bb = \bigoplus_{j=1}^{m}\theta_j \Jb \oplus \mathbf{0}_{d-2m},\qquad
\sigma(\Lb)=\{\pm i\theta_j\}_{j=1}^{m}\cup \{0\}^{d-2m},$$ for some
$\Qb\in\mathop{\mathrm{SO}}(d)$. Consequently,
$$\sigma\big(\exp(n \Lb)\big)=\{e^{\pm i n \theta_j}\}_{j=1}^{m}\cup\{1\}^{d-2m}.$$
This recovers RoPE when the planes are the coordinate pairs and
$\{\theta_j\}$ follow the canonical log-uniform spectrum
(Proposition [prop:rope_as_grape]).

## Additive GRAPE

We now analyze the spectral properties of the additive lifts in
$\mathrm{GL}$ introduced in
Sections 4
and 5. The key structural fact is
unipotency: all per-step factors are identity plus a rank-1 (or
few-rank) nilpotent update of index $2$.

**Setup.** Let $\Ab\in\mathfrak{gl}(d{+}1)$ (or $\mathfrak{gl}(d{+}2)$
for ALiBi) satisfy $\Ab^2=\mathbf{0}$ as
in [eq:add_generator]
and [eq:alibi_generator]. For a
scalar path parameter $s\in\mathbb{R}$, define the unipotent factor
$$\Hb(s)\ :=\ \exp(s\Ab)\ =\ \Ib + s\,\Ab,
\qquad \Hb(s)^{-1}=\Ib - s\,\Ab,\qquad \det \Hb(s)=1 .$$ For Additive
GRAPE (GRAPE-A) with offset $m=j{-}i$, $s=m\,\omega$; for GRAPE-PA,
$s=s_h(t,j):=\sum_{\ell=j+1}^{t}\psi_h(t,\ell)$ from
Eq. [eq:grape_pa_bias].



 Let $\Ab\in\mathfrak{gl}(D)$
satisfy $\Ab^2=\mathbf{0}$ and $\Ab\neq \mathbf{0}$. Then for every
$s\neq 0$, $$\sigma\big(\Hb(s)\big)=\{1\}^{D},\qquad
(\Hb(s)-\Ib)^2=\mathbf{0},\qquad
\det\Hb(s)=1,\qquad \rho(\Hb(s))=1.$$ Hence, the minimal polynomial of
$\Hb(s)$ is $(\lambda-1)^2$, and the Jordan form consists of size-$2$
Jordan blocks for the $1$-eigenspace, with the number of nontrivial
blocks equal to $\operatorname{rank}(\Ab)$.





*Proof.* Since $\Ab^2=\mathbf{0}$, $\exp(s\Ab)=\Ib+s\Ab$ and
$(\Hb(s)-\Ib)^2=s^2\Ab^2=\mathbf{0}$. The characteristic polynomial is
$(\lambda-1)^D$ for $\Hb(s)$, so all eigenvalues equal $1$. The
determinant equals the product of eigenvalues, hence $1$; the spectral
radius is therefore $1$. ◻



**Dictionary closure.** If $\{\Ab_r\}_{r=1}^R$ satisfy
$\Ab_r^2=\mathbf{0}$ and $\Ab_r\Ab_s=\mathbf{0}$ for all $r,s$, then
$$\Big(\sum_r \theta_r \Ab_r\Big)^2=\sum_r \theta_r^2 \Ab_r^2 + \sum_{r\ne s}\theta_r\theta_s \Ab_r\Ab_s=\mathbf{0},$$
so the combined generator is also index-$2$ nilpotent and yields the
same unipotent spectrum.

**Singular values.** Although $\Hb(s)$ is not orthogonal, its deviation
from $\Ib$ is rank-limited and exactly analyzable. We first give a
sharp, explicit formula for the canonical rank-1 case (ALiBi block),
then a general bound.



 Let
$E:=\eb_{p}\,\eb_{q}^\top$ with $p\neq q$ and define
$H(s):=\Ib+sE\in\mathbb{R}^{D\times D}$. Then $D-2$ singular values
equal $1$, and the remaining two are $$\label{eq:sv_pair_closed_form}
\sigma_\pm(H(s)) \;=\; \sqrt{\,1+\tfrac{s^2}{2} \;\pm\; |s|\sqrt{1+\tfrac{s^2}{4}}\,}\,,
\qquad \sigma_+(H(s))\,\sigma_-(H(s))=1 .$$ In particular,
$\kappa_2(H(s))=\sigma_+(H(s))/\sigma_-(H(s)) = 1+2|s|+O(s^2)$ as
$s\to 0$.





*Proof.* The action of $H(s)^\top H(s)$ is identity on
$\mathrm{span}\{\eb_p,\eb_q\}^\perp$. In the basis $\{\eb_q,\eb_p\}$ it
equals $\begin{psmallmatrix}1+s^2 & s\\ s & 1\end{psmallmatrix}$, whose
eigenvalues are $1+\tfrac{s^2}{2}\pm |s|\sqrt{1+\tfrac{s^2}{4}}$. Taking
square roots
yields [eq:sv_pair_closed_form].
The product equals $\sqrt{\det(H^\top H)}=|\det H|=1$. ◻





 For the exact ALiBi
generator in
Eq. [eq:alibi_generator],
$E=\eb_{d+2}\eb_{d+1}^\top$ and $s=m\,\beta_h$, so the only nontrivial
singular values of $\Gb_{\mathrm{add},h}(m)=\Ib+s\Eb$ are given by
Eq. [eq:sv_pair_closed_form].
For the single-vector additive lift
Eq. [eq:add_generator] with
$\Ab=\begin{psmallmatrix}\mathbf{0}&\ub\\ \mathbf{0}^\top & 0\end{psmallmatrix}$
and $\|\ub\|=1$, the same formula holds with $\Eb$ replaced by an
orthogonally similar rank-1 update and $s=m\,\omega$.





 For any $\Ab$
with $\Ab^2=\mathbf{0}$ and any $s\in\mathbb{R}$,
$$1-|s|\,\|\Ab\|_2 \;\le\; \sigma_{\min}(\Ib+s\Ab)
\;\le\; \sigma_{\max}(\Ib+s\Ab) \;\le\; 1+|s|\,\|\Ab\|_2 .$$ When
$\operatorname{rank}(\Ab)=1$ and $\|\Ab\|_2=1$, these bounds are tight
and coincide with
Lemma [lem:sv_pair_exact] at first
order in $|s|$.





*Proof.* Use the triangle inequality
$\|(\Ib+s\Ab)\xb\|_2\le \|\xb\|_2+|s|\,\|\Ab\|_2\|\xb\|_2$ and its
reverse form applied to $(\Ib+s\Ab)^{-1}=\Ib-s\Ab$; see also Weyl
inequalities for singular values under rank-1 perturbations. ◻



**Cancellation in the relative logit.** While $\Hb(s)$ can be
anisotropic
(Lemma [lem:sv_pair_exact]), the
Additive GRAPE (GRAPE-A) scoring uses a paired inverse-transpose
(Eq. [eq:add_transform]), which cancels
all multiplicative distortions and yields a pure additive term:
$$\begin{aligned}
\widetilde{\qb}_i^\top \widetilde{\kb}_j
&= \widehat{\qb}_i^\top \big(\Ib + i\,\Ab\big)^\top \big(\Ib - j\,\Ab^\top\big)\,\widehat{\kb}_j
\;=\; \widehat{\qb}_i^\top \big(\Ib + (i{-}j)\,\Ab^\top\big)\widehat{\kb}_j
\;=\; \widehat{\qb}_i^\top \Gb_{\mathrm{add}}(j{-}i)^{-{\top}} \widehat{\kb}_j ,
\end{aligned}$$ since $(\Ab^\top)^2=\mathbf{0}$. This reproduces the
exact relative law
Eq. [eq:add_relative_law] and the
closed form
Eq. [eq:add_closed_form]
(e.g. Eq. [eq:add_bias_keygated]),
independently of $\sigma_\pm(H(s))$.

**GRAPE-AP as a path‑integral unipotent.** Fix a head $h$ and endpoint
$t$. The per-row path product in
Section 5 is $$\begin{aligned}
\prod_{\ell=j+1}^t \big(\Ib + \psi_h(t,\ell)\,E\big)
\;=\; \Ib + \Big(\sum_{\ell=j+1}^t \psi_h(t,\ell)\Big) E
\;=\; \Ib + s_h(t,j)\,E,    
\end{aligned}$$ because $E^2=\mathbf{0}$. Thus GRAPE-AP inherits the
unipotent spectrum of
Prop. [prop:unipotent_spectrum]
with row-dependent $s=s_h(t,j)\le 0$ (since $\psi_h\le 0$ by
construction). Its only two nontrivial singular values are
exactly [eq:sv_pair_closed_form]
with $s\mapsto s_h(t,j)$; the rest equal $1$. Consequently,
$$\kappa_2\big(\text{PA factor}\big)
\,=\, \frac{\sigma_+\big(s_h(t,j)\big)}{\sigma_-\big(s_h(t,j)\big)}
\,=\,1 + 2\,|s_h(t,j)| + O\big(s_h(t,j)^2\big),$$ while the determinant
remains $1$ and eigenvalues are all $1$. As in Additive GRAPE (GRAPE-A),
the paired inverse-transpose used in the bilinear scoring removes any
multiplicative anisotropy, leaving the bounded additive term $b_h(t,j)$
in Eq. [eq:grape_pa_bias].

**Implications.** Now we summarize the implications of previous results.
For all $s$, $\Hb(s)$ is invertible with $\Hb(s)^{-1}=\Ib - s\Ab$;
eigenvalues do not grow with offset length (spectral radius $=1$). The
operator norm grows at most linearly in $|s|$
(Lemma [lem:sv_bounds]) and is exactly
characterized in the rank-1 canonical cases
(Lemma [lem:sv_pair_exact]).

Secondly, $\det \Hb(s)=1$ implies no net volume change; any expansion
along one direction is exactly balanced by contraction along its paired
direction (product $\sigma_+\sigma_-=1$). Despite anisotropy, the
GRAPE-A and GRAPE-AP logits remain exactly relative because the key
transform uses $\Hb(s)^{-{\top}}$, algebraically eliminating
multiplicative distortion and yielding the closed-form additive bias
(Eqs. [eq:add_relative_law],
[eq:add_closed_form],
[eq:grape_pa_bias]).

## Comparison to PaTH Attention

PaTH Attention  proposes a contextual multiplicative position map given
by a cumulative product of identity-plus-rank-one matrices
$$\Hb_t = \Ib - \beta_t\, \wb_t \wb_t^\top, \qquad \|\wb_t\|_2=1,\quad \beta_t\in(0,2),$$
applied along the path between key position $j$ and query position $i$
as $\prod_{s=j+1}^{i} \Hb_s$ (see Section 2 of the PaTH paper). In
contrast to **GRAPE-M** factors, each $\Hb_t$ is *not* orthogonal unless
$\beta_t\in\{0,2\}$. This has immediate spectral consequences.

**Per-step spectrum.** Since $\Hb_t$ is symmetric rank-1 perturbation of
the identity with projector $\Pb_t := \wb_t \wb_t^\top$,
$$\sigma(\Hb_t)=\{\,1-\beta_t,\,\underbrace{1,\ldots,1}_{d-1}\,\},\qquad
\det(\Hb_t)=1-\beta_t,\qquad
\|\Hb_t\|_2=\max\{1,|1-\beta_t|\}=1.$$ Thus $\Hb_t$ is norm nonexpansive
(operator norm $1$) but *not norm-preserving* unless
$\beta_t\in\{0,2\}$. Singular values equal the absolute eigenvalues
because $\Hb_t$ is symmetric; the component along $\wb_t$ is scaled by
$|1-\beta_t|1$ (a design choice in PaTH to allow negative
eigenvalues for state-tracking).

**Path product is contractive and near-singular.** Let
$\Pb_{j\to i}=\prod_{s=j+1}^{i} \Hb_s$. Submultiplicativity of singular
values gives
$$\sigma_{\max}(\Pb_{j\to i}) \le \prod_{s=j+1}^{i}\|\Hb_s\|_2 = 1,
\qquad
\sigma_{\min}(\Pb_{j\to i}) \ge \prod_{s=j+1}^{i} \sigma_{\min}(\Hb_s) = \prod_{s=j+1}^{i} |1-\beta_s|.$$
Hence $\Pb_{j\to i}$ is (at best) nonexpansive, with a worst-case
exponential lower bound on the smallest singular value governed by the
path-length product of $|1-\beta_s|$. Whenever some $\beta_s$ is close
to $1$, $\Hb_s$ is nearly singular (and exactly singular if
$\beta_s=1$), driving $\sigma_{\min}(\Pb_{j\to i})$ toward zero. Volume
contraction is quantified by
$$\det(\Pb_{j\to i}) = \prod_{s=j+1}^{i} (1-\beta_s),$$ which typically
decays exponentially in $i-j$ unless $\beta_s$ concentrates at the
orthogonal endpoints $\{0,2\}$.

**Aligned-plane special case.** If the directions are time-invariant,
$\wb_s\equiv \wb$, then $\Pb_t = \wb \wb^\top$ is an idempotent
projector and the factors commute: $$\prod_{s=j+1}^{i} \Hb_s
= \prod_{s=j+1}^{i} \big(\Ib - \beta_s \Pb\big)
= \Ib - \Big(1 - \prod_{s=j+1}^{i} (1-\beta_s)\Big) \Pb,$$ so the
eigenvalue along $w$ is exactly $\prod_{s=j+1}^{i}(1-\beta_s)$, making
the contraction along $w$ explicit and exponential in path length unless
$\beta_s\in\{0,2\}$.

**Implications for long-context modeling.** Because the PaTH transport
multiplies the Q/K bilinear by $\Pb_{j\to i}$, any persistent deviation
of $\beta_t$ from $\{0,2\}$ yields cumulative energy loss along a moving
one-dimensional subspace. This concentrates mass in progressively fewer
directions and can flatten or attenuate long-range logits
$\qb_i^\top \Pb_{j\to i} \kb_j$ as $i-j$ grows, unless additional
renormalizations or forget-gates are introduced. In contrast,
**GRAPE-M** maps lie in $\mathop{\mathrm{SO}}(d)$, so for both
non‑contextual and contextual types, all singular values are $1$;
volumes and norms are preserved, and Lyapunov exponents are $0$,
avoiding contraction‑induced degradation of long‑range interactions.



For $\Hb_t = \Ib - \beta_t \wb_t \wb_t^\top$ with $\|\wb_t\|=1$, $\Hb_t$
is orthogonal iff $\beta_t\in\{0,2\}$. For
$\beta_t\in(0,2)\setminus\{0,2\}$, $\Hb_t$ is symmetric, diagonalizable
with eigenvalues in $(-1,1]\cup\{1\}$, and strictly contractive on
$\mathrm{span}\{\wb_t\}$.



[^1]: Core contribution;  $^\dagger$Corresponding authors.

[^2]: Definitions of $\mathop{\mathrm{SO}}(d)$ and other mathematical
    terms are postponed to
    Table [tab:notation_summary] in
    the Appendix.

[^3]: Our experiments take $g(z)=\log(\text{Sigmoid}(z))$; then
    $g'(z)=1-\text{Sigmoid}(z)\in(0,1)$, ensuring $1$-Lipschitzness.