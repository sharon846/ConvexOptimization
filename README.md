# ConvexOptimization
Various classic problems from course, as learned with boyd

## 1. PnP
The main goal of this course and research is to try and formulate the PnP problem as a convex one. 
By that, maybe solve the long time opened problem. In Folder (1) you will find the transition from the 
original problem to a quadratic one. There is also a python script that tries to solve the problem. 
Of course it will fail, as the problem (in its current form) is a non convex one.

## 2. PnP via SDP
One possible approach to make PnP a convex problem is look at the largest eigenvalue of matrix. 
The natrual definiton, $\lambda^*=\max_{||x||=1} ||Ax||$
is non convex in x. denote the flattened rotation matrix we seek as x, the paper shows how the problem of PnP is 
convex in x. However, it ignores the constraint that x must represent a flattened rotation matrix. Moreover, it is
not possible to enforce a rank of 1 on a matrix. The python script finds a solution, which is indeed, not a rotation matrix.

## 3. PnP via polynoms
As shown in the (1), the problem is 4d polynomial in q=(x,y,z,w) s.t. $||q||=1$ (in quaternion representation).
Therefore, we can try the direct approach of minimizing it, we barely can solve 4d degree polynom with one variable, so 4 variables?
Hence, Following the [paper of  the paper of M. Osadchy and D. Keren],
https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nZEtlZoAAAAJ&sortby=pubdate&citation_for_view=nZEtlZoAAAAJ:fPk4N6BV_jEC
We only need to find the first $\gamma$ such that $q4(x,y,z,w)=p4(x,y,z,w)-\gamma||(x,y,z,w)||^2\geq 0$ (for any q).
In folder (3) there is:
- A nonconvex solver which gives pretty accurate approxmation.
- A python script to convert a given problem to a 4d polynom in q=(x,y,z,w).
- A text file, contains the required data to turn $q4$ into an sdp problem with $q=[x^2,xy,xz,xw,y^2,yz,yw,z^2,zw,w^2]$.
This might be a very good direction, but need somone who will continue this.

## 4. SDP cones
Exploring little more properties and behavior of SDP matrices. At least for symmteric 2x2 matrices, there is an isomorphism
between $\mathbb{R}^{2\times 2}\rightarrow \mathbb{R}^3$, so it can be visualized (using desmos/3d).

## 5. Multivariate roots approximation
Following (3) and after exploring generic root finding for functions, built a generic script for very good root approximation. 
In that folder you will also find an example for pnp, usually solution is close with 4 digits precision. 
However, this huristic is not practical for embedding, as the real running times should be milliseconds.

## 6. Ellipsoid method
Following Dan Feldman[https://simons.berkeley.edu/sites/default/files/docs/9692/provable-learning-real-time-big-data-using-core-sets.pdf] slide 39
It is possible to approximate the 4d polynom, using the fact that $p4=||r4||^2$, the squared norm 2 of another function.
Using John's ellipsoid, can upper bound the norm above and below, but it's not the best solution.
Murda's amazing source code tries to do that, but I failed to make this practical for the problem.
https://github.com/muradtuk/LzModelCompression

## 7. PnP via Geometric programming
Another approach to deal with the PnP problem might be using GP. In general GP is an approach for
transforming posynomial problems to convex using exp and log, constrained by other posynomials and monomials.
This approach required more research, since in the folder there is only the start.

## 8. L2 optimaization relaxation
Leaving PnP, focus on minimizing $||Ax-b||_2^2$ such that $||x||=1$. 
- The 'R2' paper formulates the simple case for $x\in\mathbb{R}^2. The python script exectues that algorithm.
- The 'SDP' paper gives pretty easy solution, based on Boyd appendix B. Implementation included.
- Another convex approach using "subconvex" problems on the original problem. The true reason for this approach to work
is hidden in the proof of Boyd for strong duality.

## 9. L1 optimaization relaxation
Tried similar approach of "subconvex" for the open problem of L1 optimization. 
The paper gives a correct algorithm for maximization problem. However, for the minimization problem its incorrect.
A counter example here: https://www.desmos.com/3d/66d5bc932e

## 10. Appendix
A paper shows how to convert the rank=1 constrain of a matrix to a SDP problem.



