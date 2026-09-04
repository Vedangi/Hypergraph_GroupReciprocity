# Graph Reciprocity and Multiplicity-Aware Directed-Hypergraph Generalizations

This note collects non-temporal definitions of reciprocity used for directed
graphs, weighted directed graphs, and directed multigraphs, followed by
multiplicity-aware generalizations for single-sender directed hypergraphs.

The literature definitions and the proposed hypergraph extensions are kept
separate. The hypergraph measures in Sections 8–13 are proposed constructions;
their main design requirements are:

1. recover the existing participation measures on deduplicated directed
   hypergraphs;
2. recover standard directed-multigraph reciprocity when every hyperedge is
   dyadic;
3. distinguish broad but rare sender-role participation from balanced repeated
   participation; and
4. remain in the interval \([0,1]\).

The note does not consider temporal ordering or response latency.

---

## 1. Basic graph notation and a counting convention

Let \(G=(V,E)\) be a loop-free directed graph with binary adjacency matrix
\(A=(a_{ij})\), where

\[
a_{ij}=\mathbf 1\{(i,j)\in E\}.
\]

Write

\[
L=\sum_{i\ne j}a_{ij}
\]

for the number of directed edges. Let \(M\) be the number of **unordered
mutual dyads**, and let \(A_d\) be the number of unordered asymmetric dyads.
Then

\[
L=2M+A_d.
\]

This convention matters. A mutual dyad contains two directed edges. Therefore:

- \(M/L\) has maximum \(1/2\);
- \(2M/L\) has maximum \(1\); and
- the usual phrase “fraction of edges that are reciprocated” means \(2M/L\).

Self-loops should normally be excluded because they are automatically equal to
their own reverse and carry no information about reciprocity between distinct
vertices.

---

## 2. Binary edge reciprocity

The common edge-based definition is

\[
r_{\mathrm{edge}}
=
\frac{\sum_{i\ne j}a_{ij}a_{ji}}
     {\sum_{i\ne j}a_{ij}}
=
\frac{2M}{2M+A_d}.
\]

It is the fraction of directed edges whose reverse edge exists. It equals zero
when no edge has a reverse and one when every edge has a reverse.

This is the standard binary quantity discussed by Newman et al. and by
Garlaschelli and Loffredo, and is the convention implemented by common graph
software such as NetworkX.

---

## 3. Dyad-based reciprocity and mutual-dyad density

An alternative is the fraction of active unordered dyads that are mutual:

\[
r_{\mathrm{dyad}}
=
\frac{M}{M+A_d}.
\]

This weights each active unordered pair equally. In a binary graph it is a
monotone transform of edge reciprocity:

\[
r_{\mathrm{edge}}
=
\frac{2r_{\mathrm{dyad}}}{1+r_{\mathrm{dyad}}},
\qquad
r_{\mathrm{dyad}}
=
\frac{r_{\mathrm{edge}}}{2-r_{\mathrm{edge}}}.
\]

Another quantity is the density of mutual dyads,

\[
d_{\mathrm{mutual}}
=
\frac{M}{\binom{|V|}{2}}.
\]

Unlike \(r_{\mathrm{dyad}}\), this divides by all possible dyads and therefore
combines graph density with mutuality. It should be called a reciprocal-link or
mutual-dyad density rather than an unqualified reciprocity coefficient.

---

## 4. Density-adjusted correlation reciprocity

Raw binary reciprocity generally increases with graph density even if opposite
edges are independent. Define directed edge density

\[
\bar a=\frac{L}{|V|(|V|-1)}.
\]

Garlaschelli and Loffredo define the correlation between \(a_{ij}\) and
\(a_{ji}\):

\[
\rho
=
\frac{
\sum_{i\ne j}(a_{ij}-\bar a)(a_{ji}-\bar a)
}{
\sum_{i\ne j}(a_{ij}-\bar a)^2
}
=
\frac{r_{\mathrm{edge}}-\bar a}{1-\bar a}.
\]

Its interpretation is relative to the independent-edge density baseline:

- \(\rho>0\): more reciprocal than predicted by density alone;
- \(\rho=0\): consistent with that baseline; and
- \(\rho<0\): anti-reciprocal relative to that baseline.

This measure is more suitable than raw \(r_{\mathrm{edge}}\) for some
cross-network comparisons, although a degree-preserving null model may be more
appropriate when the degree sequence is heterogeneous.

---

## 5. Weighted and directed-multigraph reciprocity

Let \(w_{ij}\ge 0\) be a directed weight. Squartini et al. define the
reciprocated part of the \(i,j\) interaction as

\[
w_{ij}^{\leftrightarrow}=\min(w_{ij},w_{ji})
=w_{ji}^{\leftrightarrow}.
\]

The total weight and total reciprocated weight are

\[
W=\sum_{i\ne j}w_{ij},
\qquad
W^{\leftrightarrow}
=\sum_{i\ne j}\min(w_{ij},w_{ji}),
\]

and weighted reciprocity is

\[
r_w=\frac{W^{\leftrightarrow}}{W}.
\]

For a directed multigraph, let \(m_{ij}\in\mathbb N_0\) be the number of
parallel edges from \(i\) to \(j\). Taking \(w_{ij}=m_{ij}\) gives

\[
r_{\mathrm{multi}}
=
\frac{
2\sum_{i<j}\min(m_{ij},m_{ji})
}{
\sum_{i<j}(m_{ij}+m_{ji})
}.
\]

Thus \(\min(m_{ij},m_{ji})\) is the maximum number of opposite-direction edge
pairs, and the factor two counts the two directed edge occurrences in each
pair. Cazabet, Takeda, and Hamasaki use this minimum-of-parallel-edges
construction explicitly for a non-temporal interaction multigraph.

Two different questions should not be conflated:

1. **Support reciprocity:** threshold \(m_{ij}\) to
   \(a_{ij}=\mathbf 1\{m_{ij}>0\}\) and ask whether the reverse relation ever
   occurs.
2. **Multiplicity reciprocity:** use \(\min(m_{ij},m_{ji})\) and ask how much
   repeated interaction can be matched across directions.

For example, \((m_{ij},m_{ji})=(100,1)\) has perfect support reciprocity but
multiplicity reciprocity \(2/101\).

---

## 6. Dyadic balance and different aggregation schemes

A local balance coefficient is

\[
b_{ij}
=
\frac{2\min(w_{ij},w_{ji})}{w_{ij}+w_{ji}}
=
1-\frac{|w_{ij}-w_{ji}|}{w_{ij}+w_{ji}},
\]

defined for active dyads. It is one for equal directional weights and zero for
a one-way dyad.

There are two substantively different ways to aggregate local balance:

### Event- or volume-weighted aggregation

\[
\frac{
\sum_{i<j}(w_{ij}+w_{ji})b_{ij}
}{
\sum_{i<j}(w_{ij}+w_{ji})
}
=r_w.
\]

This answers: “What fraction of all interaction volume is reciprocated?”

### Dyad-weighted aggregation

\[
\frac{1}{|\mathcal D|}
\sum_{\{i,j\}\in\mathcal D}b_{ij},
\]

where \(\mathcal D\) is the set of active dyads. This answers: “How balanced is
the typical active dyad?”

These estimands need not agree. A few high-volume dyads can dominate the first
but not the second.

---

## 7. Null-model-adjusted and model-based reciprocity

A generic null-model correction is

\[
\rho_{\mathrm{NM}}
=
\frac{r-\mathbb E_{\mathrm{NM}}[r]}
     {1-\mathbb E_{\mathrm{NM}}[r]},
\]

and a standardized alternative is

\[
z_r
=
\frac{r-\mathbb E_{\mathrm{NM}}[r]}
     {\sqrt{\operatorname{Var}_{\mathrm{NM}}(r)}}.
\]

The null model must be reported. Possibilities include preserving total edge
count, in/out-degree sequences, total weight, or in/out-strength sequences.
Squartini et al. show that different strength-constrained null models can yield
different conclusions about weighted reciprocity.

In exponential random graph and Holland–Leinhardt-type models, reciprocity can
instead be a model parameter associated with the mutual-edge statistic

\[
M(A)=\sum_{i<j}a_{ij}a_{ji}.
\]

For example,

\[
\Pr(A)\propto
\exp\{\theta_L L(A)+\theta_M M(A)+\cdots\}.
\]

Here \(\theta_M\), rather than \(M/L\), quantifies the conditional tendency to
form an edge when its reverse exists, controlling for the remaining model
terms.

---

## 8. Single-sender directed multi-hypergraph notation

Let \(H=(V,\mathcal E)\) be a directed multi-hypergraph whose hyperedges have
one sender:

\[
e=(s\to R),
\qquad s\notin R.
\]

Its team and team size are

\[
T(e)=\{s\}\cup R,
\qquad k_{T(e)}=|T(e)|\ge 2.
\]

Hyperedges are in the same team when their underlying node sets agree. For
each team \(T\) and member \(i\in T\), define the sender-role multiplicity

\[
m_{T,i}
=
\#\{e\in\mathcal E:e=(i\to T\setminus\{i\})\}.
\]

Also define

\[
M_T=\sum_{i\in T}m_{T,i},
\qquad
q_T=\sum_{i\in T}\mathbf 1\{m_{T,i}>0\},
\qquad
|\mathcal E|=\sum_T M_T.
\]

Here:

- \(M_T\) is the number of hyperedge occurrences on team \(T\), counting
  duplicates;
- \(q_T\) is the number of distinct members observed in the sender role; and
- \(\mathbf m_T=(m_{T,i}:i\in T)\) is the sender-role multiplicity profile.

For the original deduplicated measures, the participation count of every edge
on team \(T\) is \(p_e=q_T\). This makes explicit that \(p_e\) includes the
focal sender.

---

## 9. Existing support-based participation measures

There is an off-by-one ambiguity in the supplied notation. The condition
\(p_e\ge1\) matches a definition in which \(p_e\) counts **other** participating
members and excludes the focal sender. But \(p_e=|e|\) and
\((p_e-1)/(|e|-1)\) match a definition in which \(p_e\) **includes** the focal
sender. No single convention makes all three supplied formulas consistent.

The two consistent conventions are equivalent:

| Convention | Any | Strict | Partial |
|---|---:|---:|---:|
| \(q_e\) includes the focal sender | \(\mathbf 1\{q_e\ge2\}\) | \(\mathbf 1\{q_e=|e|\}\) | \((q_e-1)/(|e|-1)\) |
| \(p_e=q_e-1\) counts only other senders | \(\mathbf 1\{p_e\ge1\}\) | \(\mathbf 1\{p_e=|e|-1\}\) | \(p_e/(|e|-1)\) |

This note uses \(p_e=q_T\), including the focal sender, because that is the
convention implied by the supplied strict and partial formulas. With that
clarification, for a deduplicated hypergraph the three measures are

### Any-group participation

\[
s_1(H)
=
\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}\mathbf 1\{p_e\ge 2\}.
\]

This is written with \(p_e\ge2\), rather than \(p_e\ge1\), because \(p_e\)
includes the focal sender. Equivalently, at least one member other than the
focal sender must initiate an edge for the same team. If \(p_e\) were defined
to exclude the focal sender, the condition would instead be \(p_e\ge1\).

### Strict participation reciprocity

\[
r_1(H)
=
\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}\mathbf 1\{p_e=|e|\}.
\]

### Partial participation reciprocity

\[
r_2(H)
=
\frac{1}{|\mathcal E|}
\sum_{e\in\mathcal E}
\frac{p_e-1}{|e|-1}.
\]

Applying these support indicators unchanged to duplicate occurrences produces
undesirable behavior. Once every sender role is represented, arbitrarily many
copies from one dominant sender receive full credit. The vector
\((100,1,1)\), for example, is treated like \((1,1,1)\).

The remedy below replaces independent scoring of duplicates by capacities
computed from \(\mathbf m_T\).

---

## 10. Proposed multiplicity-aware any-group participation

For sender \(i\), there are \(m_{T,i}\) occurrences initiated by \(i\) and
\(M_T-m_{T,i}\) initiated by other team members. The amount of \(i\)'s
activity supportable by other-member activity is

\[
\min\{m_{T,i},M_T-m_{T,i}\}.
\]

Define

\[
\boxed{
s_1^{\mathrm{multi}}(H)
=
\frac{1}{|\mathcal E|}
\sum_T\sum_{i\in T}
\min\{m_{T,i},M_T-m_{T,i}\}.
}
\]

The team-local version is

\[
s_{1,T}^{\mathrm{multi}}
=
\frac{
\sum_{i\in T}\min\{m_{T,i},M_T-m_{T,i}\}
}{M_T}.
\]

This is a permissive, **any-other-sender capacity** measure. It retains the
original support semantics: activity from any other sender role may support the
focal role, and no particular responding member is required.

The construction does not impose a single global one-to-one pairing among all
event copies. If exclusive event pairing is scientifically required, Section
14 gives an alternative; that alternative does not exactly recover the
original any-participation measure for all deduplicated teams.

---

## 11. Proposed multiplicity-aware strict participation reciprocity

A strict all-member reciprocal unit consumes one hyperedge occurrence from
every member of team \(T\). The maximum number of complete units is

\[
c_T=\min_{i\in T}m_{T,i}.
\]

Each unit contains \(k_T\) hyperedge occurrences. Define

\[
\boxed{
r_1^{\mathrm{multi}}(H)
=
\frac{1}{|\mathcal E|}
\sum_T k_T\min_{i\in T}m_{T,i}.
}
\]

The local score is

\[
r_{1,T}^{\mathrm{multi}}
=
\frac{k_T\min_{i\in T}m_{T,i}}{M_T}.
\]

It is zero if any team member never sends, and equals one exactly when all team
members send equally often. A single occurrence by a rare sender can complete
only one strict reciprocal unit, rather than reciprocating arbitrarily many
copies from a dominant sender.

---

## 12. Proposed multiplicity-aware partial participation reciprocity

For each unordered pair of possible sender roles \(i,j\in T\), match their
sender occurrences using

\[
\min(m_{T,i},m_{T,j}).
\]

Normalize the accumulated pairwise overlap by the \(k_T-1\) possible other
sender roles per member:

\[
\boxed{
r_2^{\mathrm{multi}}(H)
=
\frac{1}{|\mathcal E|}
\sum_T
\frac{2}{k_T-1}
\sum_{\{i,j\}\subseteq T}
\min(m_{T,i},m_{T,j}).
}
\]

The team-local score is

\[
r_{2,T}^{\mathrm{multi}}
=
\frac{
2\sum_{\{i,j\}\subseteq T}\min(m_{T,i},m_{T,j})
}{
(k_T-1)M_T
}.
\]

This is a pairwise sender-role overlap measure. It gives partial credit when
several, but not all, members share the sender role, and it also accounts for
imbalance in their repeated participation.

The factor \(2\) counts the two sender occurrences in a matched pair. The
factor \(k_T-1\) averages the pairwise credit over all possible counterpart
sender roles. The normalization follows from

\[
2\sum_{i<j}\min(m_{T,i},m_{T,j})
\le
\sum_{i<j}(m_{T,i}+m_{T,j})
=(k_T-1)M_T.
\]

Hence \(0\le r_{2,T}^{\mathrm{multi}}\le1\).

---

## 13. Reduction results

### 13.1 Recovery of the deduplicated hypergraph measures

Suppose there are no duplicate oriented hyperedges, so
\(m_{T,i}\in\{0,1\}\), and let \(q_T\) be the number of active sender roles.

For any participation,

\[
\sum_i\min(m_{T,i},M_T-m_{T,i})
=q_T\mathbf 1\{q_T\ge2\}.
\]

This is exactly the total contribution of the \(q_T\) edges on team \(T\) to
the original any-group measure.

For strict participation,

\[
k_T\min_i m_{T,i}
=
\begin{cases}
k_T,&q_T=k_T,\\
0,&q_T<k_T,
\end{cases}
=q_T\mathbf 1\{q_T=k_T\}.
\]

For partial participation, exactly \(\binom{q_T}{2}\) pairs have overlap one:

\[
\frac{2}{k_T-1}\sum_{i<j}\min(m_{T,i},m_{T,j})
=
\frac{2\binom{q_T}{2}}{k_T-1}
=
q_T\frac{q_T-1}{k_T-1}.
\]

This equals the sum of the original partial scores
\((q_T-1)/(k_T-1)\) over the \(q_T\) edges on team \(T\). Thus all three
proposed definitions recover the support-based definitions exactly on a
deduplicated hypergraph.

### 13.2 Recovery of directed-multigraph reciprocity

Suppose every team is dyadic. For \(T=\{i,j\}\), set

\[
x=m_{T,i}=m_{ij},
\qquad
y=m_{T,j}=m_{ji},
\qquad
k_T=2.
\]

The three team numerators all become

\[
2\min(x,y):
\]

\[
\min(x,y)+\min(y,x)=2\min(x,y),
\]

\[
2\min(x,y),
\]

and

\[
\frac{2}{2-1}\min(x,y)=2\min(x,y).
\]

Therefore

\[
\boxed{
s_1^{\mathrm{multi}}(H)
=r_1^{\mathrm{multi}}(H)
=r_2^{\mathrm{multi}}(H)
=
\frac{2\sum_{i<j}\min(m_{ij},m_{ji})}
     {\sum_{i<j}(m_{ij}+m_{ji})}
}
\]

whenever all hyperedges have size two. This is exactly standard
directed-multigraph/weighted reciprocity.

### 13.3 Range and ordering

Each proposed local and global score lies in \([0,1]\). At the team level the
definitions also obey

\[
r_{1,T}^{\mathrm{multi}}
\le
r_{2,T}^{\mathrm{multi}}
\le
s_{1,T}^{\mathrm{multi}}.
\]

For the first inequality, for every pair \(i,j\),

\[
\min(m_{T,i},m_{T,j})\ge \min_{v\in T}m_{T,v}.
\]

Summing over \(\binom{k_T}{2}\) pairs yields

\[
r_{2,T}^{\mathrm{multi}}
\ge
\frac{k_T\min_v m_{T,v}}{M_T}
=r_{1,T}^{\mathrm{multi}}.
\]

For the second inequality, fix \(i\). Since

\[
\sum_{j\ne i}\min(m_{T,i},m_{T,j})
\le
(k_T-1)\min\{m_{T,i},M_T-m_{T,i}\},
\]

summing over \(i\) gives
\(r_{2,T}^{\mathrm{multi}}\le s_{1,T}^{\mathrm{multi}}\).

---

## 14. Optional exclusive-pairing variant of “any” reciprocity

The proposed \(s_1^{\mathrm{multi}}\) preserves the semantics of the original
support measure: an occurrence can provide evidence that more than one sender
role has some other-member participation. If instead every event occurrence
must belong to at most one reciprocal pair, consider a maximum matching among
event copies, where two copies can be paired only if they have different
senders.

For team \(T\), the maximum number of event occurrences covered by such pairs
is

\[
C_T^{\mathrm{pair}}
=
2\min\left\{
\left\lfloor\frac{M_T}{2}\right\rfloor,
M_T-\max_{i\in T}m_{T,i}
\right\}.
\]

An exclusive-pairing score is therefore

\[
s_{\mathrm{pair}}(H)
=
\frac{\sum_T C_T^{\mathrm{pair}}}{|\mathcal E|}.
\]

It reduces to standard multigraph reciprocity for dyads. However, it does not
recover the original any-group score for every deduplicated hypergraph: a team
with three distinct one-time senders has original any-group score one, whereas
only two of its three occurrences can belong to disjoint pairs. This is why it
should be treated as a different estimand, not as the primary extension of
\(s_1\).

---

## 15. Worked examples

### Dyad with strongly imbalanced multiplicities

For \(T=\{a,b\}\) and

\[
(m_{T,a},m_{T,b})=(100,1),
\]

all three multiplicity-aware measures give

\[
\frac{2\min(100,1)}{101}=\frac{2}{101}.
\]

The collapsed binary support graph instead has reciprocity one. Both are valid,
but they answer different questions.

### Three-member team with full but imbalanced sender support

For

\[
\mathbf m_T=(100,1,1),
\qquad M_T=102,
\qquad k_T=3,
\]

the support-only definitions treat all three sender roles as present. The
multiplicity-aware scores are

\[
s_{1,T}^{\mathrm{multi}}
=
\frac{\min(100,2)+\min(1,101)+\min(1,101)}{102}
=\frac{4}{102},
\]

\[
r_{1,T}^{\mathrm{multi}}
=
\frac{3\min(100,1,1)}{102}
=\frac{3}{102},
\]

and

\[
r_{2,T}^{\mathrm{multi}}
=
\frac{2[\min(100,1)+\min(100,1)+\min(1,1)]}
     {(3-1)102}
=\frac{3}{102}.
\]

### Four-member team with partial support

For

\[
\mathbf m_T=(8,3,1,0),
\qquad M_T=12,
\qquad k_T=4,
\]

the scores are

\[
s_{1,T}^{\mathrm{multi}}
=\frac{\min(8,4)+\min(3,9)+\min(1,11)}{12}
=\frac{8}{12},
\]

\[
r_{1,T}^{\mathrm{multi}}=0,
\]

and

\[
r_{2,T}^{\mathrm{multi}}
=
\frac{2(3+1+1)}{3\cdot12}
=\frac{10}{36}.
\]

The strict score is zero because one member never sends. The partial score
captures overlap among the three observed sender roles, and the any score asks
only whether activity is supportable by some other sender.

---

## 16. Micro versus macro aggregation for teams

The global definitions above are event-weighted:

\[
r^{\mathrm{micro}}(H)
=
\frac{\sum_T M_T r_T}{\sum_T M_T}.
\]

They answer questions such as: “What fraction of all communication activity is
reciprocal?” Event weighting is also what makes the dyadic restriction equal
to conventional global multigraph reciprocity.

If the scientific unit is instead the team, use

\[
r^{\mathrm{macro}}(H)
=
\frac{1}{|\mathcal T|}
\sum_{T\in\mathcal T}r_T,
\]

where \(\mathcal T=\{T:M_T>0\}\). This answers: “How reciprocal is the typical
active team?”

It is useful to report both, because a few prolific teams may dominate the
micro score. No aggregation can simultaneously weight all teams equally and
reduce to conventional edge-weighted global multigraph reciprocity in every
dyadic multigraph: those are different estimands.

---

## 17. Recommended reporting scheme

For a directed multi-hypergraph, report:

1. the support/deduplicated scores \(s_1,r_1,r_2\), which measure whether
   sender roles are broadly represented at all;
2. the event-weighted multiplicity scores
   \(s_1^{\mathrm{multi}},r_1^{\mathrm{multi}},r_2^{\mathrm{multi}}\), which
   measure how much repeated activity is supported across sender roles;
3. macro-averaged versions of the multiplicity scores, if the typical team is
   scientifically meaningful; and
4. a null-model comparison when networks differ substantially in team sizes,
   sender activity, or team-event counts.

The support and multiplicity measures should be reported as complementary,
rather than as competing definitions. Their difference is itself informative:
high support reciprocity with low multiplicity reciprocity indicates broad but
highly unequal participation.

Among the new measures, \(r_2^{\mathrm{multi}}\) is the natural primary
breadth-sensitive statistic. The quantities \(r_1^{\mathrm{multi}}\) and
\(s_1^{\mathrm{multi}}\) provide strict and permissive companions.

---

## References

1. Newman, M. E. J., Forrest, S., and Balthrop, J. (2002). “Email networks and
   the spread of computer viruses.” *Physical Review E*, 66, 035101.
   <https://doi.org/10.1103/PhysRevE.66.035101>

2. Garlaschelli, D., and Loffredo, M. I. (2004). “Patterns of Link Reciprocity
   in Directed Networks.” *Physical Review Letters*, 93, 268701.
   <https://doi.org/10.1103/PhysRevLett.93.268701>
   Open manuscript: <https://arxiv.org/abs/cond-mat/0404521>

3. Squartini, T., Picciolo, F., Ruzzenenti, F., and Garlaschelli, D. (2013).
   “Reciprocity of weighted networks.” *Scientific Reports*, 3, 2729.
   <https://doi.org/10.1038/srep02729>
   Open manuscript: <https://arxiv.org/abs/1208.4208>

4. Cazabet, R., Takeda, H., and Hamasaki, M. (2015). “Characterizing the nature
   of interactions for cooperative creation in online social networks.”
   *Social Network Analysis and Mining*, 5, 43.
   <https://doi.org/10.1007/s13278-015-0284-y>

5. Wang, C., Strathman, A., Lizardo, O., Hachen, D., Toroczkai, Z., and Chawla,
   N. V. (2013). “A dyadic reciprocity index for repeated interaction
   networks.” *Network Science*, 1(1), 31–48.
   <https://doi.org/10.1017/nws.2012.5>
   Open manuscript: <https://arxiv.org/abs/1108.2822>

6. Holland, P. W., and Leinhardt, S. (1981). “An exponential family of
   probability distributions for directed graphs.” *Journal of the American
   Statistical Association*, 76(373), 33–50.
   <https://doi.org/10.1080/01621459.1981.10477598>

7. Gemmetto, V., Squartini, T., Picciolo, F., Ruzzenenti, F., and Garlaschelli,
   D. (2016). “Multiplexity and multireciprocity in directed multiplexes.”
   *Physical Review E*, 94, 042316. This is relevant when duplicate-looking
   hyperedges represent distinct relation types or layers rather than repeated
   occurrences of the same relation.
   <https://doi.org/10.1103/PhysRevE.94.042316>
   Open manuscript: <https://arxiv.org/abs/1411.1282>
