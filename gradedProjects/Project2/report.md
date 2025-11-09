# Artificial Intelligence - Report 3 

**Name:** [Mathis De Herdt]

**Student number:** [s0233480]


## Bayesian Networks (BN)

## Q2: Exact Calculation

**Task:**  
Compute the exact probability of:  
 P(either=yes | xray=yes)

using the conditional probability tables from the appendix in the assignment.

**Show your work:**  
See attached file `BN_Q2.pdf`

**Exact Value:**  
P(either=yes | xray=yes) = 0,57603

**Comparison:**   
The sampling calculation was: 0.583924 this is slightly different from the exact value of 0.57603 which I manually calculated. The difference could be due to rounding errors or bad samples. However, the values are quite close, indicating that the approach used was generally correct.


---

## Q3: Exact with Extra Evidence

**Task:**  
Compute the exact probability  
P(either=yes | xray=yes, tub=yes).

**Derivation:**  
See attached file `BN_Q3.pdf`

**Exact Value:**  
P(either=yes | xray=yes, tub=yes) = 1,0


---

## Q4: D-Separation

**Task:**  
Determine whether each conditional independence statement holds in the Asia network.  
If it does True at leas(no active path)

| Statement                        | Independent? | Active Path (if any)           |
|----------------------------------|--------------|--------------------------------|
| (a) asia ⟂ xray                  | False        | asia -> tub -> either -> xray  |
| (b) tub ⟂ smoke \| either        | False        | tub -> either <- lung <- smoke |
| (c) tub ⟂ bronc                  | True         | (no active path)               |
| (d) tub ⟂ bronc \| dysp          | False        | tub -> either -> dysp <- bronc |
| (e) tub ⟂ bronc \| smoke, either | True         | (no active path)               |

---

## References

- [BNLearn Asia network](https://www.bnlearn.com/bnrepository/discrete-small.html#asia)  

---

## Notes

I used handwritten calculations for the exact probabilities in Q2 and Q3. The derivations are included in the attached PDF files respectively `BN_Q2.pdf` and `BN_Q3.pdf` For Q4, I analyzed the network structure to determine d-separation and active paths. 

---