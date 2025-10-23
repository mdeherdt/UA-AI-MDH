# Artificial Intelligence - Report

**Name:** Mathis De Herdt

**Student number:** s0233480



## (Un)Informed Search - Discuss heuristic

**Describe your heuristic for question 1:**

- De heuristiek die ik heb geïmplementeerd is een overweging van 2 heuristieken. Het doel is om een zo hoog mogelijke (informatieve) schatting te geven van de resterende kosten, zonder ooit de werkelijke kosten te overschatten (admissibiliteit).

  - Mijn heuristiek h(n) berekent het maximum van twee afzonderlijke heuristieken:

  - Heuristiek 1 (Afstand Pacman tot verste dot): De Manhattan-afstand van Pacman's huidige positie tot de voedseldot die het verst weg is.

  - Heuristiek 2 (Grootste afstand tussen dots): De Manhattan-afstand tussen de twee voedseldots die het verst van elkaar verwijderd zijn.

  - De uiteindelijke heuristiek is dus: h(n) = max(h1, h2)

- **Admissibility:**
  - Ja deze heuristiek is admissible omdat het nooit de werkelijke resterende kosten zal overschatten.
  - Heuristiek 1 is admissible omdat Pacman minstens de afstand tot de verste dot moet afleggen.
  - Heuristiek 2 is ook admissible omdat Pacman minstens de afstand tussen de twee verste dots moet afleggen om ze allemaal te verzamelen.
  - Door het maximum van deze twee heuristieken te nemen, blijft de resulterende heuristiek ook admissible.
- **Consistency:**
  - Ja deze heuristiek is consistent omdat het voldoet aan de driehoekongelijkheid.
  - Heuristiek 1 is consistent omdat de afstand tot de verste dot nooit meer kan toenemen dan de werkelijke kosten om daar te komen.
  - Heuristiek 2 is ook consistent omdat de afstand tussen de twee verste dots nooit meer kan toenemen dan de werkelijke kosten om ze te bereiken.
  - Door het maximum van deze twee heuristieken te nemen, blijft de resulterende heuristiek ook consistent.

## Adversarial Search - Discuss evaluation function

**Describe your evaluation function for question 4:**
- [short description here]