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
- Mijn evaluatiefunctie berekent een score voor een gegeven 'state' door een gewogen som van verschillende heuristische componenten te nemen. Het doel is om Pacman te sturen naar situaties die leiden tot een hogere score en winst, terwijl direct gevaar wordt vermeden.

- De score wordt als volgt opgebouwd:
    - **Basisscore:** De score begint met de huidige gamestate-score (`currentGameState.getScore()`). Dit vormt het fundament van de evaluatie.
    - **Voedsel:**
        - Er wordt een significante straf (`-4.0 * len(food)`) gegeven op basis van het aantal resterende voedselbolletjes. Dit moedigt de agent sterk aan om het level uit te spelen.
        - Er wordt een bonus gegeven op basis van de nabijheid tot het dichtstbijzijnde voedselbolletje (`+ 3.0 / (d_food + 1.0)`). Dit zorgt ervoor dat Pacman actief naar voedsel zoekt en niet passief blijft.
    - **Spoken (Ghosts):** Dit is het meest complexe en belangrijkste deel, opgesplitst in twee scenario's:
        - **Bange Spoken (`scaredTimer > 0`):** Pacman wordt sterk aangemoedigd om bange spoken te achtervolgen. Hij krijgt een bonus voor nabijheid (`+ 10.0 / (d + 1.0)`) en een zeer grote extra bonus (`+ 50.0`) voor het daadwerkelijk 'opeten' van het spook (wanneer `d == 0`).
        - **Actieve Spoken:** Overleving is hier prioriteit. Als een spook gevaarlijk dichtbij is (`d <= 1`), wordt een *extreem hoge negatieve penalty* (`-1000.0`) toegepast om deze situatie te allen tijde te vermijden. Voor actieve spoken die verder weg zijn, geldt een kleinere, afnemende penalty (`-5.0 / d`).
    - **Capsules (Power Pellets):**
        - Er wordt een strategische bonus toegepast voor het naderen van capsules (`+ 15.0 / (d_cap + 1.0)`).
        - **Cruciaal:** Deze bonus wordt enkel geactiveerd als Pacman in direct gevaar is (`danger_close == True`). Dit moedigt de agent aan om capsules te gebruiken als een ontsnappingsmiddel, in plaats van ze onnodig te verbruiken.