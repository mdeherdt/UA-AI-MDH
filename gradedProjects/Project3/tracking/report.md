# Project 3: HMM - Uitgebreid Rapport

## Inleiding

In dit project heb ik gewerkt aan het implementeren van verschillende inferentie-algoritmen voor het volgen van spoken in het Pacman-spel. Pacman kan de spoken niet direct zien, maar krijgt alleen ruizige metingen van de afstand tot elk spook. Door gebruik te maken van Hidden Markov Models (HMM's) en verschillende inferentietechnieken, kan Pacman de waarschijnlijke locaties van de spoken bepalen en ze efficiënt vangen.

Het project bestond uit zeven hoofdopdrachten, waarbij ik zowel exacte inferentie (Forward Algorithm) als benaderende inferentie (Particle Filtering) heb geïmplementeerd. Hieronder beschrijf ik mijn aanpak en oplossingen voor elke opdracht.

## Opdracht 1: DiscreteDistribution Klasse en Observatie Waarschijnlijkheid (q5)

### Probleem
Deze opdracht bestond uit twee delen:
1. Het implementeren van de `normalize` en `sample` methoden in de `DiscreteDistribution` klasse
2. Het implementeren van de `getObservationProb` methode in de `InferenceModule` klasse

### Aanpak en Implementatie

#### normalize methode
Voor de `normalize` methode moest ik ervoor zorgen dat alle waarden in de distributie optellen tot 1, terwijl de verhoudingen tussen de waarden behouden blijven. Ik heb eerst de totale som van alle waarden berekend met de `total` methode. Als deze som 0 is, hoeft er niets te gebeuren. Anders deel ik elke waarde door de totale som:

```python
def normalize(self):
    total = self.total()
    if total == 0:
        return  # Doe niets als totaal 0 is
    
    for key in self:
        self[key] = self[key] / total
```

#### sample methode
Voor de `sample` methode moest ik een steekproef nemen uit de distributie, waarbij de kans dat een sleutel wordt gekozen evenredig is aan de bijbehorende waarde. Ik heb de volgende stappen geïmplementeerd:

1. Controleren of de totale som 0 is (in dat geval return None)
2. De items sorteren voor consistente volgorde
3. De distributie normaliseren indien nodig
4. Een willekeurige waarde kiezen en de bijbehorende sleutel vinden:

```python
def sample(self):
    if self.total() == 0:
        return None
    
    # Items ophalen en sorteren voor consistente volgorde
    items = sorted(self.items())
    keys = [item[0] for item in items]
    values = [item[1] for item in items]
    
    # Normaliseer de distributie indien nodig
    distribution = values
    if sum(distribution) != 1:
        total = sum(distribution)
        distribution = [val / total for val in distribution]
    
    # Kies een willekeurige waarde en vind de bijbehorende sleutel
    choice = random.random()
    i, total = 0, distribution[0]
    while choice > total and i < len(distribution) - 1:
        i += 1
        total += distribution[i]
    
    return keys[i]
```

#### getObservationProb methode
Voor de `getObservationProb` methode moest ik de waarschijnlijkheid berekenen van een ruizige afstandsmeting, gegeven de posities van Pacman en het spook. Ik heb de volgende gevallen afgehandeld:

1. Als het spook in de gevangenis zit:
   - Return 1.0 als de ruizige afstand None is
   - Return 0.0 als de ruizige afstand niet None is
2. Als de observatie None is en het spook niet in de gevangenis zit, return 0.0
3. Bereken de werkelijke afstand tussen Pacman en het spook
4. Gebruik `busters.getObservationProbability` om de waarschijnlijkheid te berekenen:

```python
def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
    # Controleer of het spook in de gevangenis zit
    if ghostPosition == jailPosition:
        if noisyDistance is None:
            return 1.0
        else:
            return 0.0
    
    # Controleer of de observatie None is
    if noisyDistance is None:
        return 0.0
    
    # Bereken de werkelijke afstand
    trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
    
    # Krijg de observatie waarschijnlijkheid
    return busters.getObservationProbability(noisyDistance, trueDistance)
```

### Resultaten
Alle tests voor q5 zijn geslaagd, wat bevestigt dat mijn implementaties correct zijn. De autograder gaf een score van 2/2 voor deze opdracht.

## Opdracht 2: Exacte Inferentie Observatie (q6)

### Probleem
In deze opdracht moest ik de `observeUpdate` methode in de `ExactInference` klasse implementeren om de belief distributie van de agent over spookposities bij te werken op basis van een observatie van Pacman's sensoren.

### Aanpak en Implementatie
Voor deze opdracht heb ik Bayes' regel toegepast: P(ghostPosition | observation) ∝ P(observation | ghostPosition) * P(ghostPosition).

Ik heb de volgende stappen geïmplementeerd:
1. Voor elke mogelijke positie in `self.allPositions`:
   - Bereken de waarschijnlijkheid van de observatie gegeven de positie van het spook en Pacman
   - Werk de belief bij door deze waarschijnlijkheid te vermenigvuldigen met de huidige belief
2. Normaliseer de resulterende distributie:

```python
def observeUpdate(self, observation, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    jailPosition = self.getJailPosition()
    
    # Werk de belief bij voor elke positie op basis van de observatie
    for position in self.allPositions:
        # Bereken P(observation | ghost position, pacman position)
        observationProb = self.getObservationProb(observation, pacmanPosition, position, jailPosition)
        
        # Werk de belief bij met Bayes' regel: P(ghost position | observation) ∝ P(observation | ghost position) * P(ghost position)
        self.beliefs[position] = observationProb * self.beliefs[position]
    
    self.beliefs.normalize()
```

### Resultaten
Alle tests voor q6 zijn geslaagd zonder inferentiefouten. De autograder bevestigt dat mijn implementatie van de `observeUpdate` methode correct is. De opdracht kreeg een score van 3/3.

## Opdracht 3: Exacte Inferentie met Tijdsverloop (q7)

### Probleem
In deze opdracht moest ik de `elapseTime` methode in de `ExactInference` klasse implementeren om de belief distributie bij te werken nadat er tijd is verstreken, rekening houdend met de mogelijke bewegingen van het spook.

### Aanpak en Implementatie
Voor deze opdracht heb ik de totale waarschijnlijkheid gebruikt: P(newPos) = Σ_oldPos P(newPos | oldPos) * P(oldPos).

Ik heb de volgende stappen geïmplementeerd:
1. Maak een nieuwe distributie aan om de bijgewerkte beliefs op te slaan
2. Voor elke mogelijke huidige positie van het spook:
   - Haal de huidige belief waarschijnlijkheid op voor deze positie
   - Als deze waarschijnlijkheid groter dan 0 is:
     - Krijg de distributie over nieuwe posities voor het spook
     - Werk de belief bij voor elke mogelijke nieuwe positie
3. Werk de belief distributie bij:

```python
def elapseTime(self, gameState):
    # Maak een nieuwe distributie aan om de bijgewerkte beliefs op te slaan
    newBeliefs = DiscreteDistribution()
    
    # Voor elke mogelijke huidige positie van het spook
    for oldPos in self.allPositions:
        # Krijg de huidige belief waarschijnlijkheid op deze positie
        oldProb = self.beliefs[oldPos]
        
        # Als er een niet-nul waarschijnlijkheid is dat het spook op deze positie is
        if oldProb > 0:
            # Krijg de distributie over nieuwe posities voor het spook
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            
            # Werk de belief bij voor elke mogelijke nieuwe positie
            for newPos, prob in newPosDist.items():
                # P(newPos) += P(newPos | oldPos) * P(oldPos)
                newBeliefs[newPos] += prob * oldProb
    
    # Werk de belief distributie bij
    self.beliefs = newBeliefs
```

### Resultaten
Alle tests voor q7 zijn geslaagd zonder inferentiefouten. De autograder bevestigt dat mijn implementatie van de `elapseTime` methode correct is. De opdracht kreeg een score van 2/2.

## Opdracht 4: Exacte Inferentie Volledige Test (q8)

### Probleem
In deze opdracht moest ik de `chooseAction` methode in de `GreedyBustersAgent` klasse implementeren om Pacman te laten bewegen naar het dichtstbijzijnde spook op basis van de belief distributie.

### Aanpak en Implementatie
Voor deze opdracht heb ik de volgende stappen geïmplementeerd:
1. Vind de meest waarschijnlijke positie van elk nog niet gevangen spook
2. Bepaal welk spook het dichtst bij is
3. Kies een actie die de doolhofafstand tot het dichtstbijzijnde spook minimaliseert:

```python
def chooseAction(self, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    legal = [a for a in gameState.getLegalPacmanActions()]
    livingGhosts = gameState.getLivingGhosts()
    livingGhostPositionDistributions = \
        [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
         if livingGhosts[i+1]]
    
    # Vind de meest waarschijnlijke positie van elk spook
    mostLikelyPositions = [dist.argMax() for dist in livingGhostPositionDistributions]
    
    # Als er geen levende spoken zijn, return een willekeurige legale actie
    if not mostLikelyPositions:
        return random.choice(legal)
    
    # Vind het dichtstbijzijnde spook
    closestGhostDistance = float('inf')
    closestGhostPosition = None
    
    for ghostPos in mostLikelyPositions:
        distance = self.distancer.getDistance(pacmanPosition, ghostPos)
        if distance < closestGhostDistance:
            closestGhostDistance = distance
            closestGhostPosition = ghostPos
    
    # Kies de actie die de afstand tot het dichtstbijzijnde spook minimaliseert
    bestAction = None
    minDistance = float('inf')
    
    for action in legal:
        successorPosition = Actions.getSuccessor(pacmanPosition, action)
        distance = self.distancer.getDistance(successorPosition, closestGhostPosition)
        if distance < minDistance:
            minDistance = distance
            bestAction = action
    
    return bestAction
```

### Resultaten
Alle tests voor q8 zijn geslaagd zonder inferentiefouten. De agent won 10 van de 10 spellen met een gemiddelde score van 763,3, wat ruim boven de vereiste 700 is. De opdracht kreeg een score van 1/1.

## Opdracht 5: Benaderende Inferentie Initialisatie en Beliefs (q9)

### Probleem
In deze opdracht moest ik de functies `initializeUniformly` en `getBeliefDistribution` in de `ParticleFilter` klasse implementeren voor het particle filtering algoritme.

### Aanpak en Implementatie

#### initializeUniformly methode
Voor de `initializeUniformly` methode moest ik deeltjes gelijkmatig (niet willekeurig) verdelen over legale posities:

```python
def initializeUniformly(self, gameState):
    self.particles = []
    
    # Verdeel deeltjes gelijkmatig over legale posities
    numPositions = len(self.legalPositions)
    for i in range(self.numParticles):
        # Gebruik modulo om door posities te lopen
        position = self.legalPositions[i % numPositions]
        self.particles.append(position)
```

#### getBeliefDistribution methode
Voor de `getBeliefDistribution` methode moest ik de lijst van deeltjes omzetten in een `DiscreteDistribution` object:

```python
def getBeliefDistribution(self):
    beliefs = DiscreteDistribution()
    
    # Tel voorkomens van elke positie in de deeltjeslijst
    for particle in self.particles:
        beliefs[particle] += 1.0
    
    # Normaliseer de distributie
    beliefs.normalize()
    return beliefs
```

### Resultaten
Alle tests voor q9 zijn geslaagd zonder inferentiefouten. De autograder bevestigt dat mijn implementaties van de `initializeUniformly` en `getBeliefDistribution` methoden correct zijn. De opdracht kreeg een score van 1/1.

## Opdracht 6: Benaderende Inferentie Observatie (q10)

### Probleem
In deze opdracht moest ik de `observeUpdate` methode in de `ParticleFilter` klasse implementeren om de deeltjes bij te werken op basis van een observatie.

### Aanpak en Implementatie
Voor deze opdracht heb ik de volgende stappen geïmplementeerd:
1. Maak een gewichtsdistributie over deeltjes aan
2. Bereken het gewicht voor elk deeltje op basis van de observatie
3. Controleer of alle deeltjes een gewicht van nul hebben (in dat geval, herinitialiseer de deeltjes)
4. Normaliseer de gewichten
5. Neem een nieuwe steekproef van deeltjes op basis van hun gewichten:

```python
def observeUpdate(self, observation, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    jailPosition = self.getJailPosition()
    
    # Maak een gewichtsdistributie over deeltjes aan
    weights = DiscreteDistribution()
    
    # Bereken het gewicht voor elk deeltje
    for particle in self.particles:
        # Gewicht is de waarschijnlijkheid van de observatie gegeven Pacman's positie en de deeltjeslocatie
        weight = self.getObservationProb(observation, pacmanPosition, particle, jailPosition)
        weights[particle] += weight
    
    # Controleer of alle deeltjes een gewicht van nul hebben
    if weights.total() == 0:
        # Herinitialiseer deeltjes gelijkmatig
        self.initializeUniformly(gameState)
        return
    
    # Normaliseer de gewichten
    weights.normalize()
    
    # Neem een nieuwe steekproef van deeltjes op basis van hun gewichten
    self.particles = []
    for _ in range(self.numParticles):
        self.particles.append(weights.sample())
```

### Resultaten
Alle tests voor q10 zijn geslaagd zonder inferentiefouten. De agent won 10 van de 10 spellen met een gemiddelde score van 186,0. De opdracht kreeg een score van 1/1.

## Opdracht 7: Benaderende Inferentie met Tijdsverloop (q11)

### Probleem
In deze opdracht moest ik de `elapseTime` methode in de `ParticleFilter` klasse implementeren om de deeltjes bij te werken nadat er tijd is verstreken.

### Aanpak en Implementatie
Voor deze opdracht heb ik de volgende stappen geïmplementeerd:
1. Maak een nieuwe lijst aan om bijgewerkte deeltjes op te slaan
2. Voor elk deeltje:
   - Krijg de distributie over nieuwe posities voor het spook
   - Neem een steekproef van een nieuwe positie uit die distributie
   - Voeg de nieuwe positie toe aan de nieuwe deeltjeslijst
3. Werk de deeltjeslijst bij:

```python
def elapseTime(self, gameState):
    newParticles = []
    
    # Voor elk deeltje
    for oldPos in self.particles:
        # Krijg de distributie over nieuwe posities voor het spook
        newPosDist = self.getPositionDistribution(gameState, oldPos)
        
        # Neem een steekproef van een nieuwe positie uit de distributie
        newPos = newPosDist.sample()
        
        # Voeg de nieuwe positie toe aan de nieuwe deeltjeslijst
        newParticles.append(newPos)
    
    # Werk de deeltjeslijst bij
    self.particles = newParticles
```

### Resultaten
Alle tests voor q11 zijn geslaagd zonder inferentiefouten. De agent won 5 van de 5 spellen met een gemiddelde score van 366,2. De opdracht kreeg een score van 1/1.

## Conclusie

In dit project heb ik met succes verschillende inferentie-algoritmen geïmplementeerd voor het volgen van spoken in het Pacman-spel. Ik heb zowel exacte inferentie (Forward Algorithm) als benaderende inferentie (Particle Filtering) geïmplementeerd en getest.

De belangrijkste concepten die ik heb geleerd en toegepast zijn:
- Het gebruik van Bayes' regel voor het bijwerken van beliefs op basis van observaties
- Het gebruik van de totale waarschijnlijkheid voor het voorspellen van beliefs na tijdsverloop
- Het implementeren van particle filtering als een efficiënte benadering voor inferentie
- Het gebruik van gewogen steekproeven voor het bijwerken van deeltjes op basis van observaties

Alle opdrachten zijn met succes voltooid en alle tests zijn geslaagd. De implementaties zijn efficiënt en effectief in het volgen van spoken en het helpen van Pacman om ze te vangen.