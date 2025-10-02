# Setup & Autograder Tutorial

## Setup

Make sure you have Python 3.8+ installed.  
You can check dependencies with:

```bash
python autograder.py --check-dependencies
```

If everything is installed correctly, you should see a spinning line.

---

## Autograding

To get familiar with the autograder, we will code, test, and run it on three small questions.

In this folder you should see:

```
addition.py
autograder.py
buyLotsOfFruit.py
grading.py
projectParams.py
shop.py
shopSmart.py
testClasses.py
testParser.py
test_cases/
tutorialTestClasses.py
```

- **Files you edit or run:**  
  - `addition.py` (Q1)  
  - `buyLotsOfFruit.py` (Q2)  
  - `shop.py` + `shopSmart.py` (Q3)  
  - `autograder.py`

- **Files you don’t need to touch:**  
  -  `grading.py`, `testClasses.py`, `testParser.py`, `tutorialTestClasses.py`, `projectParams.py`, `test_cases/`

Run all tests:

```bash
python autograder.py
```

Run only one question:

```bash
python autograder.py -q q1
```

---

## Exercises

### Q1: Addition

Edit `addition.py` and implement:

```python
def add(a, b):
    "Return the sum of a and b"
    "*** YOUR CODE HERE ***"
    return 0
```

Test:

```bash
python autograder.py -q q1
```

---

### Q2: buyLotsOfFruit

Edit `buyLotsOfFruit.py` and implement:

```python
def buyLotsOfFruit(orderList):
    # orderList: List of (fruit, numPounds)
    # Returns cost of order, or None if fruit not found.
    pass
```

Test:

```bash
python autograder.py -q q2
```

---

### Q3: shopSmart

Edit `shopSmart.py` and implement:

```python
def shopSmart(orderList, fruitShops):
    # orderList: List of (fruit, numPounds)
    # fruitShops: List of FruitShop objects
    # Return the shop with the lowest total order cost.
    pass
```

Test:

```bash
python autograder.py -q q3
```
---
These exercises are heavily based on the projects from [Introduction to Artificial Intelligence at UC Berkeley](https://ai.berkeley.edu/home).
