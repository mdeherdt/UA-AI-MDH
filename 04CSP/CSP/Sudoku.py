from typing import Set, Dict

from CSP import CSP, Variable, Value


class Sudoku(CSP):
    def __init__(self, MRV=True, LCV=True):
        super().__init__(MRV=MRV, LCV=LCV)
        # TODO: Implement Sudoku::__init__ (problem 4)
        grid  = []
        for i in range(9):
            grid.append([Cell(i,j,0) for j in range(9)])
        self.grid = grid

    @property
    def variables(self) -> Set['Cell']:
        """ Return the set of variables in this CSP. """
        # TODO: Implement Sudoku::variables (problem 4)
        vars = set()
        for i in range(9):
            for j in range(9):
                vars.add(self.grid[i][j])
        return vars

    def getCell(self, x: int, y: int) -> 'Cell':
        """ Get the  variable corresponding to the cell on (x, y) """
        return self.grid[y][x]

    def neighbors(self, var: 'Cell') -> Set['Cell']:
        """ Return all variables related to var by some constraint. """
        neighbors_set = set()

        neighbors_set.update(self.grid[var.row])

        for i in range(9):
            neighbors_set.add(self.grid[i][var.col])

        start_row = (var.row //3) * 3
        start_col = (var.col // 3) * 3

        for r in range(start_row,start_row+3):
            for c in range(start_col,start_col+3):
                neighbors_set.add(self.grid[r][c])

        neighbors_set.remove(var)

        return neighbors_set

    def isValidPairwise(self, var1: 'Cell', val1: Value, var2: 'Cell', val2: Value) -> bool:
        """ Return whether this pairwise assignment is valid with the constraints of the csp. """
        # TODO: Implement Sudoku::isValidPairwise (problem 4)

        if val1 == val2:
            return False
        return True



    def assignmentToStr(self, assignment: Dict['Cell', Value]) -> str:
        """ Formats the assignment of variables for this CSP into a string. """
        s = ""
        for y in range(9):
            if y != 0 and y % 3 == 0:
                s += "---+---+---\n"
            for x in range(9):
                if x != 0 and x % 3 == 0:
                    s += '|'

                cell = self.getCell(x, y)
                s += str(assignment.get(cell, ' '))
            s += "\n"
        return s

    def parseAssignment(self, path: str) -> Dict['Cell', Value]:
        """ Gives an initial assignment for a Sudoku board from file. """
        initialAssignment = dict()

        with open(path, "r") as file:
            for y, line in enumerate(file.readlines()):
                if line.isspace():
                    continue
                assert y < 9, "Too many rows in sudoku"

                for x, char in enumerate(line):
                    if char.isspace():
                        continue

                    assert x < 9, "Too many columns in sudoku"

                    var = self.getCell(x, y)
                    val = int(char)

                    if val == 0:
                        continue

                    assert val > 0 and val < 10, f"Impossible value in grid"
                    initialAssignment[var] = val
        return initialAssignment


class Cell(Variable):
    def __init__(self,row = None,col = None, value = 0):
        super().__init__()
        # TODO: Implement Cell::__init__ (problem 4)
        # You can add parameters as well.
        self.row = row
        self.col = col
        self.value = value

    @property
    def startDomain(self) -> Set[Value]:
        """ Returns the set of initial values of this variable (not taking constraints into account). """
        # TODO: Implement Cell::startDomain (problem 4)
        if self.hasValue():
            return {int(self.value)}

        return {v for v in range(1, 10)}

    def hasValue(self):
        return self.value != 0


