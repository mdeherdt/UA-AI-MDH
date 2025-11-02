import random
import copy
from copy import deepcopy
import math

from typing import Set, Dict, List, TypeVar, Optional
from abc import ABC, abstractmethod

from util import monitor


Value = TypeVar('Value')


class Variable(ABC):
    @property
    @abstractmethod
    def startDomain(self) -> Set[Value]:
        """ Returns the set of initial values of this variable (not taking constraints into account). """
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


class CSP(ABC):
    def __init__(self, MRV=True, LCV=True):
        self.MRV = MRV
        self.LCV = LCV

    @property
    @abstractmethod
    def variables(self) -> Set[Variable]:
        """ Return the set of variables in this CSP.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def remainingVariables(self, assignment: Dict[Variable, Value]) -> Set[Variable]:
        """ Returns the variables not yet assigned. """
        return self.variables.difference(assignment.keys())

    @abstractmethod
    def neighbors(self, var: Variable) -> Set[Variable]:
        """ Return all variables related to var by some constraint.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def assignmentToStr(self, assignment: Dict[Variable, Value]) -> str:
        """ Formats the assignment of variables for this CSP into a string. """
        s = ""
        for var, val in assignment.items():
            s += f"{var} = {val}\n"
        return s

    def isComplete(self, assignment: Dict[Variable, Value]) -> bool:
        """ Return whether the assignment covers all variables.
            :param assignment: dict (Variable -> value)
        """
        # TODO: Implement CSP::isComplete (problem 1)

        return len(assignment) == len(self.variables)

    @abstractmethod
    def isValidPairwise(self, var1: Variable, val1: Value, var2: Variable, val2: Value) -> bool:
        """ Return whether this pairwise assignment is valid with the constraints of the csp.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def isValid(self, assignment: Dict[Variable, Value]) -> bool:
        """ Return whether the assignment is valid (i.e. is not in conflict with any constraints).
            You only need to take binary constraints into account.
            Hint: use `CSP::neighbors` and `CSP::isValidPairwise` to check that all binary constraints are satisfied.
            Note that constraints are symmetrical, so you don't need to check them in both directions.
        """
        # TODO: Implement CSP::isValid (problem 1)

        assigned_variables_list = list(assignment.keys())

        num_assigned = len(assigned_variables_list)

        for i in range(num_assigned):
            var1 = assigned_variables_list[i]
            val1 = assignment[var1]
            for j in range(i+1,num_assigned):
                var2 = assigned_variables_list[j]
                val2 = assignment[var2]

                if var2 in self.neighbors(var1):
                    if not self.isValidPairwise(var1,val1,var2,val2):
                        return False

        return True


    def solveBruteForce(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with brute force technique.
            Initializes the domains and calls `CSP::_solveBruteForce`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        return self._solveBruteForce(initialAssignment, domains)

    @monitor
    def _solveBruteForce(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm to brute force this CSP.
            Use `CSP::isComplete`, `CSP::isValid`, `CSP::selectVariable` and `CSP::orderDomain`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        # TODO: Implement CSP::_solveBruteForce (problem 1)

        if self.isComplete(assignment):
            return assignment

        var = self.selectVariable(assignment,domains)

        for dom in self.orderDomain(assignment,domains,var):
            assignment[var] = dom
            if self.isValid(assignment):
                result = self._solveBruteForce(assignment,domains)
                if result is not None:
                    return result
            del assignment[var]
        return None



    def solveForwardChecking(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with forward checking.
            Initializes the domains and calls `CSP::_solveForwardChecking`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.forwardChecking(initialAssignment, domains, var)
        return self._solveForwardChecking(initialAssignment, domains)

    @monitor
    def _solveForwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm with forward checking.
            Use `CSP::forwardChecking` and you should no longer need to check if an assignment is valid.
            :return: a complete and valid assignment if one exists, None otherwise.
        """

        if self.isComplete(assignment):
            return assignment



        var = self.selectVariable(assignment,domains)

        for val in self.orderDomain(assignment,domains,var):
            assignment[var] = val
            new_domains = self.forwardChecking(assignment,domains,var)
            is_dead_end = any(len(new_domains[v]) == 0 for v in self.remainingVariables(assignment))

            if not is_dead_end:
                result = self._solveForwardChecking(assignment,new_domains)
                if result is not None:
                    return result

            del assignment[var]
        return None






    def forwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], variable: Variable) -> Dict[Variable, Set[Value]]:
        """ Implement the forward checking algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains after enforcing all constraints.
        """
        # TODO: Implement CSP::forwardChecking (problem 2)

        current_val = assignment[variable]

        new_domains = copy.deepcopy(domains)

        for neighbour in self.neighbors(variable):
            if neighbour not in assignment:
                for val in new_domains[neighbour].copy():
                    if not self.isValidPairwise(variable,current_val,neighbour,val):
                        new_domains[neighbour].remove(val)

        return new_domains



    def selectVariable(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Variable:
        """ Implement a strategy to select the next variable to assign. """
        if not self.MRV:
            return random.choice(list(self.remainingVariables(assignment)))

        # TODO: Implement CSP::selectVariable (problem 2)
        min = math.inf
        best_var = None

        for var in self.remainingVariables(assignment):
            current_size = len(domains[var])
            if current_size < min:
                best_var = var
                min = current_size
        return best_var




    def orderDomain(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], var: Variable) -> List[Value]:
        """ Implement a smart ordering of the domain values. """
        if not self.LCV:
            return list(domains[var])

        # TODO: Implement CSP::orderDomain (problem 2)

        score_list = []

        for val in domains[var]:
            score = 0
            for neighbour in self.neighbors(var):
                if neighbour in assignment:
                    continue
                for neighbour_val in domains[neighbour]:
                    if not self.isValidPairwise(var,val,neighbour,neighbour_val):
                        score += 1

            score_list.append((score,val))
        sorted_list = sorted(score_list)

        var_list = [v[1] for v in sorted_list]

        return var_list


    def solveAC3(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with AC3.
            Initializes domains and calls `CSP::_solveAC3`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.ac3(initialAssignment, domains, var)
        return self._solveAC3(initialAssignment, domains)

    @monitor
    def _solveAC3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[Dict[Variable, Value]]:
        """
            Implement the actual backtracking algorithm with AC3.
            Use `CSP::ac3`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        if self.isComplete(assignment):
            return assignment

        var = self.selectVariable(assignment,domains)

        for value in self. orderDomain(assignment,domains,var):
            assignment[var] = value

            new_domains = self.ac3(assignment,domains,var)

            is_dead_end = any(len(new_domains[v]) == 0 for v in self.remainingVariables(assignment))

            if not is_dead_end:
                result = self._solveAC3(assignment,new_domains)
                return result

            del assignment[var]

        return None

    def ac3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], variable: Variable) -> Dict[Variable, Set[Value]]:
        """ Implement the AC3 algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains ensuring arc consistency.
        """
        new_domains = copy.deepcopy(domains)

        queue = []

        for neighbor in self.neighbors(variable):
            queue.append((neighbor,variable))


        while queue:
            (xi,xj) = queue.pop(0)

            revised = False

            for val_i in new_domains[xi].copy():

                is_supported = False

                for val_j in new_domains[xj]:
                    if self.isValidPairwise(xi,val_i,xj,val_j):
                        is_supported = True
                        break

                if not is_supported:
                    new_domains[xi].remove(val_i)
                    revised = True


            if revised:
                for xk in self.neighbors(xi):
                    if xk != xj and xk not in assignment:
                        queue.append((xk,xi))

        return new_domains


def domainsFromAssignment(assignment: Dict[Variable, Value], variables: Set[Variable]) -> Dict[Variable, Set[Value]]:
    """ Fills in the initial domains for each variable.
        Already assigned variables only contain the given value in their domain.
    """
    domains = {v: v.startDomain for v in variables}
    for var, val in assignment.items():
        domains[var] = {val}
    return domains
