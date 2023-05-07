import abc

"""ABC - Abstract Base Class - In Python, abstract base classes provide a blueprint for concrete classes. 
They don't contain implementation. Instead, they provide an interface and make sure that derived concrete classes are properly implementation.
"""

class Environment(abc.ABC):
    """Environment Class-Basically inotializing the environment-
    Here we have a abstract class that help other subclasses for implementation
    Atrr- "source , target , explored(set)"
    """
    source = None
    target = None
    explored = set()

    def goal_test(self, state) -> bool:
        """Goal test -Checking the given state is a goal state or not.
        """
        return state == self.target


    @abc.abstractmethod
    """  A decorator indicating abstract method.
        """
    def get_actions(self, state) -> list:
        """ Getting some action from environment
        it will take Param- Current state
        Return-Action in the form of list - A list of action that can be taken in a state.
        """
        ...

    @abc.abstractmethod
    """  A decorator indicating abstract method"""
    def transition_model(self, state, action):
        """Transition model
        It will take params-
            state: Current state.
            action: Action to be taken.
        Returns- new_state: State resulting from performing the action.
        """
        ...

    def cost_to_target(self, state) -> int:

        """Estimation of cost or time in terms of implementing that method.
        This will provide cost to reach the target state in terms of an integer.
            Using Admissible heuristic-
            In computer science, specifically in algorithms related to pathfinding, a heuristic function is said to be admissible if it never overestimates the cost of reaching the goal,
            i.e. the cost it estimates to reach the goal is not higher than the lowest possible cost from the current point in the path
            f(n)=g(n)+h(n),where
            f(n) = the evaluation function.
            g(n) = the cost from the start node to the current node
            h(n) = estimated cost from current node to goal.

        """
        raise NotImplementedError
        #marker -To-do or something like that

class Solver:
    """
       A generic solver that implements various algorithms to perform a search.
       It requires an environment that must be subclassed from the base in this file.
       Attributes:
           environment (Environment): Current application environment subclassed.
       """
    DFS, BFS, GREEDY_BFS, A_STAR = range(4)

    def __init__(self, environment):
        self.environment = environment

    def search_path(self, algorithm = A_STAR):
        """Search path from source to target
                It returns list of actions that connects source to target using the selected algorithm.
                Parameters:
                    algorithm: Algorithm to be used to perform the search.
                Returns:
                    path (list): List of actions in the form of (state, action).
                    None: If there is no possible path.
                """
        # The algorithm selected determines the type of frontier to be used
        frontier = StackFrontier() if algorithm == self.DFS else \
                   QueueFrontier() if algorithm == self.BFS else \
                   GreedyFrontier(lambda node: self.environment.cost_to_target(node.state)) if algorithm == self.GREEDY_BFS else \
                   GreedyFrontier(lambda node: self.environment.cost_to_target(node.state) + node.cost_from_source)

        # Initialize frontier with just the starting position
        start = Node(state=self.environment.source, parent=None, action=None)
        frontier.add(start)

        # Initialize an empty explored set
        self.environment.explored = set()

        # Keep looping until solution found
        while True:

            # If there is nothing left in the frontier, then there is no solution
            if frontier.empty():
                return None

            # Choose a node from the frontier (implementation changes according to the type of frontier chosen)
            node = frontier.remove()

            # Process every action that can be taken from current state
            for action in self.environment.get_actions(node.state):

                # Get state resulting from performing given action
                state = self.environment.transition_model(node.state, action)

                if not frontier.contains_state(state) and state not in self.environment.explored:
                    child = Node(state=state, parent=node, action=action)

                    # If child is the goal, then there is a solution
                    if self.environment.goal_test(child.state):
                        # Returns a list of (state, action) to be taken to reach the target
                        path = []

                        while child.parent is not None:
                            path.append((child.state, child.action))
                            child = child.parent

                        path.reverse()
                        return path
                    else:
                        frontier.add(child)

            # Mark node as explored
            self.environment.explored.add(node.state)


class Node:
    """Minimal data structure.

    Attributes:
        state: Node state.
        parent (Node): Node that generated this node.
        action: Action applied to parent to get node.
        cost_from_source (int):  a pathcost from initial state to node.
    """
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

        # Calculates the cost to reach this node from the source node
        if parent is None:
            self.cost_from_source = 0
        else:
            self.cost_from_source = parent.cost_from_source + 1

    def __str__(self):
        return f"state: {self.state}, action: {self.action}, cost: {self.cost_from_source}"

    def __repr__(self):
        return self.__str__()


class StackFrontier:
    """Stack Frontier (LIFO) m
    Frontier class that always expands the deepest node.
    The deepest node is the last one that was added.
    
    Used in: 
        depth-first search algorithm (DFS)

    Attributes:
        list (list): A list of all nodes in the frontier.
    """
    def __init__(self):
        self.list = []

    def __str__(self):
        return str(self.list)

    def add(self, node):
        """Add node

        Parameters:
            node (Node): Node to be added.
        """
        self.list.append(node)

    def contains_state(self, state):
        """Contains state

        Parameters:
            state: State to be checked.

        Returns:
            True: If state is in the frontier.
            False: If state is not in the frontier.
        """
        return any(node.state == state for node in self.list)

    def empty(self):
        """Is empty

        Returns:
            True: If there are no nodes in the frontier.
            False: If there are nodes in the frontier.
        """
        return len(self.list) == 0

    def remove(self):
        """Remove node

        Returns:
            node (Node): The deepest node (last in the list).
        """
        if self.empty():
            raise Exception("Empty frontier")
        else:
            node = self.list[-1]
            self.list = self.list[:-1]
            return node      


class QueueFrontier(StackFrontier):
    """Queue Frontier (FIFO)

    Frontier class that always expands the shallowest node.
    The shallowest node is the first one that was added to the list.
    
    Used in:
        breath-first search algorithm (BFS)

    Attributes:
        list (list): A list of all nodes in the frontier.
    """
    def remove(self):
        """Remove node

        Returns:
            node (Node): The shallowest node (first in the list).
        """
        if self.empty():
            raise Exception("Empty frontier")
        else:
            node = self.list[0]
            self.list = self.list[1:]
            return node


class GreedyFrontier(StackFrontier):
    """Greedy Frontier

    Frontier class that always expands the node that is closest to the goal,
    as estimated by a heuristic function (cost_function).
    It does so by adding them in order using the cost function.

    Used in:
        greedy best-first search algorithm (GREEDY BFS)
        A* algorithm (A STAR)

    Attributes:
        list (list): A list of all nodes in the frontier.
        cost_function (function): Method for calculating the total cost of each node.
    """
    def __init__(self, cost_function):
        super().__init__()
        self.cost_function = cost_function

    def add(self, new):
        """Add node

        Insert a node to the list ordered by the cost function.
        The lower the cost, the higher in the list.

        Parameters:
            new (Node): Node to be added.
        """
        position = 0

        for node in self.list:
            if self.cost_function(new) <= self.cost_function(node):
                position += 1
                continue
            else:
                break

        self.list.insert(position, new)
