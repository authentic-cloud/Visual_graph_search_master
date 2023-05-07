import pygame

from solver import Environment

class Board(Environment):
    """Board class

    Subclass of environment with application specific methods.
    Every state corresponds to the current coordinates of the agent.

    Attributes:
        source: Starting state of the system.
        target: Goal state of the system.
        explored (set): Set of explored states.

        screen: PyGame display.
        origin (tuple): Coordinates for the top left corner of the board.
        size (tuple): Pixel size for the board.
        rows: Row quantity.
        cols: Column quantity.

        cells: Array with all the cell objects in the board.
        walls (set): Set with all the wall cells.
        path (list): List of steps from source to target cell.
    """
    def __init__(self, screen, origin, size, rows = 8, cols = 8):
        self.screen = screen
        self.origin = origin
        self.size = size

        self.rows = rows
        self.cols = cols

        self.cells = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):

                cell = Cell(self, (i, j))
            
                row.append(cell)
            self.cells.append(row)

        self.reset()

    def reset(self):
        """Reset board

        Resets the board to its initial state. With no source nor target and
        without walls.

        Also cleans all explored and path cells.
        """
        self.walls = set()
        self.source = None
        self.target = None
        self.clean()

    def clean(self):
        """Clean board

        Resets explored set, path and updates drawing. It does not modify
        selected source and target cells and walls.
        """
        self.explored = set()
        self.path = []
        self.draw()
        pygame.display.flip()

    def get_actions(self, state):
        """Get actions from environment

        Returns a list of possible actions in the given state.
        Each action is represented with a string value (up, down, left, right).

        Parameters:
            state: Current state (agent coordinates).

        Returns:
            actions (list): A list of actions that can be taken in a state.
        """
        actions = set()

        if state[0] > 0:
            actions.add('up')
        if state[0] < self.rows - 1:
            actions.add('down')
        if state[1] > 0:
            actions.add('left')
        if state[1] < self.cols - 1:
            actions.add('right')

        return actions

    def transition_model(self, state, action):
        """Transition model

        Returns the new state resulting from performing given action in the
        current state. If there is no wall in the direction indicated by action,
        updates the display with the new position. Otherwise returns the same
        state.

        Parameters:
            state: Current state (agent coordinates).
            action: Action to be taken.

        Returns:
            new_state: State resulting from performing the action.
        """
        if action == 'up':
            i, j = state[0]-1, state[1]
        if action == 'down':
            i, j = state[0]+1, state[1]
        if action == 'left':
            i, j = state[0], state[1]-1
        if action == 'right':
            i, j = state[0], state[1]+1

        if (i, j) not in self.walls:
            self.cells[i][j].draw(Cell.ACTIVE)
            pygame.display.flip()

            return i, j
        else:
            return state

    def cost_to_target(self, cell):
        """Estimate cost to target

        Get estimated cost to reach target state from current state.
        As every state is a pair of coordinates we calculate the Manhattan's 
        distance from the current position to the target.

        Parameters:
            state: Current state.

        Returns:
            cost (int): integer representing the estimated cost.
        """
        return (abs(cell[0] - self.target[0]) + abs(cell[1] - self.target[1]))

    def draw(self):
        """Draw board

        Update the board in the display.
        """
        for row in self.cells:
            for cell in row:

                if cell.position == self.source:
                    cell.draw(Cell.SOURCE)
                elif cell.position == self.target:
                    cell.draw(Cell.TARGET)
                elif self.path is not None and cell.position in [item[0] for item in self.path]:
                    # If cell is part of the path found
                    cell.draw(Cell.PATH)
                elif cell.position in self.explored:
                    cell.draw(Cell.EXPLORED)
                elif cell.position in self.walls:
                    cell.draw(Cell.WALL)
                else:
                    cell.draw()
                
        
class Cell:
    """Cell class

    Represents a cell in the board.
    """
    EMPTY, WALL, PATH, EXPLORED, ACTIVE, SOURCE, TARGET = range(7)

    def __init__(self, board, position):
        
        self.board = board
        self.position = self.i, self.j = position

        self.size = int(min(board.size[0] / board.cols, board.size[1] / board.rows))
        self.coord = (board.origin[0] + self.j * self.size, board.origin[1] + self.i * self.size)

        OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
        self.font = pygame.font.Font(OPEN_SANS, int(self.size * 0.7))

    def draw(self, style = EMPTY):

        self.rect = pygame.Rect(
            self.coord[0],
            self.coord[1],
            self.size, self.size
        )

        if style == self.EMPTY:
            color = (0, 0, 0)
            text = None

        elif style == self.SOURCE:
            color = (255, 0, 0)
            text = 'A'

        elif style == self.TARGET:
            color = (0, 255, 0)
            text = 'B'

        elif style == self.WALL:
            color = (64, 64, 64)
            text = None
        
        elif style == self.PATH:
            color = (255, 255, 0)
            text = None

        elif style == self.EXPLORED:
            color = (128, 128, 128)
            text = None

        elif style == self.ACTIVE:
            color = (128, 128, 64)
            text = None
        
        pygame.draw.rect(self.board.screen, color, self.rect)
        pygame.draw.rect(self.board.screen, (255, 255, 255), self.rect, 1)

        if text is not None:
            text = self.font.render(str(text), True, (0, 0, 0))
            font_rect = text.get_rect()
            font_rect.center = self.rect.center
            self.board.screen.blit(text, font_rect)