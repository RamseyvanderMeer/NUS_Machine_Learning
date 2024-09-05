from collections import deque
import math
from heapq import heappop, heappush
# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "ucs": ucs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
    }.get(searchMethod)(maze)


class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position



def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    # Create start and end node
    start_node = Node(None, maze.getStart())
    queue = [start_node]
    visited = set()
    visited.add(start_node.position)
    while queue:
        current_node = queue.pop(0)
        if maze.isObjective(current_node.position[0], current_node.position[1]):
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        else:
            for pos in maze.getNeighbors(current_node.position[0], current_node.position[1]):
                if pos not in visited:
                    visited.add(pos)
                    new_node = Node(current_node, pos)
                    queue.append(new_node)
    return []



def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_node = Node(None, maze.getStart())
    stack = deque([start_node])
    visited = set()
    visited.add(start_node.position)
    while stack:
        current_node = stack.pop()
        if maze.isObjective(current_node.position[0], current_node.position[1]):
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        else:
            for pos in maze.getNeighbors(current_node.position[0], current_node.position[1]):
                if pos not in visited:
                    visited.add(pos)
                    new_node = Node(current_node, pos)
                    stack.append(new_node)
    return []

def ucs(maze):
    """
    Runs ucs for part 1 of the assignment. aka dikstras

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    node = Node(None, maze.getStart())
    node.g = 0
    frontier = [node]
    explored = set()
    explored.add(node.position)
    while frontier:
        current_node = frontier.pop(0)
        if maze.isObjective(current_node.position[0], current_node.position[1]):
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        else:
            for pos in maze.getNeighbors(current_node.position[0], current_node.position[1]):
                if pos not in explored:
                    explored.add(pos)
                    new_node = Node(current_node, pos)
                    new_node.g = current_node.g + 1
                    frontier.append(new_node)
                    frontier.sort(key=lambda x: x.g)
    return []



def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    start_node = Node(None, maze.getStart())
    start_node.g = 0
    start_node.h = math.sqrt((maze.getStart()[0] - maze.getObjectives()[0][0])**2 + (maze.getStart()[1] - maze.getObjectives()[0][1])**2)
    start_node.f = start_node.g + start_node.h
    frontier = [start_node]
    explored = set()
    explored.add(start_node.position)
    while frontier:
        current_node = frontier.pop(0)
        if maze.isObjective(current_node.position[0], current_node.position[1]):
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        else:
            for pos in maze.getNeighbors(current_node.position[0], current_node.position[1]):
                if pos not in explored:
                    explored.add(pos)
                    new_node = Node(current_node, pos)
                    new_node.g = current_node.g + 1
                    new_node.h = math.sqrt((pos[0] - maze.getObjectives()[0][0])**2 + (pos[1] - maze.getObjectives()[0][1])**2)
                    new_node.f = new_node.g + new_node.h
                    frontier.append(new_node)
                    frontier.sort(key=lambda x: x.f)
    return []



def huristic(pos, objectives):
    return min([math.sqrt((pos[0] - obj[0])**2 + (pos[1] - obj[1])**2) for obj in objectives])

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # write an astar function that takes in a maze and returns a path
    # there are multiple objectives
    # we need to navigate to each corner of the maze

    start_node = Node(None, maze.getStart())
    start_node.g = list(maze.getObjectives())
    start_node.h = huristic(maze.getStart(), maze.getObjectives())
    start_node.f = set()
    frontier = [start_node]

    while frontier:
        current_node = frontier.pop(0)

        if current_node.position in current_node.g:
            current_node.g.remove(current_node.position)
            current_node.f = set()
            print(current_node.g)

            if not current_node.g:
                path = []
                while current_node is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

        current_node.f.add(current_node.position)

        for pos in maze.getNeighbors(current_node.position[0], current_node.position[1]):
            if pos in current_node.f:
                continue


            if pos not in current_node.f and all(frontier_node.position != pos for frontier_node in frontier):
                new_node = Node(current_node, pos)
                new_node.g = list(current_node.g)
                new_node.h = huristic(pos, new_node.g) + len(new_node.g) + current_node.h
                new_node.f = set(current_node.f)
                frontier.append(new_node)


        frontier.sort(key=lambda x: x.h)

    return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start_node = Node(None, maze.getStart())
    start_node.g = list(maze.getObjectives())
    start_node.h = huristic(maze.getStart(), maze.getObjectives())
    start_node.f = set()
    frontier = [start_node]

    while frontier:
        current_node = frontier.pop(0)

        if current_node.position in current_node.g:
            current_node.g.remove(current_node.position)
            current_node.f = set()
            print(current_node.g)

            if not current_node.g:
                path = []
                while current_node is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

        current_node.f.add(current_node.position)

        for pos in maze.getNeighbors(current_node.position[0], current_node.position[1]):
            if pos in current_node.f:
                continue


            if pos not in current_node.f and all(frontier_node.position != pos for frontier_node in frontier):
                new_node = Node(current_node, pos)
                new_node.g = list(current_node.g)
                new_node.h = huristic(pos, new_node.g) + len(new_node.g) + current_node.h
                new_node.f = set(current_node.f)
                frontier.append(new_node)


        frontier.sort(key=lambda x: x.h)

    return []