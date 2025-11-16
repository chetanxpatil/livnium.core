"""
Search Engine: Tree Expansion and State Space Search

Implements search strategies for problem solving.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq


class SearchStrategy(Enum):
    """Search strategies."""
    BFS = "breadth_first"      # Breadth-first search
    DFS = "depth_first"        # Depth-first search
    A_STAR = "a_star"          # A* search
    BEAM = "beam"              # Beam search
    GREEDY = "greedy"          # Greedy best-first


@dataclass
class SearchNode:
    """
    Node in search tree.
    
    Represents a state in the search space.
    """
    state: Any
    parent: Optional['SearchNode'] = None
    action: Optional[str] = None
    cost: float = 0.0
    heuristic: float = 0.0
    depth: int = 0
    children: List['SearchNode'] = field(default_factory=list)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)
    
    def get_path(self) -> List['SearchNode']:
        """Get path from root to this node."""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class SearchEngine:
    """
    Search engine for Livnium Core System.
    
    Implements various search strategies for problem solving.
    """
    
    def __init__(self, 
                 initial_state: Any,
                 goal_test: Callable[[Any], bool],
                 successors: Callable[[Any], List[Tuple[str, Any]]],
                 heuristic: Optional[Callable[[Any], float]] = None):
        """
        Initialize search engine.
        
        Args:
            initial_state: Initial state
            goal_test: Function that returns True if state is goal
            successors: Function that returns list of (action, new_state) tuples
            heuristic: Optional heuristic function for A*
        """
        self.initial_state = initial_state
        self.goal_test = goal_test
        self.successors = successors
        self.heuristic = heuristic or (lambda s: 0.0)
        
        self.visited: set = set()
        self.nodes_expanded = 0
        self.max_depth = 0
    
    def search(self, strategy: SearchStrategy = SearchStrategy.BFS,
              max_depth: int = 100,
              beam_width: int = 10) -> Optional[SearchNode]:
        """
        Perform search.
        
        Args:
            strategy: Search strategy
            max_depth: Maximum search depth
            beam_width: Beam width for beam search
            
        Returns:
            Goal node if found, None otherwise
        """
        self.visited.clear()
        self.nodes_expanded = 0
        self.max_depth = 0
        
        if strategy == SearchStrategy.BFS:
            return self._bfs(max_depth)
        elif strategy == SearchStrategy.DFS:
            return self._dfs(max_depth)
        elif strategy == SearchStrategy.A_STAR:
            return self._a_star(max_depth)
        elif strategy == SearchStrategy.BEAM:
            return self._beam_search(max_depth, beam_width)
        elif strategy == SearchStrategy.GREEDY:
            return self._greedy(max_depth)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _bfs(self, max_depth: int) -> Optional[SearchNode]:
        """Breadth-first search."""
        queue = deque([SearchNode(self.initial_state)])
        self.visited.add(self._state_hash(self.initial_state))
        
        while queue:
            node = queue.popleft()
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, node.depth)
            
            if self.goal_test(node.state):
                return node
            
            if node.depth >= max_depth:
                continue
            
            for action, new_state in self.successors(node.state):
                state_hash = self._state_hash(new_state)
                if state_hash not in self.visited:
                    self.visited.add(state_hash)
                    child = SearchNode(
                        state=new_state,
                        parent=node,
                        action=action,
                        cost=node.cost + 1.0,
                        depth=node.depth + 1
                    )
                    node.children.append(child)
                    queue.append(child)
        
        return None
    
    def _dfs(self, max_depth: int) -> Optional[SearchNode]:
        """Depth-first search."""
        stack = [SearchNode(self.initial_state)]
        self.visited.add(self._state_hash(self.initial_state))
        
        while stack:
            node = stack.pop()
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, node.depth)
            
            if self.goal_test(node.state):
                return node
            
            if node.depth >= max_depth:
                continue
            
            for action, new_state in reversed(self.successors(node.state)):
                state_hash = self._state_hash(new_state)
                if state_hash not in self.visited:
                    self.visited.add(state_hash)
                    child = SearchNode(
                        state=new_state,
                        parent=node,
                        action=action,
                        cost=node.cost + 1.0,
                        depth=node.depth + 1
                    )
                    node.children.append(child)
                    stack.append(child)
        
        return None
    
    def _a_star(self, max_depth: int) -> Optional[SearchNode]:
        """A* search."""
        open_set = []
        start_node = SearchNode(
            state=self.initial_state,
            heuristic=self.heuristic(self.initial_state)
        )
        heapq.heappush(open_set, start_node)
        self.visited.add(self._state_hash(self.initial_state))
        
        while open_set:
            node = heapq.heappop(open_set)
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, node.depth)
            
            if self.goal_test(node.state):
                return node
            
            if node.depth >= max_depth:
                continue
            
            for action, new_state in self.successors(node.state):
                state_hash = self._state_hash(new_state)
                if state_hash not in self.visited:
                    self.visited.add(state_hash)
                    child = SearchNode(
                        state=new_state,
                        parent=node,
                        action=action,
                        cost=node.cost + 1.0,
                        heuristic=self.heuristic(new_state),
                        depth=node.depth + 1
                    )
                    node.children.append(child)
                    heapq.heappush(open_set, child)
        
        return None
    
    def _beam_search(self, max_depth: int, beam_width: int) -> Optional[SearchNode]:
        """Beam search."""
        current_level = [SearchNode(self.initial_state)]
        self.visited.add(self._state_hash(self.initial_state))
        
        for depth in range(max_depth):
            next_level = []
            
            for node in current_level:
                if self.goal_test(node.state):
                    return node
                
                for action, new_state in self.successors(node.state):
                    state_hash = self._state_hash(new_state)
                    if state_hash not in self.visited:
                        self.visited.add(state_hash)
                        child = SearchNode(
                            state=new_state,
                            parent=node,
                            action=action,
                            cost=node.cost + 1.0,
                            heuristic=self.heuristic(new_state),
                            depth=depth + 1
                        )
                        node.children.append(child)
                        next_level.append(child)
            
            # Keep only top beam_width nodes
            next_level.sort(key=lambda n: n.cost + n.heuristic)
            current_level = next_level[:beam_width]
            self.nodes_expanded += len(current_level)
            self.max_depth = depth + 1
        
        return None
    
    def _greedy(self, max_depth: int) -> Optional[SearchNode]:
        """Greedy best-first search."""
        open_set = []
        start_node = SearchNode(
            state=self.initial_state,
            heuristic=self.heuristic(self.initial_state)
        )
        heapq.heappush(open_set, start_node)
        self.visited.add(self._state_hash(self.initial_state))
        
        while open_set:
            node = heapq.heappop(open_set)
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, node.depth)
            
            if self.goal_test(node.state):
                return node
            
            if node.depth >= max_depth:
                continue
            
            for action, new_state in self.successors(node.state):
                state_hash = self._state_hash(new_state)
                if state_hash not in self.visited:
                    self.visited.add(state_hash)
                    child = SearchNode(
                        state=new_state,
                        parent=node,
                        action=action,
                        cost=node.cost + 1.0,
                        heuristic=self.heuristic(new_state),
                        depth=node.depth + 1
                    )
                    node.children.append(child)
                    heapq.heappush(open_set, child)
        
        return None
    
    def _state_hash(self, state: Any) -> str:
        """Hash state for visited set."""
        if isinstance(state, (tuple, list)):
            return str(state)
        elif hasattr(state, '__hash__'):
            return str(hash(state))
        else:
            return str(id(state))
    
    def get_search_statistics(self) -> Dict:
        """Get search statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'max_depth': self.max_depth,
            'visited_states': len(self.visited),
        }

