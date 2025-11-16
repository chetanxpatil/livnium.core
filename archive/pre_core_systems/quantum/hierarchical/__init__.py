"""
Hierarchical Geometry Machine

This is a quantum-inspired classical system. It is NOT a physical quantum computer.
"""

from quantum.hierarchical.geometry.level0.base_geometry import BaseGeometry, BaseGeometricState
from quantum.hierarchical.geometry.level1.geometry_in_geometry import GeometryInGeometry
from quantum.hierarchical.geometry.level2.geometry_in_geometry_in_geometry import HierarchicalGeometrySystem
from quantum.hierarchical.core.quantum_processor import QuantumProcessor

__all__ = [
    'BaseGeometry',
    'BaseGeometricState',
    'GeometryInGeometry',
    'HierarchicalGeometrySystem',
    'QuantumProcessor',
]

