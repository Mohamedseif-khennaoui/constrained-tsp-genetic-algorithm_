# Heuristic and Meta-Heuristic Search for Constrained TSP

A research implementation comparing local search and genetic algorithm approaches for solving the Traveling Salesman Problem with distance constraints.

## Overview

This repository contains the implementation and experimental results from research conducted at the University of Abdelhamid Mehri, Constantine 2, Algeria. The work investigates the effectiveness of hill climbing algorithms versus genetic algorithms for solving a constrained variant of the TSP.

## Problem Definition

Given 100 cities distributed in a 500×500 km bounded space, find the shortest Hamiltonian cycle where cities can only be connected if their Euclidean distance is less than 90 km.

### Mathematical Formulation

```
Distance: d(i,j) = √[(x_i - x_j)² + (y_i - y_j)²]
Constraint: edge(i,j) exists ⟺ d(i,j) < 90 km
Objective: minimize Σ d(city_i, city_{i+1})
```

This constraint transforms the problem into a graph-theoretic challenge on a sparse graph structure, maintaining NP-hard complexity.

## Algorithms

### Hill Climbing

Local search method with iterative improvement:
- Initial solution generation via random cycle
- Neighborhood exploration through node replacement
- Greedy selection of improved solutions
- Termination at local optimum

### Genetic Algorithm

Population-based evolutionary approach:
- Population size: 50 individuals
- Maximum generations: 300
- Mutation rate: 0.2
- Selection: Proportional (roulette wheel)
- Crossover: Constraint-aware operator
- Fitness: Total distance with constraint penalties

## Results

Experimental comparison on 100-city instance:

| Metric | Hill Climbing | Genetic Algorithm |
|--------|--------------|-------------------|
| Execution Time (ms) | 38,141.99 | 688.23 |
| Current Solution (km) | 11,681 | 5,954 |
| Best Solution (km) | 7,289 | 5,208 |

The genetic algorithm demonstrates 49% improvement in solution quality with 98% reduction in execution time.

## Implementation

### Structure

```
constrained-tsp-genetic-algorithm/
├── src/
│   ├── hill_climbing.py
│   ├── genetic_algorithm.py
│   └── utils.py
├── experiments/
├── data/
├── results/
│   ├── figures/
│   └── tables/
├── docs/
└── tests/
```

### Dependencies

```
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
pandas>=1.3.0
```

### Usage

```python
from src.genetic_algorithm import GeneticAlgorithm
from src.utils import load_cities

cities = load_cities('data/cities_100.txt')
ga = GeneticAlgorithm(cities, population_size=50, max_generations=300)
solution, cost = ga.run()
```

## Analysis

### Convergence Behavior

Hill climbing exhibits rapid initial improvement followed by premature convergence to local optima. The genetic algorithm maintains steady improvement throughout execution, demonstrating superior exploration capabilities.

### Computational Complexity

While genetic algorithms have higher per-generation complexity, improved convergence characteristics result in better overall time-to-solution metrics for this constrained problem variant.

### Hybrid Approaches

Potential hybrid implementations include:
- Initial population seeding via hill climbing
- Local optimization of genetic algorithm offspring
- Parent refinement before crossover operations

Trade-offs involve exponential complexity growth versus solution quality improvements.

## Future Work

- Extension to larger problem instances
- Multi-objective optimization variants
- Parallel implementation strategies
- Application to real-world routing problems
- Comparative analysis with other metaheuristics

## Citation

```bibtex
@techreport{khennaoui2025constrained,
  title={Heuristic and Meta-Heuristic Search to Solve Cycle Finding Problem},
  author={Khennaoui, Mohamed Seif},
  institution={University of Abdelhamid Mehri, Constantine 2},
  year={2025},
  month={November}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

Mohamed Seif Khennaoui  
University of Abdelhamid Mehri, Constantine 2, Algeria  
Email: mohamed.khennaoui@univ-constantine2.dz

## Acknowledgments

This research was conducted as part of the Foundation/Master1 SDIA program at the University of Abdelhamid Mehri, Constantine 2, Algeria.

## Note on Code Availability

Source code will be made publicly available following publication of associated research papers. This follows standard academic practice to protect intellectual property during peer review.
