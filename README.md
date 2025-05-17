# Exam Timetabling System - Genetic Algorithm Solution

## Overview 
This project provides an intelligent solution for university exam scheduling using genetic algorithms. The system automatically generates conflict-free exam timetables while optimizing resource utilization and meeting various academic constraints.

## Key Features 

### Core Algorithm
-  Multiple genetic operators (selection, crossover, mutation)
-  Hybrid fitness function with weighted constraints
-  Adaptive parameter tuning
-  Parallel evaluation for faster computation

### Scheduling Capabilities
-  Ensures no student exam conflicts
-  Optimizes classroom utilization
-  Distributes exams evenly across available days
-  Handles special constraints for difficult subjects

## Technical Implementation 

### Algorithm Components
| Component | Options Available |
|-----------|-------------------|
| Initialization | Random, Heuristic, Size-based |
| Selection | Tournament, Roulette Wheel, Exponential Rank |
| Crossover | Uniform, Single-point, Two-point |
| Mutation | Timeslot, Room Assignment, Split Rooms, Day Change |
| Survivor Selection | Generational, Steady-state, Elitism |

### Data Requirements
The system requires these CSV files:
1. schedule.csv - Student-course registrations
2. classrooms.csv - Room capacities  
3. timeslots.csv - Available timeslots
4. courses.csv - Course information
5. students.csv - Student IDs

## Usage Guide 

### Basic Execution
python
from ea import FinalExamScheduler

config = {
    "population_size": 100,
    "generations": 200,
    "mutation_rate": 0.3,
    "crossover_rate": 0.8
}

scheduler = FinalExamScheduler(config=config)
best_schedule = scheduler.run_evolution(num_runs=1)


### Web Interface
bash
streamlit run app.py


## Output Analysis 

The system provides comprehensive reporting:
- Detailed timetable visualization
- Constraint violation breakdown
- Room utilization statistics
- Evolution progress plots
- Student conflict analysis

## Configuration Options 

python
config = {
    "initialization_method": "Heuristic",  # Random/Heuristic/Size Based
    "crossover_type": "Two Point",  # Uniform/Single Point/Two Point
    "mutation_type": "Time Slot",  # Time Slot/Room Assignment/etc.
    "parent_selection": "Tournament",  # Tournament/Roulette/Exponential
    "population_size": 100,  # 10-200
    "generations": 150,  # 10-500
    "mutation_rate": 0.3,  # 0-1
    "crossover_rate": 0.8  # 0-1
}


