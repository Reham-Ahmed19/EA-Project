import pandas as pd
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

class PopulationComparison:
    def __init__(self):
        # Constants
        self.SMALL_POP_SIZE = 200
        self.LARGE_POP_SIZE = 1000
        self.GENERATIONS = 100
        self.MUTATION_RATE = 0.3
        self.CROSSOVER_RATE = 0.8
        self.OVER_SELECTION_THRESHOLD = 0.2
        self.OVER_SELECTION_BIAS = 0.7
        
        # Data structures
        self.student_courses = None
        self.course_students = None
        self.timeslot_to_day = None
        self.room_capacities = None
        self.courses = None
        self.rooms = None
        self.timeslots = None
        
        # Results storage
        self.small_pop_results = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
        self.large_pop_results = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

    def load_data(self):
        """Load the exam scheduling data"""
        try:
            # Load schedule data
            schedule_df = pd.read_csv('schedule.csv')
            self.student_courses = defaultdict(list)
            self.course_students = defaultdict(list)
            
            for _, row in schedule_df.iterrows():
                self.student_courses[row['student_id']].append(row['course_id'])
                self.course_students[row['course_id']].append(row['student_id'])
            
            self.courses = list(self.course_students.keys())
            
            # Load classroom data
            classrooms_df = pd.read_csv('classrooms.csv')
            self.room_capacities = {row['classroom_id']: row['capacity'] for _, row in classrooms_df.iterrows()}
            self.rooms = list(self.room_capacities.keys())
            
            # Load timeslot data
            timeslots_df = pd.read_csv('timeslots.csv')
            self.timeslot_to_day = {row['timeslot_id']: row['day'] for _, row in timeslots_df.iterrows()}
            self.timeslots = list(self.timeslot_to_day.keys())
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def create_individual(self):
        """Create a random timetable"""
        timetable = {}
        for course in self.courses:
            time = random.choice(self.timeslots)
            students = self.course_students[course].copy()
            random.shuffle(students)
            
            room_assignments = []
            while students:
                room = random.choice(self.rooms)
                capacity = self.room_capacities[room]
                assigned = students[:capacity]
                students = students[capacity:]
                room_assignments.append((room, assigned))
            
            timetable[course] = (time, room_assignments)
        return timetable

    def calculate_fitness(self, timetable):
        """Evaluate timetable quality"""
        conflicts = 0
        room_usage = defaultdict(set)
        student_schedule = defaultdict(set)
        
        for course, (time, assignments) in timetable.items():
            for room, students in assignments:
                # Check room capacity
                if len(students) > self.room_capacities[room]:
                    conflicts += 1
                
                # Check room double booking
                if time in room_usage[room]:
                    conflicts += 1
                room_usage[room].add(time)
                
                # Check student conflicts
                for student in students:
                    if time in student_schedule[student]:
                        conflicts += 1
                    student_schedule[student].add(time)
        
        return -conflicts  # Higher is better

    def over_selection(self, population, fitness_scores):
        """Select individuals with over-selection"""
        # Sort by fitness (best first)
        ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
        
        # Split into top and bottom groups
        split = int(len(population) * self.OVER_SELECTION_THRESHOLD)
        top = [ind for (_, ind) in ranked[:split]]
        bottom = [ind for (_, ind) in ranked[split:]]
        
        selected = []
        for _ in range(len(population)):
            if random.random() < self.OVER_SELECTION_BIAS and top:
                selected.append(random.choice(top))
            elif bottom:
                selected.append(random.choice(bottom))
        
        return selected

    def evolve_population(self, population_size):
        """Run evolution for a population size"""
        population = [self.create_individual() for _ in range(population_size)]
        fitness = [self.calculate_fitness(ind) for ind in population]
        
        best_fitness = []
        avg_fitness = []
        diversity = []
        
        for _ in range(self.GENERATIONS):
            # Selection
            parents = self.over_selection(population, fitness)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    p1, p2 = parents[i], parents[i+1]
                    
                    # Crossover
                    if random.random() < self.CROSSOVER_RATE:
                        c1 = {course: p1[course] if random.random() < 0.5 else p2[course] 
                             for course in self.courses}
                        c2 = {course: p2[course] if random.random() < 0.5 else p1[course] 
                             for course in self.courses}
                    else:
                        c1, c2 = p1.copy(), p2.copy()
                    
                    # Mutation
                    if random.random() < self.MUTATION_RATE:
                        c1 = self.mutate(c1)
                    if random.random() < self.MUTATION_RATE:
                        c2 = self.mutate(c2)
                    
                    offspring.extend([c1, c2])
            
            # Evaluate offspring
            population = offspring
            fitness = [self.calculate_fitness(ind) for ind in population]
            
            # Track stats
            best_fitness.append(max(fitness))
            avg_fitness.append(np.mean(fitness))
            diversity.append(self.calculate_diversity(population))
        
        return best_fitness, avg_fitness, diversity

    def mutate(self, individual):
        """Apply mutation to an individual"""
        mutated = individual.copy()
        course = random.choice(self.courses)
        
        # Time slot mutation
        if random.random() < 0.5:
            mutated[course] = (random.choice(self.timeslots), mutated[course][1])
        # Room assignment mutation
        else:
            time, assignments = mutated[course]
            students = [s for room, students in assignments for s in students]
            random.shuffle(students)
            
            new_assignments = []
            while students:
                room = random.choice(self.rooms)
                capacity = self.room_capacities[room]
                new_assignments.append((room, students[:capacity]))
                students = students[capacity:]
            
            mutated[course] = (time, new_assignments)
        
        return mutated

    def calculate_diversity(self, population):
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
        
        unique = set()
        for ind in population:
            key = tuple(sorted((course, time) for course, (time, _) in ind.items()))
            unique.add(key)
        
        return len(unique) / len(population)

    def run_comparison(self):
        """Run comparison between small and large populations"""
        print("Running evolution for small population...")
        (self.small_pop_results['best_fitness'],
         self.small_pop_results['avg_fitness'],
         self.small_pop_results['diversity']) = self.evolve_population(self.SMALL_POP_SIZE)
        
        print("Running evolution for large population...")
        (self.large_pop_results['best_fitness'],
         self.large_pop_results['avg_fitness'],
         self.large_pop_results['diversity']) = self.evolve_population(self.LARGE_POP_SIZE)
        
        self.plot_results()

    def plot_results(self):
        """Plot comparison results"""
        plt.figure(figsize=(15, 5))
        
        # Best fitness comparison
        plt.subplot(1, 3, 1)
        plt.plot(self.small_pop_results['best_fitness'], label=f'Small ({self.SMALL_POP_SIZE})')
        plt.plot(self.large_pop_results['best_fitness'], label=f'Large ({self.LARGE_POP_SIZE})')
        plt.title('Best Fitness Comparison')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        # Average fitness comparison
        plt.subplot(1, 3, 2)
        plt.plot(self.small_pop_results['avg_fitness'], label=f'Small ({self.SMALL_POP_SIZE})')
        plt.plot(self.large_pop_results['avg_fitness'], label=f'Large ({self.LARGE_POP_SIZE})')
        plt.title('Average Fitness Comparison')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        # Diversity comparison
        plt.subplot(1, 3, 3)
        plt.plot(self.small_pop_results['diversity'], label=f'Small ({self.SMALL_POP_SIZE})')
        plt.plot(self.large_pop_results['diversity'], label=f'Large ({self.LARGE_POP_SIZE})')
        plt.title('Population Diversity Comparison')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Population Size Comparison for Exam Scheduling")
    comparator = PopulationComparison()
    comparator.load_data()
    comparator.run_comparison()