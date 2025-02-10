import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

class EvolutionaryAlgorithm:
    def __init__(
        self, 
        num_individuals: int,
        reward_component_names: List[str],
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.5,
        elitism_rate: float = 0.1,
        ema_decay: float = 0.9
    ) -> None:
        self.population = [
            {name: np.random.uniform(0, 1) for name in reward_component_names} 
            for _ in range(num_individuals)
        ]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.reward_component_names = reward_component_names
        self.ema_decay = ema_decay
        self.ema_weights: Dict[str, float] = {name: 1.0 / len(reward_component_names) for name in reward_component_names}
        self.best_fitness: float = float('-inf')
        self.best_individual: Dict[str, float] = {}

    def set_weights(self, new_weights: Dict[str, float]) -> None:
        for name, weight in new_weights.items():
            if name in self.reward_component_names:
                self.ema_weights[name] = weight
            else:
                raise ValueError(f"No reward function registered under the name {name}.")

    def evaluate_fitness(self, task: Any, model: Any, data_loader: Any) -> List[float]:
        fitness_scores = []
        for individual_weights in self.population:
            task.set_reward_weights(individual_weights)
            results = task.evaluation(model, data_loader)
            fitness = sum(result.get('CIDEr', 0) for result in results) / len(results)
            fitness_scores.append(fitness)
        return fitness_scores

    def evolve(self, task: Any, model: Any, data_loader: Any) -> Tuple[Dict[str, float], Dict[str, float], float]:
        fitness_scores = self.evaluate_fitness(task, model, data_loader)
        best_index = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_index]
        best_individual = self.population[best_index].copy()

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_individual = best_individual

        self.select(fitness_scores)
        self.crossover_and_mutate()

        for key in self.best_individual:
            self.ema_weights[key] = self.ema_decay * self.ema_weights[key] + (1 - self.ema_decay) * self.best_individual[key]

        total = sum(self.ema_weights.values())
        self.ema_weights = {k: v / total for k, v in self.ema_weights.items()}

        print(f"Best fitness: {self.best_fitness}")
        print("Best individual: ", self.best_individual)
        print("EMA weights: ", self.ema_weights)

        return self.best_individual, self.ema_weights, best_fitness

    def select(self, fitness_scores: List[float]) -> None:
        selected = []
        tournament_size = 5

        for _ in tqdm(range(len(self.population)), desc="Selection"):
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            contenders = [self.population[i] for i in indices]
            contender_scores = [fitness_scores[i] for i in indices]
            best_index = np.argmax(contender_scores)
            winner = contenders[best_index]
            selected.append(winner)

        self.population = selected

    def crossover_and_mutate(self) -> None:
        next_generation = []
        num_elites = int(self.elitism_rate * len(self.population))
        
        sorted_population = sorted(self.population, key=lambda x: sum(x.values()), reverse=True)
        next_generation.extend(sorted_population[:num_elites])

        while len(next_generation) < len(self.population):
            parent1, parent2 = np.random.choice(self.population, 2, replace=False)
            if np.random.rand() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.extend([self.mutate(child1), self.mutate(child2)])
            else:
                next_generation.extend([self.mutate(parent1.copy()), self.mutate(parent2.copy())])

        self.population = next_generation[:len(self.population)]

    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        crossover_point = np.random.randint(1, len(self.reward_component_names))
        child1 = {**dict(list(parent1.items())[:crossover_point]), **dict(list(parent2.items())[crossover_point:])}
        child2 = {**dict(list(parent2.items())[:crossover_point]), **dict(list(parent1.items())[crossover_point:])}
        return child1, child2

    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        for gene in individual:
            if np.random.rand() < self.mutation_rate:
                individual[gene] += np.random.normal(0, 0.1)
                individual[gene] = max(0, min(individual[gene], 1))
        return individual