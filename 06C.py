import numpy as np


class GeneticAlgorithm:

    def __init__(
            self,
            tasks_count: int,
            tasks: np.ndarray,
            tasks_time: np.ndarray,
            devs_count: int,
            coefficients: np.ndarray
    ):
        self.tasks_count = tasks_count
        self.tasks = tasks
        self.tasks_time = tasks_time
        self.devs_count = devs_count
        self.coefficients = coefficients

        self.rng = np.random.default_rng()
        self.population_size = 2500
        self.max_generations = 1000

        self.population = self.rng.integers(low=0, high=self.devs_count, size=(self.population_size, self.tasks_count))

    def fitness(self, individual):
        total_time = np.zeros(self.devs_count)
        for i in range(self.tasks_count):
            total_time[individual[i]] += self.tasks_time[i] * self.coefficients[
                individual[i], self.tasks[i] - 1]
        return np.max(total_time)

    def selection(self):
        # турирная селекция
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        selected_indices = np.argsort(fitness_values)[:self.population_size // 2 * 2]
        return self.population[selected_indices]

    def crossover(self, parent1, parent2):
        # скрещивание с помощью одноточечного кроссовера
        crossover_point = self.rng.integers(low=1, high=self.tasks_count)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutation(self, individual):
        # мутация с некоторой вероятностью (0.0002, условие ниже)
        mutated_gene = self.rng.integers(low=0, high=self.devs_count)
        mutation_point = self.rng.integers(low=0, high=self.tasks_count)
        individual[mutation_point] = mutated_gene
        return individual

    def step(self):
        for generation in range(self.max_generations):
            selected_population = self.selection()

            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)

                if self.rng.random() < 0.0002:
                    child1 = self.mutation(child1)
                if self.rng.random() < 0.0002:
                    child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            top_percent = int(0.25 * self.population_size)  # выбор 25 % детей
            self.population[:top_percent] = self.selection()[:top_percent]
            self.population[top_percent:] = np.array(new_population)[:self.population_size - top_percent]

        best_solution = self.population[np.argmin([self.fitness(ind) for ind in self.population])]
        return best_solution


# Чтение входных данных из файла
with open('input.txt') as f:
    tasks_count = int(f.readline())
    tasks = np.array(list(map(int, f.readline().split())))
    tasks_time = np.array(list(map(float, f.readline().split())))
    devs_count = int(f.readline())
    coefficients = np.array([list(map(float, f.readline().split())) for i in range(devs_count)])

ga = GeneticAlgorithm(tasks_count, tasks, tasks_time, devs_count, coefficients)

best_solution = ga.step()

with open('output.txt', 'w') as file:
    file.write(" ".join(map(lambda x: str(x + 1), best_solution)))

best_developer_times = np.zeros(ga.devs_count)
for i in range(ga.tasks_count):
    best_developer_times[best_solution[i]] += ga.tasks_time[i] * ga.coefficients[best_solution[i], ga.tasks[i] - 1]

Tmax = np.max(best_developer_times)

score = 100 / Tmax
print("Score:", score)
