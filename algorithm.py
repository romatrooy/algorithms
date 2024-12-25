import numpy as np
import pandas as pd
import random
import math
from multiprocessing import Pool

class AntColonyOptimizer:
    def __init__(self, distance_matrix, num_ants=50, num_iterations=100,
                 alpha=1.0, beta=3.0, evaporation_rate=0.1, 
                 initial_pheromone=1.0, elite_factor=2.0):
        """
        Улучшенные параметры алгоритма
        """
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Влияние феромона
        self.beta = beta    # Влияние расстояния
        self.evaporation_rate = evaporation_rate
        self.elite_factor = elite_factor  # Усиление лучшего маршрута
        self.pheromone_matrix = np.full(distance_matrix.shape, initial_pheromone)
        self.num_nodes = distance_matrix.shape[0]

    def _calculate_probability(self, current_node, unvisited_nodes):
        """
        Оптимизированное вычисление вероятностей перехода
        """
        pheromones = self.pheromone_matrix[current_node, unvisited_nodes]
        distances = self.distance_matrix[current_node, unvisited_nodes]
        # Используем np.divide для избежания деления на 0
        inverse_distances = np.divide(1.0, distances, where=distances!=0)
        attractiveness = np.multiply(
            np.power(pheromones, self.alpha),
            np.power(inverse_distances, self.beta)
        )
        # Нормализуем вероятности
        total = np.sum(attractiveness)
        if total == 0:
            # Равномерное распределение если все значения 0
            return np.ones_like(attractiveness) / len(attractiveness)
        return attractiveness / total

    def _update_pheromones(self, all_routes, all_costs):
        """
        Улучшенное обновление феромонов с элитизмом
        """
        # Испарение
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Находим лучший маршрут
        best_idx = np.argmin(all_costs)
        best_route = all_routes[best_idx]
        best_cost = all_costs[best_idx]
        
        # Обновление феромонов для всех маршрутов
        for route, cost in zip(all_routes, all_costs):
            for i in range(len(route) - 1):
                from_node, to_node = route[i], route[i + 1]
                delta = 1.0 / cost
                self.pheromone_matrix[from_node, to_node] += delta
                self.pheromone_matrix[to_node, from_node] += delta
        
        # Элитное усиление лучшего маршрута
        for i in range(len(best_route) - 1):
            from_node, to_node = best_route[i], best_route[i + 1]
            delta = self.elite_factor / best_cost
            self.pheromone_matrix[from_node, to_node] += delta
            self.pheromone_matrix[to_node, from_node] += delta

    def _ant_tour(self, start_node):
        """
        Отдельная функция для одного муравья
        """
        current_node = start_node
        unvisited = set(range(self.num_nodes)) - {current_node}
        route = [current_node]
        cost = 0
        
        while unvisited:
            probabilities = self._calculate_probability(current_node, list(unvisited))
            next_node = np.random.choice(list(unvisited), p=probabilities)
            cost += self.distance_matrix[current_node, next_node]
            route.append(next_node)
            current_node = next_node
            unvisited.remove(next_node)
        
        # Замыкаем маршрут
        cost += self.distance_matrix[route[-1], route[0]]
        route.append(route[0])
        
        return route, cost

    def optimize(self):
        """
        Параллельная версия оптимизации
        """
        best_route = None
        best_cost = float('inf')
        
        with Pool() as pool:
            for iteration in range(self.num_iterations):
                # Параллельный запуск муравьев
                start_nodes = np.random.randint(0, self.num_nodes, self.num_ants)
                results = pool.map(self._ant_tour, start_nodes)
                
                all_routes, all_costs = zip(*results)
                
                # Обновление лучшего решения
                min_cost_idx = np.argmin(all_costs)
                if all_costs[min_cost_idx] < best_cost:
                    best_route = all_routes[min_cost_idx]
                    best_cost = all_costs[min_cost_idx]
                
                self._update_pheromones(all_routes, all_costs)
                
                if iteration % 10 == 0:
                    print(f"Итерация {iteration}: Лучшая стоимость = {best_cost}")
        
        return best_route, best_cost

def load_data(file_path, limit=100):
    """
    Загружает данные из файла и создает матрицу расстояний.
    
    :param file_path: Путь к файлу data.csv
    :param limit: Ограничение количества точек для тестирования
    :return: Матрица расстояний и названия населенных пунктов
    """
    data = pd.read_csv(file_path)
    data = data.head(limit)  # берем только первые limit строк
    coords = data[["latitude_dd", "longitude_dd"]].values / 100
    num_points = len(coords)

    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i, j] = math.sqrt(
                    (coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2
                )

    return distance_matrix, data["settlement"].tolist()

def save_solution(route, distances, output_file):
    """
    Сохраняет маршрут и расстояния в файл.

    :param route: Маршрут (индексы узлов)
    :param distances: Матрица расстояний
    :param output_file: Имя выходного файла
    """
    with open(output_file, "w") as f:
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = distances[from_node, to_node]
            f.write(f"{from_node};{distance}\n")

# Основной код
if __name__ == "__main__":
    # Путь к файлам для локального тестирования
    input_file = "data.csv"  # локальный путь к файлу
    output_file = "solution.csv"

    # Загрузка данных с ограничением количества точек
    distance_matrix, settlements = load_data(input_file, limit=100)  # увеличено с 100 до 1000 точек

    # Создание оптимизатора с уменьшенными параметрами для быстрого тестирования
    aco = AntColonyOptimizer(
        distance_matrix,
        num_ants=50,        # оставляем 20 для скорости
        num_iterations=50,   # оставляем 20 для скорости
        alpha=1.0,
        beta=3.0,
        evaporation_rate=0.1,
        elite_factor=2.0
    )

    # Запуск оптимизации
    best_route, best_cost = aco.optimize()

    # Вывод результатов
    print("Лучший маршрут:", [settlements[i] for i in best_route])
    print("Стоимость маршрута:", best_cost)

    # Сохранение результатов локально
    save_solution(best_route, distance_matrix, output_file)
