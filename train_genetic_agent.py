import torch
import random
import sys
import os

from src.Board import Board
from src.Colour import Colour
from src.Game import Game
from src.Player import Player
from src.Move import Move

from agents.GeneticAgents.NeuralNetwork import NeuralNetwork
from agents.GeneticAgents.GeneticAgent import GeneticAgent

class GeneticTrainer:
    '''
    Trains neural networks
    '''

    def __init__(self, population_size=30, board_size=11):
        self.population_size = population_size
        self.board_size = board_size
        self.population = [NeuralNetwork() for _ in range(population_size)]
        self.generation = 0

    def evaluate_fitness(self,  games_per_matchup=2):
        print(f"\n=== Generation {self.generation} ===")

        fitness_scores = [0.0] * len(self.population)
        game_counts = [0] * len(self.population)

        num_opponents = min(10, len(self.population) - 1)

        for i, network in enumerate(self.population):
            opponents_idx = random.sample(
                [j for j in range(len(self.population)) if j != i],
                num_opponents
            )

            for opp_idx in opponents_idx:
                for _ in range(games_per_matchup):
                    winner = self.play_game(network, self.population[opp_idx])
                    game_counts[i] += 1
                    game_counts[opp_idx] += 1

                    if winner == 0:
                        fitness_scores[i] += 1
                    else:
                        fitness_scores[opp_idx] += 1

        fitness = [
            fitness_scores[i] / game_counts[i] if game_counts[i] > 0 else 0.0
            for i in range(len(self.population))
        ]

        sorted_pairs = sorted(zip(self.population, fitness), key=lambda x: x[1],reverse=True)
        self.population = [net for net, _ in sorted_pairs]

        print(f"Best Fitness: {fitness[0]:.3f}")
        print(f"Avg Fitness: {sum(fitness) / len(fitness):.3f}")

        return fitness
    
    def play_game(self, network1, network2):
        '''
        Returns 0 if network1 wins, 1 if network2 wins
        '''

        p1 = GeneticAgent(Colour.RED, network=network1)
        p2 = GeneticAgent(Colour.BLUE, network=network2)

        game = Game(
            player1 = Player(name="Network1", agent=p1),
            player2 = Player(name="Network2", agent=p2),
            board_size=11,
            logDest=os.devnull,
            verbose=False,
            silent=True
        )

        result = game.run()
        winner = result["winner"]

        if(winner == "Network1" ):
            return 0
        return 1
    def evolve(self, elite_count=3, mutation_rate = 0.1):
        '''
        Create next generation
        '''

        self.generation += 1

        new_population = self.population[:elite_count]

        while len(new_population) < self.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            child = NeuralNetwork.crossover(parent1, parent2)
            child = NeuralNetwork.mutate(child, strength=mutation_rate)

            new_population.append(child)
        self.population = new_population

    def tournament_select(self, tournament_size=3):
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return tournament[0]
    
    def save_best(self, filepath="agents/GeneticAgents/best_brain.pt"):
        self.population[0].save(filepath)
        print(f"Saved best network to {filepath}")

    def train(self, generations=50):
        for gen in range(generations):
            fitness = self.evaluate_fitness(games_per_matchup=2)

            if gen % 10 == 0:
                self.save_best()

            self.evolve(elite_count=3, mutation_rate=0.1)
        self.save_best()
        print(f"\nTraining complete. Best fitness: {fitness[0]:.3f}")

if __name__ == "__main__":
    trainer = GeneticTrainer(population_size=30, board_size=11)
    trainer.train(generations=50)