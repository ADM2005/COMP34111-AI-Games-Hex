import torch
import numpy as np
import random
import time

from collections import deque
from agents.GeneticAgents.NeuralNetworkCUDA import NeuralNetworkCUDA

class RandomNetwork(torch.nn.Module):
    def __init__(self, board_size=11, device=None):
        super().__init__()

        self.board_size = board_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)

        # If tensor does not have a batch dimension, add one
        if x.dim() == 3:
            x = x.unsqueeze(0)  

        return torch.randn(x.size(dim=0), self.board_size * self.board_size).to(self.device)
    
class BatchedHexGames:
    '''

    Manages games simultaneously and advances games in parallel, leveraging batched processing.
    Each network makes moves for all its games at once.

    '''

    def __init__(self, board_size=11, device='cuda'):
        self.board_size = board_size
        self.device = device if torch.cuda.is_available() else 'cpu'

    def play_all_matchups(self, network_pairs, games_per_pair=1):
        '''
        Plays all matchups with full batching.

        Args:
            network_pairs: List of (network1, network2, pair_id) tuples
            games_per_pair: How many games per matchup

        Returns:
            results: Dict mapping pair_id to (wins1, wins2)
        '''

        # Expands pairs to individual games
        game_configs = []
        for n1, n2, pair_id in network_pairs:
            for _ in range(games_per_pair):
                game_configs.append( (n1, n2, pair_id) )
        
        total_games = len(game_configs)
        if total_games == 0:
            return {}
        
        print(f"Playing {total_games} games in parallel...", end='', flush=True)

        # Iniitalize all boards
        # (total_games, board_size, board_size, 2)

        all_boards = torch.zeros(total_games, self.board_size, self.board_size, 2, device=self.device)

        active = torch.ones(total_games, dtype=torch.bool, device=self.device)
        winners = torch.zeros(total_games, dtype=torch.long, device=self.device)

        max_moves = self.board_size * self.board_size
        current_player = 1

        for move_num in range(max_moves):
            if not active.any():
                break

            # Group games by witch network is playing
            network_to_games = {}

            for game_idx in range(total_games):
                if not active[game_idx]:
                    continue

                n1, n2, pair_id = game_configs[game_idx]
                current_net = n1 if current_player == 1 else n2

                if id(current_net) not in network_to_games:
                    network_to_games[id(current_net)] = {
                        'network': current_net,
                        'game_indices': []
                    }
                network_to_games[id(current_net)]['game_indices'].append(game_idx)

            all_moves = [None] * total_games

            for net_info in network_to_games.values():

                network = net_info['network']
                game_indices = net_info['game_indices']

                batch_boards = all_boards[game_indices]  # (batch, h, w, 2)

                canonical_boards = self.canonicalize_batch(batch_boards, current_player)

                with torch.no_grad():
                    logits = network.forward(canonical_boards)

                    legal_masks = self.get_legal_moves_batch(canonical_boards)
                    masked_logits = logits + torch.where(
                        legal_masks,
                        torch.tensor(0.0, device=self.device),
                        torch.tensor(float('-inf'), device=self.device)
                    )

                    moves = torch.argmax(masked_logits, dim=1)
                for i, game_idx in enumerate(game_indices):
                    all_moves[game_idx] = moves[i].item()

            for game_idx in range(total_games):
                if not active[game_idx]:
                    continue

                move = all_moves[game_idx]
                if move is None:
                    continue


                row = move // self.board_size
                col = move % self.board_size
                channel = current_player - 1
                if current_player == 2:
                    row, col = col, row
                all_boards[game_idx, row, col, channel] = 1.0

            self.check_winners_batch(all_boards, active, winners, current_player)

            active = active & (winners == 0)

            current_player = 2 if current_player == 1 else 1
        print("Done")

        results = {}
        for game_idx, (n1, n2, pair_id) in enumerate(game_configs):
            if pair_id not in results:
                results[pair_id] = {"wins1": 0, "wins2": 0}
            
            if winners[game_idx] == 1:
                results[pair_id]["wins1"] += 1
            elif winners[game_idx] == 2:
                results[pair_id]["wins2"] += 1
        return results
    
    def canonicalize_batch(self, boards, current_player):
        if current_player == 2:
            boards = boards.permute(0, 2, 1, 3)
            boards = boards[..., [1,0]]
        return boards
    
    def get_legal_moves_batch(self, boards):
        occupied = (boards[:, :, :, 0] + boards[:, :, :, 1]) > 0
        legal = ~occupied
        return legal.reshape(boards.size(0), -1)

    def check_winners_batch(self, boards, active_mask, winners, last_player):
        active_indices = torch.where(active_mask)[0]

        if len(active_indices) == 0:
            return
            
        boards_cpu = boards[active_indices].cpu().numpy()
        for i, game_idx in enumerate(active_indices):
            board = boards_cpu[i]

            if self.check_connection(board[:, :, 0], axis=0):
                winners[game_idx] = 1
            elif self.check_connection(board[:, :, 1], axis=1):
                winners[game_idx] = 2
    def check_connection(self, board, axis):

        # Uses BFS to check for connection

        if axis == 0:
            # Check Top to Bottom for a Connection
            start_positions = [ (0, j) for j in range(self.board_size) if board[0,j] == 1]
            target_row = self.board_size - 1
            is_target = lambda pos: pos[0] == target_row

        else:
            start_positions = [(i,0) for i in range(self.board_size) if board[i, 0] == 1]
            target_col = self.board_size - 1
            is_target = lambda pos: pos[1] == target_col

        if not start_positions:
            return False
        
        visited = set(start_positions)
        queue = deque(start_positions)

        while queue:
            row, col = queue.popleft()

            if is_target( (row, col) ):
                return True
            
            for nr, nc in self.get_hex_neighbors(row, col):
                if (nr, nc) not in visited and board[nr, nc] == 1:
                    visited.add( (nr, nc) )
                    queue.append(  (nr, nc))
        return False
    
    def get_hex_neighbors(self, row, col):
        neighbors = [
            (row-1, col), (row-1, col + 1),
            (row, col-1), (row, col+1),
            (row+1, col-1), (row+1, col)
        ]

        return [ (r,c) for r, c in neighbors if 0 <= r < self.board_size and 0 <= c < self.board_size ]
    
    def print_board(self, board_tensor, winning_path=None):
        """
        Print a single board from tensor format.
        
        Args:
            board_tensor: (board_size, board_size, 2) tensor
                         Channel 0: Player 1 (RED) pieces
                         Channel 1: Player 2 (BLUE) pieces
            winning_path: Optional set of (row, col) tuples to highlight in green
        
        Returns:
            String representation of the board
        """
        # Convert to numpy if tensor
        if isinstance(board_tensor, torch.Tensor):
            board = board_tensor.cpu().numpy()
        else:
            board = board_tensor
        
        if winning_path is None:
            winning_path = set()
        
        # Import Colour class (adjust import path as needed)
        from src.Colour import Colour
        
        output = ""
        
        # Top red edge (column indices in red)
        output += "  " + "".join(Colour.red(f"{i:2d}") for i in range(self.board_size)) + "\n"
        
        leading_spaces = ""
        for row in range(self.board_size):
            # Left blue edge (row index in blue)
            output += " " + leading_spaces + Colour.blue(f"{row:2d}")
            
            for col in range(self.board_size):
                # Determine what's in this cell
                has_red = board[row, col, 0] == 1.0
                has_blue = board[row, col, 1] == 1.0
                
                # Determine the symbol
                if (row, col) in winning_path:
                    # Highlight winning path in green
                    if has_red:
                        cell_char = Colour.green("R")
                    elif has_blue:
                        cell_char = Colour.green("B")
                    else:
                        cell_char = Colour.green("·")
                else:
                    # Normal display
                    if has_red:
                        cell_char = Colour.get_char(Colour.RED)
                    elif has_blue:
                        cell_char = Colour.get_char(Colour.BLUE)
                    else:
                        cell_char = "·"  # Empty cell
            
                output += cell_char + " "
            
            # Right blue edge
            output += Colour.blue(f"{row:2d}") + "\n"
            leading_spaces += " "
        
        # Bottom red edge
        output += " " + leading_spaces + "".join(
            Colour.red(f"{i:2d}") for i in range(self.board_size)
        ) + "\n"
        
        return output
    
class FullyBatchedGeneticTrainer:
    '''
    Batched trainer
    '''

    def __init__(self, population_size=30, board_size=11, use_cuda=True):
        self.population_size = population_size
        self.board_size = board_size
        self.generation = 0

        self.use_cuda = use_cuda and torch.cuda.is_available() 
        self.device = 'cuda' if self.use_cuda else 'cpu'

        if self.use_cuda:
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available, using CPU")

        self.game_manager = BatchedHexGames(board_size, device=self.device)
        self.population = [ NeuralNetworkCUDA(board_size, device=self.device) for _ in range(population_size)]

        self.baselines = self._create_baselines_()

    def _create_baselines_(self):
        baselines = {}

        random_net = RandomNetwork(self.board_size, self.device)
        baselines['random'] = random_net

        return baselines
    
    def evaluate_against_baselines(self, network, games_per_baseline=20):
        results = {}
        for baseline_name, baseline_net in self.baselines.items():
            matchup_results = self.game_manager.play_all_matchups(
                [(network, baseline_net, 0)],
                games_per_pair=games_per_baseline
            )

            wins = matchup_results[0]['wins1']
            losses = matchup_results[0]['wins2']
            total = wins + losses

            win_rate = wins / total if total > 0 else 0.0
            results[baseline_name] = win_rate

        return results
    def evaluate_fitness(self, num_opponents=10, games_per_matchup=4):
        '''
        Evaluate fitness with full batching
        '''

        print(f"\n=== Generation {self.generation} ===")

        fitness_scores = [0.0] * len(self.population)
        game_counts = [0] * len(self.population)

        all_pairs = []
        pair_to_indices = {}        # Maps pair id to indices of population

        for i in range(len(self.population)):
            opponents_idx = random.sample(
                [j for j in range(len(self.population)) if j != i],
                min(num_opponents, len(self.population) - 1)
            )

            for opp_idx in opponents_idx:
                pair_id = len(all_pairs)
                all_pairs.append((
                    self.population[i],
                    self.population[opp_idx],
                    pair_id
                ))

                pair_to_indices[pair_id] = (i, opp_idx)
        
        total_matchups = len(all_pairs)
        total_games = total_matchups * games_per_matchup

        print(f" {total_matchups} matchups x {games_per_matchup} games = {total_games} total games")

        start_time = time.time()
        results = self.game_manager.play_all_matchups(all_pairs, games_per_matchup)
        elapsed = time.time() - start_time

        print(f"Completed in {elapsed:.2f}s ({total_games/elapsed:.1f} games/sec)")

        for pair_id, result in results.items():
            i, opp_idx = pair_to_indices[pair_id]

            wins1 = result["wins1"]
            wins2 = result["wins2"]
            total = wins1 + wins2

            fitness_scores[i] += wins1
            fitness_scores[opp_idx] += wins2

            game_counts[i] += total
            game_counts[opp_idx] += total
        
        fitness = [
            fitness_scores[i] / game_counts[i] if game_counts[i] > 0 else 0.0 
            for i in range(len(self.population))
        ]

        sorted_pairs = sorted(zip(self.population, fitness), key = lambda x: x[1], reverse=True)
        self.population = [net for net, _ in sorted_pairs]
        fitness = [f for _, f in sorted_pairs]

        print(f"Best: {fitness[0]:.3f} | Avg: {sum(fitness)/len(fitness):.3f} | Worst: {fitness[-1]:.3f}")

        return fitness
    
    def evolve(self, elite_count=3, mutation_rate=0.1):
        self.generation += 1

        new_population = []
        for i in range(elite_count):
            elite = self.population[i]

            elite_copy = NeuralNetworkCUDA.deepcopy(elite)
            new_population.append(elite_copy)

        while len(new_population) < self.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            child = NeuralNetworkCUDA.crossover(parent1, parent2)
            child = NeuralNetworkCUDA.mutate(child, strength=mutation_rate)

            new_population.append(child)

        self.population = new_population

    def tournament_select(self, tournament_size=3):
        tournament = random.sample(range(len(self.population)), min(tournament_size, len(self.population)))
        return self.population[min(tournament)]
    
    def save_best(self, filepath="agents/GeneticAgents/best_brain.pt"):
        self.population[0].save(filepath)
        print(f"Saved best network to {filepath}")

    def save_checkpoint(self, filepath="agents/GeneticAgents/checkpoint.pt"):
        checkpoint = {
            'generation' : self.generation,
            'population_states': [net.state_dict() for net in self.population],
            'board_size' : self.board_size
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        self.generation = checkpoint['generation']
        self.board_size = checkpoint['board_size']

        self.population = []

        for state in checkpoint['population_states']:
            net = NeuralNetworkCUDA(self.board_size, device=self.device)
            net.load_state_dict(state)
            self.population.append(net)

        print(f"Loaded checkpoint from generation {self.generation}")

    def train(self, generations=50, save_interval=2, num_opponents=10, games_per_matchup=4, baseline_evaluation_rate=1):
        print(f"\n{'='*70}")
        print(f"Starting FULLY BATCHED training")
        print(f"Generations: {generations}")
        print(f"Population: {self.population_size}")
        print(f"Opponents per network: {num_opponents}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"{'='*70}")
        
        total_start = time.time()
        try:
            for gen in range(self.generation, generations):
                if gen % baseline_evaluation_rate == 0:
                    baseline_results = self.evaluate_against_baselines(self.population[0], 1000)
                    print(f"Winrate against random: {baseline_results['random']:.3f}")
                gen_start = time.time()

                fitness = self.evaluate_fitness(num_opponents, games_per_matchup)

                if gen % save_interval == 0:            
                    self.save_best()
                    self.save_checkpoint()

                if gen < generations - 1:
                    self.evolve(elite_count=2, mutation_rate=0.5)    

                gen_time = time.time() - gen_start
                elapsed = time.time() - total_start
                avg_time = elapsed / (gen + 1)
                eta = avg_time * (generations - gen - 1)

                print(f"Gen time: {gen_time:.1f}s | Total: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                print(f"Progress: {gen+1}/{generations}\n")

            self.save_best()
            self.save_checkpoint()

            total_time = time.time() - total_start
            print(f"\n{'='*70}")
            print(f"Training complete in {total_time/60:.1f} minutes!")
            print(f"Best fitness: {fitness[0]:.3f}")
            print(f"Average: {total_time/generations:.1f}s per generation")
            print(f"{'='*70}\n")

        except KeyboardInterrupt:
            print("\n Interrupted. Saving checkpoint...")
            self.save_checkpoint("agents/GeneticAgents/interrupted.pt")
            print("Checkpoint Saved")


        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fully Batched Genetic Hex Trainer")
    parser.add_argument('--population', type=int, default=100)
    parser.add_argument('--generations', type=int, default=200)
    parser.add_argument('--opponents', type=int, default=10,
                       help='Opponents per network per generation')
    parser.add_argument('--games', type=int, default=10,
                       help='Games per matchup')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = FullyBatchedGeneticTrainer(
        population_size=args.population,
        board_size=11,
        use_cuda=not args.no_cuda
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(
        generations=args.generations,
        num_opponents=args.opponents,
        games_per_matchup=args.games
    )