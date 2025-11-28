import torch
import os

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase
from agents.GeneticAgents.NeuralNetwork import NeuralNetwork

class GeneticAgent(AgentBase):
    def __init__(self, colour: Colour, model_path: str = None, network: NeuralNetwork = None):
        super().__init__(colour)


        if network != None:
            self.network = network
        else:
            if model_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "best_brain.pt")

            if os.path.exists(model_path):
                self.network = NeuralNetwork.load(model_path)
                print(f"Loaded neural network from {model_path}")
            else:
                print(f"Warning: {model_path} not found, using random network")
                self.network = NeuralNetwork()
            self.network.eval()  

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        '''
        Make move using neural network
        '''
        board_tensor = self.board_to_tensor(board)

        with torch.no_grad():
            logits = self.network.forward(board_tensor).squeeze(0)
        
        legal_moves = self.get_legal_moves(board)
        legal_mask = torch.full (  (121,), float('-inf'))

        for move in legal_moves:
            idx = move.y * 11 + move.x
            legal_mask[idx] = 0.0

        masked_logits = logits + legal_mask
        probabilities = torch.softmax(masked_logits, dim=0)

        move_idx = torch.argmax(probabilities).item()
        y = move_idx // 11
        x = move_idx % 11

        return Move(x,y)
    def board_to_tensor(self, board: Board) -> torch.Tensor:
        '''
        Convert board to NN input format.
        '''

        tensor = torch.zeros(11, 11, 2)

        for y in range(11):
            for x in range(11):
                cell = board.tiles[x][y]
                if cell == self._colour:
                    tensor[y, x, 0] = 1.0
                elif cell == self.opp_colour():
                    tensor[y, x, 1] = 1.0
        return tensor
    
    def get_legal_moves(self, board: Board) -> list[Move]:
        '''
        Get all legal moves
        '''
        legal_moves = []
        for y in range(board.size):
            for x in range(board.size):
                if board.tiles[x][y].colour == None:
                    legal_moves.append(Move(x,y))
        return legal_moves