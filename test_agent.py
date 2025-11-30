import sys
import os

from src.Board import Board
from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.GeneticAgents.GeneticAgent import GeneticAgent
from agents.TestAgents.ValidAgent import ValidAgent

def test_agent():
    '''
    Test the agent
    '''

    p1 = GeneticAgent(Colour.RED)
    p2 = GeneticAgent(Colour.BLUE, "")


    game = Game(
        player1=Player(name="GeneticBest", agent=p1),
        player2=Player(name="GeneticRandom", agent=p2),
        board_size=11,
        logDest=sys.stderr,
        verbose=True
    )

    game.run()


if __name__ == "__main__":
    test_agent()