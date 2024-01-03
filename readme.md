# Reconnaissance Blind Chess Bot - Zubat

This repository contains the implementation of a Reconnaissance Blind Chess (RBC) bot named "Zubat." 
The bot is designed to play at a level comparable or superior to existing bots, combining the efficiency of standard chess engines with strategies exploiting the opponent's imperfect information.

The implementation was originally derived from [StrangeFish2](https://github.com/ginop/reconchess-strangefish),
and uses its primary component as a basis. The original licence and citation are present in the [thrid_party_licence](./third_party_licence) directory.

## Methodology Summary

Zubat employs a multi-module approach for move selection, utilizing Board Tracking, Analytical, Risk Taker, Opponent’s Uncertainty Maximizer, Mediator, and Sensing Modules.

- **Analytical Module**: Evaluates moves using a chess engine, focusing on defensive play and cautious strategies. Implementation largely derived from StrangeFish2.
- **Risk Taker Module**: Identifies moves exploiting opponent uncertainty through shallow Monte Carlo simulations.
- **Opponent’s Uncertainty Maximizer Module**: Estimates the impact of moves on the opponent's confidence using an end-to-end neural network.
- **Mediator Module**: Combines evaluations from Analytical, Risk Taker, and Opponent’s Uncertainty Maximizer Modules to select the optimal move.
- **Sensing Module**: Chooses the best sensing location at the start of each turn, optimizing for effective moves and reducing board state uncertainty.

Zubat's decision-making process involves weighing the Analytical baseline, opponent uncertainty, and potential rewards/risks. This comprehensive approach aims to enhance strategic adaptability and performance in RBC games.