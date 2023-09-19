# TODO: Try with parameters subsets, maybe too many inputs
# TODO: Another option could be to train a regression model, then classify as over or under prediction, two layers?
# TODO: outlier detection? current
# TODO: split by position and train different models for each
# TODO: ways to speed up data gathering - prepull all player/gameweek data rather than going 1 by 1, or store data and only pull new if it exists
# TODO: use classifier instead, predict top scorers - either based on raw score (ex. predict if a player will score > 5) OR
# based on % (ex. predict if player will be in top 10%, 11-50% or bottom 50% of all scorers for a given gameweek)