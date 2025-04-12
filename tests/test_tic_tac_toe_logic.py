import unittest
import numpy as np
# Adjust the import based on your project structure if needed
from src.tic_tac_toe_logic import (
    create_initial_state, get_valid_actions, apply_action,
    check_win_condition, check_draw_condition, get_next_player,
    BOARD_SIZE, EMPTY, PLAYER_X, PLAYER_O
)

class TestTicTacToeLogic(unittest.TestCase):

    def test_initial_state(self):
        board = create_initial_state()
        self.assertEqual(board.shape, (BOARD_SIZE, BOARD_SIZE))
        self.assertTrue(np.all(board == EMPTY))

    def test_valid_actions_empty(self):
        board = create_initial_state()
        actions = get_valid_actions(board)
        self.assertEqual(len(actions), BOARD_SIZE * BOARD_SIZE)
        self.assertIn((0, 0), actions)
        self.assertIn((1, 1), actions)
        self.assertIn((2, 2), actions)

    def test_apply_action_valid(self):
        board = create_initial_state()
        new_board = apply_action(board, (0, 0), PLAYER_X)
        self.assertEqual(new_board[0, 0], PLAYER_X)
        self.assertEqual(board[0, 0], EMPTY) # Original board should not change

    def test_apply_action_invalid(self):
        board = create_initial_state()
        board = apply_action(board, (0, 0), PLAYER_X)
        with self.assertRaises(ValueError):
            apply_action(board, (0, 0), PLAYER_O) # Try applying to occupied cell

    def test_valid_actions_mid_game(self):
        board = create_initial_state()
        board = apply_action(board, (0, 0), PLAYER_X)
        board = apply_action(board, (1, 1), PLAYER_O)
        actions = get_valid_actions(board)
        self.assertEqual(len(actions), BOARD_SIZE * BOARD_SIZE - 2)
        self.assertNotIn((0, 0), actions)
        self.assertNotIn((1, 1), actions)
        self.assertIn((0, 1), actions)

    def test_win_condition_row(self):
        board = np.array([[PLAYER_X, PLAYER_X, PLAYER_X], [EMPTY, PLAYER_O, EMPTY], [PLAYER_O, EMPTY, EMPTY]])
        self.assertTrue(check_win_condition(board, PLAYER_X))
        self.assertFalse(check_win_condition(board, PLAYER_O))

    def test_win_condition_col(self):
         board = np.array([[PLAYER_X, PLAYER_O, PLAYER_X], [EMPTY, PLAYER_O, EMPTY], [PLAYER_O, PLAYER_O, EMPTY]])
         self.assertTrue(check_win_condition(board, PLAYER_O))
         self.assertFalse(check_win_condition(board, PLAYER_X))

    def test_win_condition_diag(self):
        board = np.array([[PLAYER_X, PLAYER_O, EMPTY], [EMPTY, PLAYER_X, EMPTY], [PLAYER_O, EMPTY, PLAYER_X]])
        self.assertTrue(check_win_condition(board, PLAYER_X))

    def test_win_condition_antidiag(self):
        board = np.array([[EMPTY, PLAYER_O, PLAYER_X], [EMPTY, PLAYER_X, EMPTY], [PLAYER_X, EMPTY, PLAYER_O]])
        self.assertTrue(check_win_condition(board, PLAYER_X))

    def test_no_win(self):
        board = np.array([[PLAYER_X, PLAYER_O, PLAYER_X], [PLAYER_O, PLAYER_O, PLAYER_X], [PLAYER_X, PLAYER_X, PLAYER_O]])
        self.assertFalse(check_win_condition(board, PLAYER_X))
        self.assertFalse(check_win_condition(board, PLAYER_O))

    def test_draw_condition(self):
        # Full board, no winner
        board = np.array([[PLAYER_X, PLAYER_O, PLAYER_X], [PLAYER_O, PLAYER_O, PLAYER_X], [PLAYER_X, PLAYER_X, PLAYER_O]])
        self.assertFalse(check_win_condition(board, PLAYER_X)) # Pre-condition
        self.assertFalse(check_win_condition(board, PLAYER_O)) # Pre-condition
        self.assertTrue(check_draw_condition(board))

    def test_not_draw_condition(self):
         board = np.array([[PLAYER_X, PLAYER_X, PLAYER_X], [EMPTY, PLAYER_O, EMPTY], [PLAYER_O, EMPTY, EMPTY]])
         self.assertFalse(check_draw_condition(board)) # Winner exists
         board_mid_game = np.array([[PLAYER_X, EMPTY, EMPTY], [EMPTY, PLAYER_O, EMPTY], [EMPTY, EMPTY, EMPTY]])
         self.assertFalse(check_draw_condition(board_mid_game)) # Not full

    def test_get_next_player(self):
        self.assertEqual(get_next_player(PLAYER_X), PLAYER_O)
        self.assertEqual(get_next_player(PLAYER_O), PLAYER_X)

if __name__ == '__main__':
    unittest.main()