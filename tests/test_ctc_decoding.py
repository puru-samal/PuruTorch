import numpy as np
import sys, os
import traceback
import json
import pickle
sys.path.append("./")
from PuruTorch.nn.CTCDecoding import GreedySearchDecoder, BeamSearchDecoder


# Test object to be used for other homeworks
class Test(object):
    def __init__(self):
        self.scores = {}

    def assertions(self, user_vals, expected_vals, test_type, test_name):
        if test_type == "type":
            try:
                assert type(user_vals) == type(expected_vals)
            except Exception as e:
                print("Type error, your type doesnt match the expected type.")
                print("Wrong type for %s" % test_name)
                print("Your type:   ", type(user_vals))
                print("Expected type:", type(expected_vals))
                return False
        elif test_type == "shape":
            try:
                assert user_vals.shape == expected_vals.shape
            except Exception as e:
                print("Shape error, your shapes doesnt match the expected shape.")
                print("Wrong shape for %s" % test_name)
                print("Your shape:    ", user_vals.shape)
                print("Expected shape:", expected_vals.shape)
                return False
        elif test_type == "closeness":
            try:
                assert np.allclose(user_vals, expected_vals,atol=1e-5)
            except Exception as e:
                print("Closeness error, your values dont match the expected values.")
                print("Wrong values for %s" % test_name)
                print("Your values:    ", user_vals)
                print("Expected values:", expected_vals)
                return False
        return True

    def print_failure(self, cur_test):
        print("*" * 50)
        print("The local autograder failed %s." % cur_test)
        print("*" * 50)
        print(" ")

    def print_name(self, cur_question):
        print("-" * 20)
        print(cur_question)

    def print_outcome(self, short, outcome):
        print(short + ": ", "PASS" if outcome else "*** FAIL ***")
        print("-" * 20)
        print()

    def get_test_scores(self):
        return sum(self.scores.values())
        
    def run_tests(self, section_title, test, test_score):
        test_name = section_title.split(' - ')[1]
        try:
            self.print_name(section_title)
            test_outcome = test()
            self.print_outcome(test_name, test_outcome)
        except Exception:
            traceback.print_exc()
            test_outcome = False
        
        if test_outcome != True:
            self.print_failure(test_name)
            if test_outcome==False:
                self.scores[test_name] = 0
            else:
                self.scores[test_name] = test_outcome[1]
            return False
        self.scores[test_name] = test_score
        return True


# DO NOT CHANGE -->
isTesting = True
EPS = 1e-20

# -->

class SearchTest(Test):
    def __init__(self):
        pass

    def test_greedy_search(self):
        SEED = 11785
        np.random.seed(11785)
        y_rands = np.random.uniform(EPS, 1.0, (4, 10, 1))
        y_sum = np.sum(y_rands, axis=0)
        y_probs = y_rands / y_sum
        SymbolSets = ["a", "b", "c"]

        expected_results = np.load(
            os.path.join("tests",  "data", "greedy_search.npy"),
            allow_pickle=True,
        )
        ref_best_path, ref_score = expected_results

        decoder = GreedySearchDecoder(SymbolSets)
        user_best_path, user_score = decoder.decode(y_probs)

        if isTesting:
            try:
                assert user_best_path == ref_best_path
            except Exception as e:
                print("Best path does not match")
                print("Your best path:    \t", user_best_path)
                print("Expected best path:\t", ref_best_path)
                return False

            try:
                assert user_score == float(ref_score)
            except Exception as e:
                print("Best Score does not match")
                print("Your score:    \t", user_score)
                print("Expected score:\t", ref_score)
                return False

        # Use to save test data for next semester
        if not isTesting:
            results = [user_best_path, user_score]
            np.save(os.path.join('tests', 
                             'data', 'greedy_search.npy'), results, allow_pickle=True)

        return True

    def test_beam_search_i(self, SEED, y_size, syms, bw, BestPath_ref, MergedPathScores_ref):
        np.random.seed(SEED)
        y_rands = np.random.uniform(EPS, 1.0, y_size)
        y_sum = np.sum(y_rands, axis=0)
        y_probs = y_rands / y_sum

        SymbolSets = syms
        BeamWidth = bw

        decoder = BeamSearchDecoder(SymbolSets, BeamWidth)
        BestPath, MergedPathScores = decoder.decode(y_probs)

        if isTesting:
            try:
                assert BestPath == BestPath_ref
            except Exception as e:
                print("BestPath does not match!")
                print("Your best path:    \t", BestPath)
                print("Expected best path:\t", BestPath_ref)
                return False

            try:
                assert len(MergedPathScores.keys()) == len(MergedPathScores)
            except Exception as e:
                print("Total number of merged paths returned does not match")
                print(
                    "Number of merged path score keys: ",
                    "len(MergedPathScores.keys()) = ",
                    len(MergedPathScores.keys()),
                )
                print(
                    "Number of merged path scores:",
                    "len(MergedPathScores)= ",
                    len(MergedPathScores),
                )
                return False

            no_path = False
            values_close = True

            for key in MergedPathScores_ref.keys():
                if key not in MergedPathScores.keys():
                    no_path = True
                    print("%s path not found in reference dictionary" % (key))
                    return False
                else:
                    if not self.assertions(
                        MergedPathScores_ref[key],
                        MergedPathScores[key],
                        "closeness",
                        "beam search",
                    ):
                        values_close = False
                        print("score for %s path not close to reference score" % (key))
                        return False
            return True
        else:
            return BestPath, MergedPathScores

    def test_beam_search(self):
        expected_results = np.load(
            os.path.join("tests", "data", "beam_search.npy"),
            allow_pickle=True,
        )

        # Initials
        ysizes = [(4, 10, 1), (5, 20, 1), (6, 20, 1)]
        symbol_sets = [["a", "b", "c"], ["a", "b", "c", "d"], ["a", "b", "c", "d", "e"]]
        beam_widths = [2, 3, 3]

        n = 3
        results = []
        for i in range(n):
            BestPathRef, MergedPathScoresRef = expected_results[i]
            y_size, syms, bw = ysizes[i], symbol_sets[i], beam_widths[i]
            result = self.test_beam_search_i(
                i, y_size, syms, bw, BestPathRef, MergedPathScoresRef
            )
            if isTesting:
                if result != True:
                    print("Failed Beam Search Test: %d / %d" % (i + 1, n))
                    return False
                else:
                    print("Passed Beam Search Test: %d / %d" % (i + 1, n))
            else:
                results.append(result)

        # Use to save test data for next semester
        if not isTesting:
            np.save(os.path.join('tests',
                             'data', 'beam_search.npy'), results, allow_pickle=True)
        return True

    def run_test(self):
        # Test Greedy Search
        self.print_name("Section 5.1 - Greedy Search")
        greedy_outcome = self.test_greedy_search()
        self.print_outcome("Greedy Search", greedy_outcome)
        if greedy_outcome == False:
            self.print_failure("Greedy Search")
            return False

        # Test Beam Search
        self.print_name("Section 5.2 - Beam Search")
        beam_outcome = self.test_beam_search()
        self.print_outcome("Beam Search", beam_outcome)
        if beam_outcome == False:
            self.print_failure("Beam Search")
            return False

        return True
