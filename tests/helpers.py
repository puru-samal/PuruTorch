import numpy as np
import json
import sys
import traceback

# ANSI escape codes for text color
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def cmp_usr_pyt_tensor(myTensor, pytTensor, test_type, test_name):
        user_vals = myTensor.data
        expected_vals = pytTensor.detach().numpy()
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
                assert np.allclose(user_vals, expected_vals,atol=1e-4)
            except Exception as e:
                print("Closeness error, your values dont match the expected values.")
                print("Wrong values for %s" % test_name)
                print("Your values:    ", user_vals)
                print("Expected values:", expected_vals)
                return False
        return True


def print_failure(cur_test, num_dashes=51):
    print('*' * num_dashes)
    print('The local autograder will not work if you do not pass %s.' % cur_test)
    print('*' * num_dashes)
    print(' ')

def print_name(cur_question):
    print(cur_question)

def print_outcome(short, outcome, point_value, num_dashes=51):
    score = point_value if outcome else 0
    if score != point_value:
        print(RED+"{}: {}/{}".format(short, score, point_value)+RESET)
        print('-' * num_dashes)

def run_tests(tests, summarize=False):
    # calculate number of dashes to print based on max line length
    title = "TESTS"
    num_dashes = calculate_num_dashes(tests, title)

    # print title of printout
    print(generate_centered_title(title, num_dashes))

    # Print each test
    scores = {}
    for t in tests:
        if not summarize:
            print_name(t['name'])
        try:
            res = t['handler']()
        except Exception:
            res = False
            traceback.print_exc()
        if not summarize:
            print_outcome(t['autolab'], res, t['value'], num_dashes)
        scores[t['autolab']] = t['value'] if res else 0

    points_available = sum(t['value'] for t in tests)
    points_gotten = sum(scores.values())

    print("\nSummary:")
    pretty_print_scores(scores, points_gotten, points_available)
    #print(json.dumps({'scores': scores}))

def calculate_num_dashes(tests, title):
    """Determines how many dashes to print between sections (to be ~pretty~)"""
    # Init based on str lengths in printout
    str_lens = [len(t['name']) for t in tests] + [len(t['autolab']) + 4 for t in tests]
    num_dashes = max(str_lens) + 1

    # Guarantee minimum 5 dashes around title
    if num_dashes < len(title) - 4:
        return len(title) + 10

    # Guarantee even # dashes around title
    if (num_dashes - len(title)) % 2 != 0:
        return num_dashes + 1

    return num_dashes

def generate_centered_title(title, num_dashes):
    """Generates title string, with equal # dashes on both sides"""
    dashes_on_side = int((num_dashes - len(title)) / 2) * "-"
    return dashes_on_side + title + dashes_on_side

def pretty_print_scores(scores, points_gotten, points_available):
    # Determine the maximum length of keys for formatting
    max_key_length = max(len(key) for key in scores)

    # Print the table header
    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')
    print(f'| {"Test".center(max_key_length)} | {"Score".center(11)} |')
    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')

    # Print the table rows
    for key, value in scores.items():
        value_padding = 12 - len(str(value))
        if value == 0: START = RED
        else: START = GREEN
        print('| ' + START + str(key).ljust(max_key_length) + RESET + ' | ' + START + str(value).center(value_padding) + RESET + ' |')

    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')
    print('| ' + 'TOTAL'.center(max_key_length) + ' | ' + f'{points_gotten}/{points_available}'.center(value_padding) + ' |')
    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')