import random
import json
import os
from typing import Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

# Constants
PADDING_LENGTH = 16
MAX_DIGITS = 15
TOTAL_PROBLEMS_COUNT = 1e9
NUM_FILES = 500
OPERATIONS = ['+', '*']
DIRECTORY = "dataset/math"

# Make the directory if it does not exist
os.makedirs(DIRECTORY, exist_ok=True)


# 1. Generate random numbers of a given length with spaced digits
def generate_random_number(length: int) -> str:
    max_value = 10**length - 1
    min_value = 1 if length == 1 else 10**(length - 1)
    number = random.randint(min_value, max_value)
    spaced_number = ' '.join((str(number)).zfill(2*PADDING_LENGTH - 1))
    return spaced_number



def generate_problem(seed: int = None) -> Tuple[str, int]:
    random.seed(seed or os.getpid() + time.time())

    num1_length = random.randint(1, MAX_DIGITS)
    num2_length = random.randint(1, MAX_DIGITS)

    num1 = generate_random_number(num1_length)
    num2 = generate_random_number(num2_length)
    operation = random.choice(OPERATIONS)

    # Compute the answer depending on the chosen operation
    num1_unspaced = ''.join(num1.split())
    num2_unspaced = ''.join(num2.split())
    answer = int(num1_unspaced) + int(num2_unspaced) if operation == '+' else int(num1_unspaced) * int(num2_unspaced)
    return f"{num1} {operation} {num2}", answer


# 3. We prepare JSON representation of the problem
def create_json_entry(problem: str, answer: int) -> dict:
    padded_answer = str(answer).zfill(PADDING_LENGTH)[::-1]
    spaced_answer = ' '.join(padded_answer)
    return {"problem": f"{problem} = {spaced_answer}"}


# 4. Generate and save a single JSON file of problems.
def generate_and_save_json_file(file_index: int, problems_per_file: int) -> None:
    problems_list = [create_json_entry(*generate_problem()) for _ in tqdm(range(problems_per_file))]
    file_path = os.path.join(DIRECTORY, f"math{file_index + 1:03d}.json")
    with open(file_path, 'w') as json_file:
        json.dump(problems_list, json_file, indent=2)


# Create the JSON files using multiprocess
def write_problems_to_json_files(processes: int):
    problems_per_file = int(TOTAL_PROBLEMS_COUNT // NUM_FILES)
    with ProcessPoolExecutor(max_workers=processes) as executor:
        process_func = partial(generate_and_save_json_file, problems_per_file=problems_per_file)
        executor.map(process_func, range(NUM_FILES))


# Top-level function
def create_training_dataset_json(processes: int):
    print(f"Creating {NUM_FILES} JSON files with arithmetic problems.")
    write_problems_to_json_files(processes)
    print(f"JSON dataset creation complete. The files are saved in the directory: {DIRECTORY}")


if __name__ == "__main__":
    create_training_dataset_json(processes=2)