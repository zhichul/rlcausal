import re
import numpy as np

threshold = 0.01
    
def extract_solution(solution_str):
    final_answer_str="ANSWER"
    match = re.search(final_answer_str + r"\s*" + r"\[(((\-?[0-9\.])+,\s+)*((\-?[0-9\.])+))\]", solution_str)
    if match is None:
        return None
    else:
        numbers = match.group(1)
        try:
            solution = [float(s) for s in numbers.split(",")]
            return solution
        except:
            return None


def tvd(p1, p2):
    if not ((isinstance(p1, list) or isinstance(p1, np.ndarray)) and \
         (isinstance(p2, list) or isinstance(p2, np.ndarray))):
        raise ValueError((p1, p2))
    if len(p1) != len(p2):
        raise ValueError((p1, p2))
    p1 = np.array(p1)
    p2 = np.array(p2)
    return (np.abs(p1 - p2).sum() / 2).item()


def compute_score(data_source, solution_str, ground_truth, extractor=extract_solution, extra_info=None):
    answer = extractor(solution_str)
    if answer is None: # format error
        score = 0
        reward= 0
    else:
        score = 0.1
        reward = 0.1 # for formatting correctly
        if len(answer) == len(ground_truth):
            reward += 0.1 # for getting the right length
            score += 0.1
            reward += 0.8 * max(1-tvd(answer, ground_truth), 0) # for getting dist. correctly
            if tvd(answer, ground_truth) < threshold:
                score += 0.8
    return {'score': score, 'reward': reward, 'pred': solution_str}