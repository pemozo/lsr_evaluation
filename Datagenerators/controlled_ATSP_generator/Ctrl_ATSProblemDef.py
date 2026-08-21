import torch

def get_random_problems(batch_size, node_cnt, problem_gen_params):

    int_min = problem_gen_params["int_min"]
    int_max = problem_gen_params["int_max"]
    scaler = problem_gen_params["scaler"]

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt), 
                             dtype=torch.int64)

    upper = torch.triu(problems, diagonal=1)
    problems = upper + upper.transpose(1, 2)

    idx = torch.arange(node_cnt)
    problems[:, idx, idx] = 0

    for k in range(node_cnt):
        via_k = (problems[:, :, k].unsqueeze(2) + problems[:, k, :].unsqueeze(1))
        problems = torch.minimum(problems, via_k)

    scaled_problems = problems.float() / scaler

    return scaled_problems

