def top_k_accuracy(results, ground_truth, k=1):
    correct = 0

    for res, gt in zip(results, ground_truth):
        if gt in res[:k]:
            correct += 1

    return correct / len(results)