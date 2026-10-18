
from math import log2


def shannon_fano(symbols, probabilities):
    if len(symbols) != len(probabilities) or not symbols:
        raise ValueError("Provide matching, non-empty inputs.")

    if len(set(symbols)) != len(symbols):
        raise ValueError("Each symbol must be unique.")

    if any(p <= 0 for p in probabilities):
        raise ValueError("Probabilities must be positive.")

    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to 1.")

    # Sort symbols from highest to lowest probability
    items = sorted(
        zip(symbols, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    codes = {symbol: "" for symbol, _ in items}

    def divide(group):
        if len(group) <= 1:
            return

        total = sum(probability for _, probability in group)
        running_sum = 0
        best_split = 1
        smallest_difference = float("inf")

        # Find the split with the closest group totals
        for i in range(1, len(group)):
            running_sum += group[i - 1][1]
            difference = abs(total - 2 * running_sum)

            if difference < smallest_difference:
                smallest_difference = difference
                best_split = i

        left_group = group[:best_split]
        right_group = group[best_split:]

        for symbol, _ in left_group:
            codes[symbol] += "0"

        for symbol, _ in right_group:
            codes[symbol] += "1"

        divide(left_group)
        divide(right_group)

    divide(items)
    return items, codes


def main():
    symbols = ["A", "B", "C", "D"]
    probabilities = [0.40, 0.30, 0.20, 0.10]

    items, codes = shannon_fano(symbols, probabilities)

    print("SHANNON-FANO CODING")
    print("-" * 38)
    print(f"{'Symbol':<12}{'Probability':<15}{'Code'}")

    for symbol, probability in items:
        print(f"{symbol:<12}{probability:<15.2f}{codes[symbol]}")

    entropy = -sum(
        p * log2(p) for _, p in items
    )

    average_length = sum(
        p * len(codes[symbol]) for symbol, p in items
    )

    efficiency = 100 * entropy / average_length

    print(f"\nSource entropy: {entropy:.4f} bits/symbol")
    print(f"Average code length: {average_length:.4f} bits/symbol")
    print(f"Coding efficiency: {efficiency:.2f}%")

    message = "ABACABAD"
    encoded = "".join(codes[ch] for ch in message)

    print(f"\nOriginal message: {message}")
    print(f"Encoded message:  {encoded}")
    print(f"Encoded length:   {len(encoded)} bits")


if __name__ == "__main__":
    main()
