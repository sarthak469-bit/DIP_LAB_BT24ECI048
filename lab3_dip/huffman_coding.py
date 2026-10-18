
import heapq
from itertools import count
from math import log2


def build_huffman_codes(symbols, probabilities):
    # Each heap item contains probability, unique ID, and tree node
    serial = count()
    heap = []

    for symbol, probability in zip(symbols, probabilities):
        heapq.heappush(
            heap, (probability, next(serial), symbol)
        )

    # Repeatedly combine the two least probable nodes
    while len(heap) > 1:
        p1, _, left = heapq.heappop(heap)
        p2, _, right = heapq.heappop(heap)

        parent = (left, right)

        heapq.heappush(
            heap, (p1 + p2, next(serial), parent)
        )

    root = heap[0][2]
    codes = {}

    # Traverse the tree to generate binary codes
    def assign_codes(node, prefix=""):
        if isinstance(node, str):
            codes[node] = prefix or "0"
            return

        left, right = node
        assign_codes(left, prefix + "0")
        assign_codes(right, prefix + "1")

    assign_codes(root)
    return codes


def main():
    symbols = ["A", "B", "C", "D"]
    probabilities = [0.40, 0.30, 0.20, 0.10]

    if len(symbols) != len(probabilities):
        raise ValueError("Symbols and probabilities must match.")

    if any(p <= 0 for p in probabilities):
        raise ValueError("Probabilities must be positive.")

    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to 1.")

    codes = build_huffman_codes(symbols, probabilities)

    print("HUFFMAN CODING")
    print("-" * 38)
    print(f"{'Symbol':<12}{'Probability':<15}{'Code'}")

    for symbol, probability in zip(symbols, probabilities):
        print(f"{symbol:<12}{probability:<15.2f}{codes[symbol]}")

    average_length = sum(
        p * len(codes[symbol])
        for symbol, p in zip(symbols, probabilities)
    )

    entropy = -sum(p * log2(p) for p in probabilities)
    efficiency = 100 * entropy / average_length

    print(f"\nSource entropy: {entropy:.4f} bits/symbol")
    print(f"Average code length: {average_length:.4f} bits/symbol")
    print(f"Coding efficiency: {efficiency:.2f}%")

    # Encode a sample message
    message = "ABACABAD"
    encoded = "".join(codes[ch] for ch in message)

    print(f"\nOriginal message: {message}")
    print(f"Encoded message:  {encoded}")
    print(f"Encoded length:   {len(encoded)} bits")


if __name__ == "__main__":
    main()
