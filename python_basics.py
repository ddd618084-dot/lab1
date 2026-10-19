from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    return sum(1 for ch in text if ch in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    return bin(data.value).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for ch in str(number):
            product *= int(ch)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)

    parts = []
    i = 0
    while i < len(factors):
        p = factors[i]
        e = 1
        while i + e < len(factors) and factors[i + e] == p:
            e += 1
        parts.append(f"({p}**{e})" if e > 1 else f"({p})")
        i += e
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s = str(data.value)
    n = len(s)
    half = n // 2
    left = sum(int(ch) for ch in s[:half])
    right = sum(int(ch) for ch in s[n - half:])
    return left == right
