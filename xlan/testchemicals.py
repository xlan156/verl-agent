def mkRandomChemicalCombination(rng, numChemicalTypes=3, minChemicals=1, minAmount=3, maxAmount=5):
    chemicalNames = ["Substance A", "Substance B", "Substance C", "Substance D", "Substance E", "Substance F"]
    chemicalNames = chemicalNames[:numChemicalTypes]
    rng.shuffle(chemicalNames)

    totalParts = rng.randint(minAmount, maxAmount) # total amounts of chemicals
    
    k = rng.randint(minChemicals, min(numChemicalTypes, totalParts)) # k is

    selected = chemicalNames[:k]

    chemicalDict = {name: 1 for name in selected}

    while True:
        s = sum(chemicalDict.values())
        if s == totalParts:
            break
        if s > totalParts:
            break
        key = rng.choice(list(chemicalDict.keys()))
        chemicalDict[key] += 1

    return chemicalDict


if __name__ == "__main__":
    import random
    rng = random.Random(13)
    for _ in range(100):
        combo = mkRandomChemicalCombination(
            rng,
            numChemicalTypes=5,
            minChemicals=1,
            minAmount=1,
            maxAmount=8
        )
        print(combo)