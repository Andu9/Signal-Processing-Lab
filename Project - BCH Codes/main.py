import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# used for constructing GF(2 ^ m)
primitivePolynomials = {
    2: 0x7,      # x ^ 2 + x + 1
    3: 0xB,      # x ^ 3 + x + 1
    4: 0x13,     # x ^ 4 + x + 1
    5: 0x25,     # x ^ 5 + x ^ 2 + 1
    6: 0x43,     # x ^ 6 + x + 1
    7: 0x89,     # x ^ 7 + x ^ 3 + 1
    8: 0x11D,    # x ^ 8 + x ^ 4 + x ^ 3 + x ^ 2 + 1
    9: 0x211,    # x ^ 9 + x ^ 4 + 1
    10: 0x409,   # x ^ 10 + x ^ 3 + 1
    11: 0x805,   # x ^ 11 + x ^ 2 + 1
    12: 0x1053,  # x ^ 12 + x ^ 6 + x ^ 4 + x + 1
    13: 0x201B,  # x ^ 13 + x ^ 4 + x ^ 3 + x + 1
    14: 0x4443,  # x ^ 14 + x ^ 10 + x ^ 6 + x + 1
    15: 0x8003,  # x ^ 15 + x + 1
    16: 0x1100B, # x ^ 16 + x ^ 12 + x ^ 3 + x + 1
}

@dataclass(frozen = True)
class ExtendedGF2m:
    extensionDegree: int
    primitivePolynomial: int

    def size(self) -> int:
        return (1 << self.extensionDegree) - 1

    def buildExtendedField(self) -> Tuple[List[int], List[int]]:
        size = self.size()
        exponent = [0] * (2 * size)
        logarithm = [-1] * (size + 1)

        alpha = 1
        for i in range(size):
            exponent[i] = alpha
            logarithm[alpha] = i
            alpha <<= 1 # multiplication by x

            if alpha & (1 << self.extensionDegree):
                alpha ^= self.primitivePolynomial # modulo p(x)

            alpha &= size

        for i in range(size, 2 * size):
            exponent[i] = exponent[i - size]

        return exponent, logarithm

@dataclass
class BCHCode:
    extensionDegree: int
    numberOfErrors: int
    primitivePolynomial: Optional[int] = None

    # gets cyclotomic coset of i, i. e. {i, 2i, 4i, 8i, ...} modulo n
    def getCyclotomicCoset(self, i: int) -> List[int]:
        n = self.size
        seenCoset = set()
        x = i % n
        coset = []
        while x not in seenCoset:
            seenCoset.add(x)
            coset.append(x)
            x = (2 * x) % n
        return coset

    # add two elements in the extended field GF(2 ^ m)
    @staticmethod
    def addExtensionElements(a: int, b: int) -> int:
        return a ^ b

    # multiply two elements in the extended field GF(2 ^ m)
    def multiplyExtensionElements(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exponent[self.logarithm[a] + self.logarithm[b]]

    # multiply two polynomials with coefficients in extended field GF(2 ^ m)
    def multiplyExtensionPolynomials(self, a: List[int], b: List[int]) -> List[int]:
        res = [0] * (len(a) + len(b) - 1)
        for i, coefficientA in enumerate(a):
            if coefficientA != 0:
                for j, coefficientB in enumerate(b):
                    if coefficientB != 0:
                        res[i + j] = self.addExtensionElements(
                            res[i + j],
                            self.multiplyExtensionElements(coefficientA, coefficientB)
                        )
        return res

    @staticmethod
    def trim(array: List[int]) -> List[int]:
        i = 0
        while i < len(array) - 1 and array[i] == 0:
            i += 1
        return array[i:]

    # get minimal polynomial for alpha ^ i over GF(2) using the coset property
    def getMinimalPolynomial(self, i: int) -> List[int]:
        coset = self.getCyclotomicCoset(i)
        polynomial = [1]

        for j in coset:
            alphaRaisedToJ = self.exponent[j]
            polynomial = self.multiplyExtensionPolynomials(polynomial,[alphaRaisedToJ, 1])

        return BCHCode.trim(list(reversed(polynomial)))

    # multiplies two polynomials with coefficients in GF(2) = {0, 1}
    @staticmethod
    def multiplyFieldPolynomials(a: List[int], b: List[int]) -> List[int]:
        a = list(reversed(BCHCode.trim(a[:])))
        b = list(reversed(BCHCode.trim(b[:])))

        res = [0] * (len(a) + len(b) - 1)
        for i, coefficientA in enumerate(a):
            if coefficientA != 0:
                for j, coefficientB in enumerate(b):
                    if coefficientB != 0:
                        res[i + j] ^= 1
        return BCHCode.trim(list(reversed(res)))

    # get the generator of the code
    def getGenerator(self) -> List[int]:
        usedCosets = set()
        generator = [1]
        for i in range(1, 2 * self.numberOfErrors + 1):
            coset = self.getCyclotomicCoset(i)
            cosetHash = frozenset(coset)

            if cosetHash not in usedCosets:
                usedCosets.add(cosetHash)
                minimalPolynomial = self.getMinimalPolynomial(i)
                generator = BCHCode.multiplyFieldPolynomials(generator, minimalPolynomial)
        return generator

    def __post_init__(self):
        if self.primitivePolynomial is None:
            self.primitivePolynomial = primitivePolynomials[self.extensionDegree]

        self.extendedField = ExtendedGF2m(self.extensionDegree, self.primitivePolynomial)
        self.size = self.extendedField.size()
        self.exponent, self.logarithm = self.extendedField.buildExtendedField()

        self.generator = self.getGenerator()
        self.redundancy = len(self.generator) - 1
        self.dimension = self.size - self.redundancy

    # returns the quotient and the remainder of dividend / divisor over GF(2)
    @staticmethod
    def divideFieldPolynomial(dividend: List[int], divisor: List[int]) -> Tuple[List[int], List[int]]:
        dividend = BCHCode.trim(dividend[:])
        divisor = BCHCode.trim(divisor[:])

        if len(dividend) < len(divisor):
            return [0], dividend

        remainder = dividend[:]
        quotient = [0] * (len(dividend) - len(divisor) + 1)
        for i in range(len(quotient)):
            if remainder[i] == 1:
                quotient[i] = 1
                for j in range(len(divisor)):
                    remainder[i + j] ^= divisor[j]

        if len(divisor) == 1:
            remainder = [0]
        else:
            remainder = remainder[-(len(divisor) - 1):]
            remainder = BCHCode.trim(remainder) if remainder else [0]
        quotient = BCHCode.trim(quotient) if quotient else [0]
        return quotient, remainder

    def encode(self, messageToEncode: List[int]) -> List[int]:
        dividend = messageToEncode[:] + [0] * self.redundancy
        _, remainder = self.divideFieldPolynomial(dividend, self.generator)

        remainder = [] if remainder == [0] else remainder[:]
        parity = [0] * (self.redundancy - len(remainder)) + remainder
        encodedMessage = messageToEncode[:] + parity
        return encodedMessage

    def evaluatePolynomialInAlpha(self, polynomial: List[int], alpha: int) -> int:
        size = self.size
        alpha = alpha % size
        res = 0
        for i, bit in enumerate(polynomial):
            if bit != 0:
                monomialDegree = (size - 1 - i) % size # correct exponent for MSB polynomial
                monomial = (alpha * monomialDegree) % size
                res ^= self.exponent[monomial]
        return res

    def getSyndromes(self, polynomial: List[int]) -> List[int]:
        return [self.evaluatePolynomialInAlpha(polynomial, alpha) for alpha in range(1, 2 * self.numberOfErrors + 1)]

    def divideExtensionElements(self, a: int, b: int) -> int:
        if a == 0:
            return 0
        return self.exponent[(self.logarithm[a] - self.logarithm[b]) % self.size]

    def BerlekampMasseyAlgorithm(self, syndromes: List[int]) -> List[int]:
        n = len(syndromes)
        C = [1] + [0] * n
        B = [1] + [0] * n
        L, b, shift = 0, 1, 1

        for i in range(n):
            d = syndromes[i]
            for j in range(1, L + 1):
                if C[j] and syndromes[i - j]:
                    d = self.addExtensionElements(d, self.multiplyExtensionElements(C[j], syndromes[i - j]))

            if d == 0:
                shift += 1
                continue

            T = C[:]
            coefficient = self.divideExtensionElements(d, b)
            for j in range(0, n - shift + 1):
                if B[j]:
                    C[j + shift] = self.addExtensionElements(C[j + shift], self.multiplyExtensionElements(coefficient, B[j]))

            if 2 * L <= i:
                L = i + 1 - L
                B = T
                b = d
                shift = 1
            else:
                shift += 1
        return C[:L + 1]

    def ChienSearch(self, lambdaPolynomial: List[int]) -> List[int]:
        n = self.size
        L = len(lambdaPolynomial) - 1
        errors = []
        for degree in range(n):
            x = self.exponent[(n - degree) % n]
            value = 0
            powerX = 1
            for i in range(L + 1):
                lambdaValueI = lambdaPolynomial[i]
                if lambdaValueI:
                    value = self.addExtensionElements(value, self.multiplyExtensionElements(lambdaValueI, powerX))
                powerX = self.multiplyExtensionElements(powerX, x)

            if value == 0:
                errors.append(n - 1 - degree) # again, MSB polynomial
        return errors

    def decode(self, messageToDecode: List[int]):
        syndromes = self.getSyndromes(messageToDecode)
        if all(syndrome == 0 for syndrome in syndromes):
            return messageToDecode[:self.dimension], messageToDecode[:]

        lambdaPolynomial = self.BerlekampMasseyAlgorithm(syndromes)
        errorsPositions = self.ChienSearch(lambdaPolynomial)

        correctMessage = messageToDecode[:]
        for errorPosition in errorsPositions:
            correctMessage[errorPosition] ^= 1

        return correctMessage[:self.dimension], correctMessage

if __name__ == '__main__':
    BCH = BCHCode(extensionDegree = 4, numberOfErrors = 3)
    print(f'BCH Parameters: n = {BCH.size}, m = {4}, t = {3}')
    print(f'Generator g(x) = {BCH.generator}')
    print(f'Redundancy r = {BCH.redundancy}')
    print(f'Dimension k = {BCH.dimension}')
    print()

    message = [random.randint(0, 1) for _ in range(BCH.dimension)]
    codeword = BCH.encode(message)

    print(f'Message = {message}')
    print(f'Encoded message = {codeword}')

    corruptedMessage = codeword[:]
    numErrors = random.randint(0, 3)
    errorPositions = random.sample(range(BCH.size), numErrors)

    for position in errorPositions:
        corruptedMessage[position] ^= 1

    print(f'Error positions = {errorPositions}')
    print(f'Corrupted message = {corruptedMessage}')

    decodedMessage, correctedMessage = BCH.decode(corruptedMessage)

    print(f'Corrected message = {correctedMessage}')
    print(f'Decoded message = {decodedMessage}')
    print()

    for trial in range(100):
        message = [random.randint(0, 1) for _ in range(BCH.dimension)]
        codeword = BCH.encode(message)

        corruptedMessage = codeword[:]
        numErrors = random.randint(0, 3)
        errorPositions = random.sample(range(BCH.size), numErrors)

        for position in errorPositions:
            corruptedMessage[position] ^= 1

        decodedMessage, correctedMessage = BCH.decode(corruptedMessage)

        if decodedMessage != message:
            print(f'Fail test {trial}/100')
            break
    else:
        print('All tests passed')
