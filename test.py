import torch

def makeConnectionMatrix(connections):
	maximum = torch.max(connections)
	minimum = torch.min(connections)
	sideLength = maximum - minimum + 1
	conMatrix = torch.zeros(sideLength, sideLength)

	for j, i in connections:
		conMatrix[j, i] = 1
		conMatrix[i, j] = 1
	return conMatrix

def removeIllegal(parallelGates, connections):
	conMatrix = makeConnectionMatrix(connections=connections)

	s2 = parallelGates + parallelGates.T
	d = s2 == 0
	c2 = (parallelGates != 0)
	d2 = d.logical_and(c2)
	illegalConnections = d2.logical_and(conMatrix == 0)
	legalConnections = d2.logical_and(conMatrix == 1)

	hasIllegal = torch.any(illegalConnections, dim=1)
	hasLegal = torch.any(legalConnections, dim=1)
	allIllegal = hasIllegal.logical_and(hasLegal.logical_not())

	return allIllegal

if __name__ == "__main__":
	connections = torch.tensor(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)]
	)
	gates = torch.tensor([[1, 2, 3, -1, 4, 2, -2, 0, 0]])
	r = removeIllegal(gates, connections)

	print(gates)
	print(r)