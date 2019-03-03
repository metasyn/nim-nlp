import strutils, tables, strformat, math
import re as re
import arraymancer

let 
    idToWord = newTable[int, string]()
    wordToId = newTable[string, int]()


type WordContext = object
    id: int
    neighbors: seq[int]


proc print(ids: seq[int]): string =
    var s = ""
    for id in ids:
        s = s & ", " & idToWord[id]
    return s

proc `$`(w: WordContext): string =
    return fmt"[ID: {idToWord[w.id]} N:{print(w.neighbors)}]"

func regularize(s: string): string =
    result = s.toLower()
    result = re.replace(result, re"[^A-Za-z\s]", "")

proc prepare(input: seq[string], window_size: int): seq[WordContext] =
    result = newSeq[WordContext]()
    var vocabSize: int

    for text in input:
        let 
            cleaned = splitWhitespace(text.regularize())
            length = len(cleaned)

        var numeric = newSeq[int]()
        for idx, word in cleaned:

            # Calculate ID  and add to our lookups
            if not wordToId.hasKey(word):
                let id = len(wordToId) + 1
                idToWord[id] = word
                wordToId[word] = id 

        for idx, word in cleaned:
            # Make a new word context
            var context = WordContext(
                id: wordToId[word],
                neighbors: newSeq[int](),
            )

            let left = max(idx - window_size, 0)
            let right = min(idx + window_size, length)

            for neighborIdx, neighbor in cleaned[left ..< right]:
                if idx != neighborIdx:
                    let neighborId = wordToId[neighbor]
                    context.neighbors.add(neighborId)

            result.add(context)

        vocabSize += length

proc makeOneHotTarget(w: WordContext): Tensor[float32] = 
    result = newTensor[float32](1, len(wordToId))
    result[0, w.id - 1] = 1.tofloat

proc makeOneHotFeature(w: WordContext): Tensor[float32] = 
    result = newTensor[float32](1, len(wordToId))
    for neighbor in w.neighbors:
        result[0, neighbor - 1] = 1.tofloat


proc makeContextTensors(contexts: seq[WordContext]): tuple[X, y:  Tensor[float32]] =
    var y = makeOneHotTarget(contexts[0])
    var X = makeOneHotFeature(contexts[0])

    for i in countup(1, len(contexts) - 1):
        y = y.concat(makeOneHotTarget(contexts[i]), axis=0)
        X = X.concat(makeOneHotFeature(contexts[i]), axis=0)

    return (X, y)


# proc glorotInitialization(shape: tuple[a: int, b: int], dimsIn, dimsOut: int): Tensor[float32] = 
#     let value: float32 = sqrt(6.0f) / sqrt((dimsIn + dimsOut).toFloat)
#     result = randomTensor[float32](shape[0], shape[1], 2.toFloat * value) .- value



let rawText = @[
    "He is the king. The king is royal. She is the royal queen"
]

# Preparation
let contexts = prepare(rawText, 2)

# These hold our mapping between words and ids
echo wordToId

# Make the tensors to train on
let (X, y) = makeContextTensors(contexts)

# Take a look 
echo X[0..5], y[0..5]
echo X.shape, y.shape


# Setup our neuralnet graph context
let
    embeddingSize = 5
    vocabSize = len(wordToId)

let
    # Specify our context
    ctx = newContext Tensor[float32]
    
    # Specify our input
    input = ctx.variable(X)

    embeddingWeights = ctx.variable(
        randomTensor[float32](embeddingSize, vocabSize, 2.0f) .- 1.0f,
        requires_grad=true)

    embeddingBias= ctx.variable(
        randomTensor[float32](1, embeddingSize, 2.0f) .- 1.0f,
        requires_grad=true)

    reconstructionWeights = ctx.variable(
        randomTensor[float32](vocabSize, embeddingSize, 2.0f) .- 1.0f,
        requires_grad=true)

    reconstructionBias= ctx.variable(
        randomTensor[float32](1, vocabSize, 2.0f) .- 1.0f,
        requires_grad=true)

    optim = newSGD[float32](
        embeddingWeights,
        embeddingBias,
        reconstructionWeights,
        reconstructionBias,
        0.01'f32)

    epochs = 10_000

for epoch in 0..epochs:

    let
        n1 = linear(input, embeddingWeights, embeddingBias)
        n2 = linear(n1, reconstructionWeights, reconstructionBias)
        loss = softmax_cross_entropy(n2, y)

    if epoch mod 1000 == 0:
        echo "Epcoch is: " & $epoch
        echo "Loss is: " & $loss.value.data[0]

    loss.backprop()
    optim.update()
