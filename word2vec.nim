import strutils, tables, strformat, math, sugar, queues, algorithm
import re as re
import arraymancer

let 
    idToWord = newTable[int, string]()
    wordToId = newTable[string, int]()


type 
    WordContext = object
        id: int
        neighbors: seq[int]
    Distance = tuple[dist: float32, word: string]


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


proc rowMax(t: Tensor[float32]): Tensor[float32] =
    result = t[_, _]
    let length = t.shape[1]
    for i in countup(0, t.shape[0] - 1):
        let max = t[i, _].argmax(axis=1).data[0]
        var replacement = zeros[float32](1, length)
        replacement[0, max] = 1.0'f32
        result[i, _] = replacement

# proc glorotInitialization(shape: tuple[a: int, b: int], dimsIn, dimsOut: int): Tensor[float32] = 
#     let value: float32 = sqrt(6.0f) / sqrt((dimsIn + dimsOut).toFloat)
#     result = randomTensor[float32](shape[0], shape[1], 2.toFloat * value) .- value

proc cosine[T](u, v: Tensor[T]): T=
    ## Cosine distance between two rows
    let u_v = (u .- v).reshape(u.shape[1])
    result = dot(u_v, u_v)

proc nearest(embedding: Tensor[float32], word: string, n: int): seq[Distance] =
    if not wordToId.hasKey(word):
        let msg = "Word is not present in embedding."
        raise newException(ValueError, msg) 

    let
        id = wordToId[word]
        targetTensor = embedding[id - 1, _]

    var distances = newSeq[Distance]()
    for k, v in wordToId:
        if v != id:
            let
                comparisonTensor = embedding[v - 1, _]
                dist = cosine(targetTensor, comparisonTensor)

            distances.add((dist, k))

    proc cmp(x, y: Distance): int =
        if x.dist < y.dist:
            return -1
        else:
            return 1
    
    distances = distances.sorted(cmp)
    return distances[0 ..< n]

proc loadData(windowSize: int = 2): seq[WordContext] =


    # http://textproject.org/classroom-materials/textproject-word-pictures/core-vocabulary/animals/
    let rawText = @[
        """All around the world there are big animals, small animals, and all
        sizes in between. Mammals, fish, birds, insects, reptiles,
        amphibians, and other critters are all animals living on Earth. Some
        animals are our friends, while others provide us with food. Let’s
        take a look at some of the core vocabulary words describing the types
        and characteristics of animals. We’ve grouped the words into four
        categories.""",
        
        """There are many different types of animals. Some animals are pets
        and others are not. Some animals are warm blooded (like mammals) and
        others are not (like reptiles and fish). Different animals have
        different characteristics. Some animals have tails and others have
        tusks. There are lots words used to describe animals!""",
        
        """Mammals are animals that have fur and feed their babies milk.
        There are lots of types of mammals all around the world. Some types
        of mammals are our pets, domesticated animals, wild animals, and
        there are even mammals that live in water.""",
        
        """Not all animals have fur. Some animals have scales or feathers,
        like fish and birds. Fish live in water and birds live mostly on
        land. There are also wild birds and domesticated birds.""",
        
        """Some people call them creepy crawlies, but insects, reptiles,
        amphibians, and other similar critters are important to the
        environment. Some are beautiful colors (like many butterflies) and
        others are slimy (like earthworms). These animals can be found all
        around the world.""",
    ]

    # Preparation
    let contexts = prepare(rawText, windowSize)

    return contexts

proc train(epochs: int = 10_000, embeddingSize: int = 5, windowSize: int = 2): Tensor[float32] =
    let 
        contexts = loadData(windowSize)
        (X, y) = makeContextTensors(contexts)
        embeddingSize = 5
        vocabSize = len(wordToId)

        # Specify our context
        ctx = newContext Tensor[float32]
        # Specify our input
        input = ctx.variable(X)

    var output: Tensor[float32]

    network ctx, Word2Vec:
        layers:
            n1: Linear(vocabSize, embeddingSize)
            n2: Linear(embeddingSize, vocabSize)
        forward x:
            let first = x.n1
            output = first.value
            result = first.n2

    let
        model = ctx.init(Word2Vec)
        optim = model.optimizerSGD(learningRate = 0.01'f32)
    

    for epoch in 0..epochs:
        let
            embedding = model.forward(input)
            loss = embedding.softmax_cross_entropy(y)

        if epoch mod 1000 == 0:
            echo "========="
            echo "Epcoch is: " & $epoch
            echo "Loss is: " & $loss.value.data[0]
            let bestGuess = embedding
                                .value
                                .softmax
                                .rowMax
            let
                errors = bestGuess .- y
                total = errors.map(x => x.abs)
                    .sum(axis=1)
                    .sum(axis=0)
                    .data[0]
                percent = (total / (y.shape[0] * y.shape[1]).toFloat) * 100
            echo "Reconstruction errors: " & $percent & "%"

        loss.backprop()
        optim.update()

    return output

when isMainModule:
    let embedding = train(5000, 10, 3)
    for word in @["animals", "insects", "feathers"]:
        echo "Checking nearest neighbors to: " & word
        echo embedding.nearest(word, 3)

        # =========
        # Epcoch is: 0
        # Loss is: 4.855082035064697
        # Reconstruction errors: 2.011531625265531%
        # =========
        # Epcoch is: 1000
        # Loss is: 4.190457820892334
        # Reconstruction errors: 1.794771751853297%
        # =========
        # Epcoch is: 2000
        # Loss is: 3.857608795166016
        # Reconstruction errors: 1.734078987297872%
        # =========
        # Epcoch is: 3000
        # Loss is: 3.616870641708374
        # Reconstruction errors: 1.699397407551914%
        # =========
        # Epcoch is: 4000
        # Loss is: 3.386004686355591
        # Reconstruction errors: 1.630034248059999%
        # =========
        # Epcoch is: 5000
        # Loss is: 3.170977830886841
        # Reconstruction errors: 1.595352668314042%
        # Checking nearest neighbors to: animals
        # @[(dist: 0.4177899360656738, word: "big"), (dist: 2.322341918945312, word: "their"), (dist: 2.644132614135742, word: "our")]
        # Checking nearest neighbors to: insects
        # @[(dist: 2.032831907272339, word: "reptiles"), (dist: 2.070099830627441, word: "birds"), (dist: 3.468972206115723, word: "amphibians")]
        # Checking nearest neighbors to: feathers
        # @[(dist: 0.5408708453178406, word: "land"), (dist: 0.9082614779472351, word: "or"), (dist: 1.213595867156982, word: "scales")]