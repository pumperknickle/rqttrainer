import RQTFoundation
import PythonKit
import Foundation
import Vapor

let sentenceEndMarks = Set<Character>([".", "!", "?"])
let embeddingCorpusPath = "embeddingcorpi/"

public extension String {
    func addSlashIfNeeded() -> String {
        return (last == "/" ? self : self + "/")
    }
    
    func segTokTokenize() -> [String] {
        return Python.import("flair.data").Sentence(self).map { String($0.text)! }
    }
}

public extension Tag {
    func attach(to taggedWords: [([(Int, String.Element)], [Self])]) -> [([(Int, String.Element)], [Self])] {
        guard let firstWord = taggedWords.first else { return [] }
        let firstLetter = firstWord.0.first!
        let lastLetter = firstWord.0.last!
        if left! <= lastLetter.0 && right! >= firstLetter.0 {
            return [(firstWord.0, firstWord.1 + [self])] + attach(to: Array(taggedWords.dropFirst()))
        }
        else {
            return [firstWord] + attach(to: Array(taggedWords.dropFirst()))
        }
    }
}

public extension Array where Element: Tag {
    func mapTagsToTarget() -> [String: [Element]] {
        return Dictionary<String, [Element]>(grouping: self, by: { $0.target.description })
    }
    
    func mapTagsToAttribute() -> [String: Element] {
        return Dictionary<String, Element>(uniqueKeysWithValues: map { ($0.attribute, $0) })
    }
    
    func transformToBio(wordTokens: [(Int, String.Element)], allAttributes: [String]) -> String {
        let attributeToTags = mapTagsToAttribute()
        let word = wordTokens.map { $0.1.description }.reduce("", +)
        let firstIndex = wordTokens.first!.0
        let bioTags = allAttributes.map { (attribute) -> String in
            let tagForAttribute = attributeToTags[attribute]
            guard let tag = tagForAttribute else { return "O" }
            if tag.left == firstIndex { return "B-" + tag.value }
            return "I-" + tag.value
        }
        return ([word] + bioTags).joined(separator: " ")
    }
    
    func attachAllTagsToWords(words: [[(Int, String.Element)]]) -> [([(Int, String.Element)], [Element])] {
        return attachTagsToWords(words: words.map { ($0, []) })
    }
    
    func attachTagsToWords(words: [([(Int, String.Element)], [Element])]) -> [([(Int, String.Element)], [Element])] {
        guard let firstTag = first else { return words }
        return Array(dropFirst()).attachTagsToWords(words: firstTag.attach(to: words))
    }
}


public extension Array where Element == (Int, String.Element) {
    func processTokens(tokens: [String]) -> [[(Int, String.Element)]] {
        guard let firstToken = tokens.first else { return [] }
        if matchesFront(token: firstToken) {
            return [Array(prefix(firstToken.count))] + Array(dropFirst(firstToken.count)).processTokens(tokens: Array<String>(tokens.dropFirst()))
        }
        else { return Array(dropFirst()).processTokens(tokens: tokens) }
    }
    
    func matchesFront(token: String) -> Bool {
        if token.isEmpty { return true }
        guard let tokenFirst = token.first else { return false }
        guard let arrayFirst = first else { return false }
        if tokenFirst == arrayFirst.1 { return Array(dropFirst()).matchesFront(token: String(token.dropFirst())) }
        else { return false }
    }
    
    func tokenize() -> [[(Int, String.Element)]] {
        return tokenize(words: [], currentWord: [])
    }
    
    func tokenize(words: [[(Int, String.Element)]], currentWord: [(Int, String.Element)]) -> [[(Int, String.Element)]] {
        guard let firstToken = first else { return words + (currentWord.count > 0 ? [currentWord] : []) }
        let rest = Array(dropFirst())
        if firstToken.1.isWhitespace { return rest.tokenize(words: words + (currentWord.count > 0 ? [currentWord] : []), currentWord: []) }
        if sentenceEndMarks.contains(firstToken.1) {
            guard let secondToken = rest.first else { return words + (currentWord.count > 0 ? [currentWord] : []) + [[firstToken]] }
            if secondToken.1.isWhitespace { return rest.tokenize(words: words + (currentWord.count > 0 ? [currentWord] : []) + [[firstToken]], currentWord: []) }
            return rest.tokenize(words: words, currentWord: currentWord + [firstToken])
        }
        return rest.tokenize(words: words, currentWord: currentWord + [firstToken])
    }
}

public extension RequirementVersion {
    func createCorpus<T: Tag>(tags: [T], allAttributes: [String]) -> String {
        let words = text.enumerated().map { ($0.offset, $0.element) }.processTokens(tokens: text.segTokTokenize())
        return tags.attachAllTagsToWords(words: words).map { $0.1.transformToBio(wordTokens: $0.0, allAttributes: allAttributes) }.joined(separator: "\n")
    }
}

public extension Sequence where Element: RequirementVersion {
    func computeEmbedding(req: Request, pathToExistingLM: String?, pathToTrainedLM: String, is_forward: Bool = true) throws -> EventLoopFuture<Void> {
        return try saveAsCorpus(req: req, pathToCorpus: embeddingCorpusPath).flatMapThrowing { (_) -> Void in
            try Trainer.computeEmbedding(pathToExistingLM: pathToExistingLM, pathToCorpus: embeddingCorpusPath, pathToTrainedLM: pathToTrainedLM, is_forward: is_forward)
        }
    }
    
    func computeAllModels<T: Tag>(req: Request, tags: [T], pathsToModels: String, pathsToLMs: [String], testSplit: Double = 0.1, devSplit: Double = 0.1) throws -> EventLoopFuture<Void> {
        let allAttributes = Array(Set(tags.map { $0.attribute }))
        let datasets = Python.import("flair.datasets")
        let pathToCorpus = "corpi/"
        let finalSubPath = pathToCorpus.addSlashIfNeeded()
        let finalPaths = pathsToLMs.map { $0.addSlashIfNeeded() }
        let columns = Dictionary<Int, String>(uniqueKeysWithValues: Array((["text"] + allAttributes).enumerated()))
        return saveTaggingCorpus(req: req, tags: tags, allAttributes: allAttributes, pathToCorpus: pathToCorpus, testSplit: testSplit, devSplit: devSplit).flatMap { (_) -> EventLoopFuture<Void> in
            let corpus = datasets.ColumnCorpus(PythonObject(finalSubPath), PythonObject(columns), PythonObject("train.txt"), PythonObject("test.txt"), PythonObject("dev.txt"))
            return allAttributes.compactMap { (element) -> EventLoopFuture<Void> in
                Trainer.computeLabellingModel(for: element, corpus: corpus, pathToTrainedModel: "resources/taggers/" + element + "/", pathsToLMs: finalPaths)
                return req.eventLoop.future()
            }.flatten(on: req.eventLoop)
        }
    }
    
    func createCorpus<T: Tag>(tags: [T], allAttributes: [String]) -> String {
        let targetToTags = tags.mapTagsToTarget()
        return map { $0.createCorpus(tags: $0.id == nil ? [] : targetToTags[$0.id!.description] ?? [], allAttributes: allAttributes) }.joined(separator: "\n\n")
    }
    
    func createCorpusSplit<T: Tag>(tags: [T], allAttributes: [String], testSplit: Double, devSplit: Double) -> (String, String, String) {
        let allReqs = Array(self)
        let testCount = Int(Double(allReqs.count) * testSplit)
        let devCount = Int(Double(allReqs.count) * devSplit)
        let testReqs = allReqs.prefix(testCount)
        let devReqs = allReqs.suffix(devCount)
        let trainReqs = allReqs.dropFirst(testCount).dropLast(devCount)
        return (trainReqs.createCorpus(tags: tags, allAttributes: allAttributes), testReqs.createCorpus(tags: tags, allAttributes: allAttributes), devReqs.createCorpus(tags: tags, allAttributes: allAttributes))
    }
    
    func saveTaggingCorpus<T: Tag>(req: Request, tags: [T], allAttributes: [String], pathToCorpus: String, testSplit: Double = 0.1, devSplit: Double = 0.1) -> EventLoopFuture<Void> {
        let corpi = createCorpusSplit(tags: tags, allAttributes: allAttributes, testSplit: testSplit, devSplit: devSplit)
        let os = Python.import("os")
        let finalSubPath = pathToCorpus.addSlashIfNeeded()
        if !Bool(os.path.exists(finalSubPath))! { os.mkdir(finalSubPath) }
        return req.fileio.writeFile(ByteBuffer(string: corpi.0), at: finalSubPath + "train.txt").flatMap { (_) -> EventLoopFuture<Void> in
            req.fileio.writeFile(ByteBuffer(string: corpi.1), at: finalSubPath + "test.txt").flatMap { (_) -> EventLoopFuture<Void> in
                return req.fileio.writeFile(ByteBuffer(string: corpi.2), at: finalSubPath + "dev.txt")
            }
        }
    }
    
    func saveAsCorpus(req: Request, pathToCorpus: String, maxStringSize: Int=10) throws -> EventLoopFuture<Void> {
        let texts = splitTexts(maxStringSize: maxStringSize)
        let os = Python.import("os")
        let a = Array(texts.dropFirst())
        if a.isEmpty { throw Abort(.badRequest, reason: "Corpus is not large enough") }
        let rest = Array(a.dropFirst())
        if rest.isEmpty { throw Abort(.badRequest, reason: "Corpus is not large enough") }
        let first = texts.first!
        let second = a.first!
        let lastCharInPath = pathToCorpus.last ?? "/"
        let finalSubPath = lastCharInPath == "/" ? pathToCorpus : pathToCorpus + "/"
        if !Bool(os.path.exists(finalSubPath))! { os.mkdir(finalSubPath) }
        if !Bool(os.path.exists(finalSubPath + "train"))! { os.mkdir(finalSubPath + "train") }
        print(finalSubPath)
        return req.fileio.writeFile(ByteBuffer(string: first), at: finalSubPath + "test.txt").flatMap { (_) -> EventLoopFuture<Void> in
            req.fileio.writeFile(ByteBuffer(string: second), at: finalSubPath + "valid.txt").flatMap { (_) -> EventLoopFuture<Void> in
                return zip(Array(0..<rest.count), rest).compactMap { (element) -> EventLoopFuture<Void> in
                    let path = finalSubPath + "train/train_split_" + String(element.0)
                    print(path)
                    return req.fileio.writeFile(ByteBuffer(string: element.1), at: path)
                }.flatten(on: req.eventLoop)
            }
        }
    }

    // Split requirements up into raw strings for embeddings
    func splitTexts(maxStringSize: Int=10) -> [String] {
        return segmentedSentences().stripPunctuationFromEndOfSentences().splitTexts(maxStringSize: maxStringSize)
    }
    
    func segmentedSentences() -> [String] {
        let spacy = Python.import("spacy")
        let nlp = spacy.load("en_core_web_sm")
        return map { nlp($0.text).sents.enumerated().map { String($0.element.text)! } }.reduce([], +)
    }
}

public extension Array where Element == String {
    func splitTexts(final: [String] = [], currentString: String = "", maxStringSize: Int=100000) -> [String] {
        guard let firstSent = first else { return final }
        if currentString.count + firstSent.count + 1 > maxStringSize {
            return Array(dropFirst()).splitTexts(final: final + [currentString + " " + firstSent], maxStringSize: maxStringSize)
        }
        return Array(dropFirst()).splitTexts(final: final, currentString: currentString + " " + firstSent, maxStringSize: maxStringSize)
    }
    
    func stripPunctuationFromEndOfSentences() -> [String] {
        return map { $0.stripPunctuationFromEndOfSentence() }
    }
}

public extension String {
    func stripPunctuationFromEndOfSentence() -> String {
        guard let lastChar = last else { return self }
        if lastChar.isPunctuation { return String(self.dropLast()).stripPunctuationFromEndOfSentence() }
        return self
    }
}

public struct Trainer {
    static func infer(for requirements: [RequirementVersionImpl], for types: [String], pathToTrainedModel: String = "models/") -> [PredictedTagImpl] {
        let dataModule = Python.import("flair.data")
        let modelsModule = Python.import("flair.models")
        return types.map { (tagType) -> [PredictedTagImpl] in
            let fullPath = pathToTrainedModel.addSlashIfNeeded() + tagType.addSlashIfNeeded()
            let sequenceTagger = modelsModule.SequenceTagger.load(fullPath)
            return requirements.map { (requirement) -> [PredictedTagImpl] in
                guard let reqId = requirement.id else { return [] }
                let text = requirement.text
                let tokenizedText = text.segTokTokenize()
                let sentence = dataModule.Sentence(tokenizedText)
                sequenceTagger.predict(sentence)
                let spans = sentence.get_spans()
                return spans.map { (span) -> [(Int, Int, String, Float)] in
                    return span.labels.map { (Int(span.start_pos)!, Int(span.end_pos)!, String($0.value)!, Float($0.score)!) }
                }.reduce([], +).map { PredictedTagImpl(id: nil, target: reqId, span: ($0.0, $0.1), attribute: tagType, value: $0.2, createdAt: Date(), confidence: $0.3) }
            }.reduce([], +)
        }.reduce([], +)
    }
    
    static func computeLabellingModel(for type: String, corpus: PythonObject, pathToTrainedModel: String = "models/", pathsToLMs: [String]) {
        let embeddingsModule = Python.import("flair.embeddings")
        let modelsModule = Python.import("flair.models")
        let trainerModule = Python.import("flair.trainers")
        let tag_type: PythonObject = PythonObject(type)
        let tag_dictionary = corpus.make_tag_dictionary(tag_type)
        let wordEmbeddingType: PythonObject = "glove"
        let embedding_types: PythonObject = PythonObject(pathsToLMs.map { embeddingsModule.FlairEmbeddings(PythonObject($0 + "best-lm.pt")) } + [embeddingsModule.WordEmbeddings(wordEmbeddingType)])
        let embeddings = embeddingsModule.StackedEmbeddings(embedding_types)
        let sequenceTagger = modelsModule.SequenceTagger(PythonObject(256), embeddings, tag_dictionary, tag_type, Python.True)
        let trainer = trainerModule.ModelTrainer(sequenceTagger, corpus)
        let fullPath = pathToTrainedModel.addSlashIfNeeded() + type.addSlashIfNeeded()
        trainer.train(PythonObject(fullPath), PythonObject(0.1), PythonObject(32), PythonObject(150))
    }
    
    // Fine tune embedding at pathToExistingLM or create new Embedding if pathToExistingLM is nil, using pathToCorpus for training data, and storing the resulting embedding at pathToTrainedLM
    static func computeEmbedding(pathToExistingLM: String?, pathToCorpus: String, pathToTrainedLM: String, is_forward: Bool = true) throws {
        print(Python.version)
        let flair = Python.import("flair")
        let lang_trainer = Python.import("flair.trainers.language_model_trainer")
        let existingPath: PythonObject = pathToExistingLM == nil ? Python.None : PythonObject(pathToExistingLM!)
        let corpusPath: PythonObject = PythonObject(pathToCorpus)
        let trainPath: PythonObject = PythonObject(pathToTrainedLM)
        let language_model = pathToExistingLM == nil ? flair.models.LanguageModel(flair.data.Dictionary.load("chars"), PythonObject(is_forward), PythonObject(128), PythonObject(1)): flair.embeddings.FlairEmbeddings(existingPath).lm
        let corpus = lang_trainer.TextCorpus(corpusPath, language_model.dictionary, language_model.is_forward_lm, Python.True)
        let trainer = lang_trainer.LanguageModelTrainer(language_model, corpus)
        trainer.train(trainPath, PythonObject(10), PythonObject(10), PythonObject(10))
    }
}
