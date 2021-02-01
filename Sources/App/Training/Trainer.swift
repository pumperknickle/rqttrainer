import RQTFoundation
import PythonKit
import Foundation
import Vapor

public extension Sequence where Element: RequirementVersion {
    func computeEmbedding(req: Request, pathToExistingLM: String?, pathToTrainedLM: String) throws -> EventLoopFuture<Void> {
        let corpusPath = "corpus/"
        return try saveAsCorpus(req: req, pathToCorpus: corpusPath).flatMapThrowing { (_) -> Void in
            try Trainer.computeEmbedding(pathToExistingLM: pathToExistingLM, pathToCorpus: corpusPath, pathToTrainedLM: pathToTrainedLM)
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
    // Fine tune embedding at pathToExistingLM or create new Embedding if pathToExistingLM is nil, using pathToCorpus for training data, and storing the resulting embedding at pathToTrainedLM
    static func computeEmbedding(pathToExistingLM: String?, pathToCorpus: String, pathToTrainedLM: String) throws {
        print(Python.version)
        let flair = Python.import("flair")
        let existingPath: PythonObject = pathToExistingLM == nil ? Python.None : PythonObject(pathToExistingLM!)
        let corpusPath: PythonObject = PythonObject(pathToCorpus)
        let trainPath: PythonObject = PythonObject(pathToTrainedLM)
        let dictionary = flair.data.Dictionary.load("chars")
        let language_model = pathToExistingLM == nil ? flair.models.LanguageModel(dictionary, Python.True, PythonObject(128), PythonObject(1)): flair.embeddings.FlairEmbeddings(existingPath).lm
        let corpus = flair.trainers.language_model_trainer.TextCorpus(corpusPath, dictionary, language_model.is_forward_lm, Python.True)
        let trainer = flair.trainers.language_model_trainer.LanguageModelTrainer(language_model, corpus)
        trainer.train(trainPath, PythonObject(10), PythonObject(10), PythonObject(10))
    }
}
