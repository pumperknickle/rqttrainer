@testable import App
@testable import RQTFoundation
import XCTVapor

final class AppTests: XCTestCase {
    func testHelloWorld() throws {
        let app = Application(.testing)
        defer { app.shutdown() }
        try configure(app)

        try app.test(.GET, "hello", afterResponse: { res in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.body.string, "Hello, world!")
        })
    }
    
    func testCorpusCreation() throws {
        let req1 = RequirementVersionImpl(id: UUID.init().description, text: "BJ Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        let req2 = RequirementVersionImpl(id: UUID.init().description, text: "George Washington went to Washington. Sam Houston stayed home.", source: UUID.init().description, createdAt: Date())
        let tag1 = RequirementTagImpl(id: nil, target: req1.id!, span: (0, 7), attribute: "NER", value: "Person", createdAt: nil)
        let tag2 = RequirementTagImpl(id: nil, target: req2.id!, span: (0, 17), attribute: "NER", value: "Person", createdAt: nil)
        let tag3 = RequirementTagImpl(id: nil, target: req2.id!, span: (25, 35), attribute: "GEO", value: "City", createdAt: nil)
        let corpusString = [req1, req2].createCorpus(tags: [tag1, tag2, tag3], allAttributes: ["NER", "GEO"])
        XCTAssertTrue(corpusString.contains("BJ B-Person O"))
        XCTAssertTrue(corpusString.contains("George B-Person O"))
        XCTAssertTrue(corpusString.contains("Washington I-Person O"))
    }
    
    func testAttachingTags() throws {
        let text = "George Washington went to Washington. Sam Houston stayed home."
        let texts = text.enumerated().map { ($0.offset, $0.element) }
        let tokenizedText = texts.tokenize()
        let tag1 = RequirementTagImpl(id: nil, target: "1", span: (0, 17), attribute: "NER", value: "Person", createdAt: nil)
        let tag2 = RequirementTagImpl(id: nil, target: "2", span: (25, 35), attribute: "NER", value: "City", createdAt: nil)
        let attachedResult = [tag1, tag2].attachAllTagsToWords(words: tokenizedText)
        XCTAssertEqual(attachedResult[0].1.count, 1)
        XCTAssertEqual(attachedResult[1].1.count, 1)
        XCTAssertEqual(attachedResult[4].1.count, 1)
    }
    
    func testTokenization() throws {
        let text = "George Washington went to Washington. Sam Houston stayed home."
        let texts = text.enumerated().map { ($0.offset, $0.element) }.tokenize()
        XCTAssertEqual(texts.first!.first!.0, 0)
        XCTAssertEqual(texts.first!.first!.1, "G")
        XCTAssertEqual(texts[5].first!.0, 36)
        XCTAssertEqual(texts[5].first!.1, ".")
    }
    
    func testTokenizationOfNumber() throws {
        let text = "George Washington went to Washington. Sam Houston stayed home for 25.0 days."
        let texts = text.enumerated().map { ($0.offset, $0.element) }.tokenize()
        XCTAssertEqual(texts.first!.first!.0, 0)
        XCTAssertEqual(texts.first!.first!.1, "G")
        // 12th token contains the decimal point at the 68th index.
        XCTAssertEqual(texts[11][2].0, 68)
        XCTAssertEqual(texts[11][2].1, ".")
    }
//
//    func testTrainEmbedding() throws {
//        let app = Application(.testing)
//        defer { app.shutdown() }
//        try configure(app)
//
//        try app.test(.GET, "train", afterResponse: { (res) in
//            XCTAssertEqual(res.status, .ok)
//            XCTAssertEqual(res.body.string, "trained/")
//        })
//    }
    
    func testPrococesingTokens() throws {
        let text = "George Washington went to Washington."
        let tokens = ["George", "Washington", "went", "to", "Washington", "."]
        let texts = text.enumerated().map { ($0.offset, $0.element) }
        let processedTokens = texts.processTokens(tokens: tokens)
        XCTAssertEqual(processedTokens.first?.first!.1, "G")
    }
    
    func testSegTokTokenize() throws {
        let text = "George Washington went to Washington."
        let tokens = text.segTokTokenize()
        let texts = text.enumerated().map { ($0.offset, $0.element) }
        let processedTokens = texts.processTokens(tokens: tokens)
        XCTAssertEqual(processedTokens.first?.first!.1, "G")
    }
    
    func testSentenceSegmentation() throws {
        let req1 = RequirementVersionImpl(id: UUID.init().description, text: "B.J. Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        XCTAssertTrue([req1].segmentedSentences().count > 1)
    }
}
