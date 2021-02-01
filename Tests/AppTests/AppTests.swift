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
    
    func testTrainEmbedding() throws {
        let app = Application(.testing)
        defer { app.shutdown() }
        try configure(app)
        
        try app.test(.GET, "train", afterResponse: { (res) in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.body.string, "trained/")
        })
    }
    
    func testSentenceSegmentation() throws {
        let req1 = RequirementVersionImpl(id: UUID.init().description, text: "B.J. Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        XCTAssertTrue([req1].segmentedSentences().count > 1)
    }
}
